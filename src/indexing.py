"""
Vector Indexing Layer (Component 3)
===================================

Implements the three indexing strategies from the proposal under identical
conditions using FAISS:

  * FlatIndex  -> exhaustive exact search (ground-truth baseline)
  * IVFIndex   -> Inverted File Index, k-means partitioning + nprobe
  * HNSWIndex  -> Hierarchical Navigable Small World graph

All indexes use inner product on L2-normalised vectors, which is equivalent to
cosine similarity. Each index exposes a uniform interface so the query engine
and evaluator treat them identically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import faiss
import numpy as np


@dataclass
class BuildStats:
    build_time_s: float          # index construction (train + add) time
    memory_bytes: int            # serialised index size (memory proxy)
    n_vectors: int


class BaseIndex:
    """Uniform interface for all FAISS index wrappers."""

    name: str = "base"
    index: faiss.Index

    def __init__(self, dim: int):
        self.dim = dim
        self.build_stats: BuildStats | None = None

    def build(self, vectors: np.ndarray) -> BuildStats:
        raise NotImplementedError

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices), shape (n_queries, k)."""
        return self.index.search(queries, k)

    def _finalise(self, vectors: np.ndarray, t0: float) -> BuildStats:
        build_time = time.perf_counter() - t0
        mem = len(faiss.serialize_index(self.index))
        self.build_stats = BuildStats(build_time, int(mem), int(vectors.shape[0]))
        return self.build_stats


class FlatIndex(BaseIndex):
    """Exhaustive exact search. Ground-truth baseline (recall = 1.0)."""

    name = "Flat"

    def build(self, vectors: np.ndarray) -> BuildStats:
        t0 = time.perf_counter()
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        return self._finalise(vectors, t0)


class IVFIndex(BaseIndex):
    """Inverted File Index with k-means partitioning.

    Parameters
    ----------
    nlist  : number of Voronoi cells (clusters) to partition the space into.
    nprobe : number of nearest cells searched at query time (speed/recall knob).
    """

    name = "IVF"

    def __init__(self, dim: int, nlist: int = 100, nprobe: int = 8):
        super().__init__(dim)
        self.nlist = nlist
        self.nprobe = nprobe

    def build(self, vectors: np.ndarray) -> BuildStats:
        t0 = time.perf_counter()
        n = vectors.shape[0]
        # FAISS expects at least ~39 training points per cluster, otherwise it
        # prints a clustering warning. Clamp nlist so that always holds (and so
        # nlist never exceeds the number of points). This makes nlist adapt to
        # very small datasets while keeping the requested value on larger ones.
        nlist = max(1, min(self.nlist, n // 39 if n >= 39 else 1))
        self.nlist = nlist
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist,
                                        faiss.METRIC_INNER_PRODUCT)
        self.index.train(vectors)
        self.index.add(vectors)
        self.index.nprobe = min(self.nprobe, nlist)
        return self._finalise(vectors, t0)

    @property
    def label(self) -> str:
        return f"IVF(nlist={self.nlist},nprobe={self.index.nprobe})"


class HNSWIndex(BaseIndex):
    """Hierarchical Navigable Small World graph index.

    Parameters
    ----------
    M               : number of bi-directional links per node (graph degree).
    ef_construction : breadth of search during graph construction (quality).
    ef_search       : breadth of search at query time (speed/recall knob).
    """

    name = "HNSW"

    def __init__(self, dim: int, M: int = 32, ef_construction: int = 80,
                 ef_search: int = 64):
        super().__init__(dim)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build(self, vectors: np.ndarray) -> BuildStats:
        t0 = time.perf_counter()
        self.index = faiss.IndexHNSWFlat(self.dim, self.M,
                                         faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.add(vectors)
        self.index.hnsw.efSearch = self.ef_search
        return self._finalise(vectors, t0)

    @property
    def label(self) -> str:
        return f"HNSW(M={self.M},efS={self.ef_search})"


def build_index(method: str, dim: int, params: dict) -> BaseIndex:
    """Factory for index objects from a method name + parameter dict."""
    method = method.lower()
    if method == "flat":
        return FlatIndex(dim)
    if method == "ivf":
        return IVFIndex(dim, nlist=params.get("nlist", 100),
                        nprobe=params.get("nprobe", 8))
    if method == "hnsw":
        return HNSWIndex(dim, M=params.get("M", 32),
                         ef_construction=params.get("ef_construction", 80),
                         ef_search=params.get("ef_search", 64))
    raise ValueError(f"Unknown index method: {method!r}")
