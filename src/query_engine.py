"""
Query Processing Engine (Component 4) + Evaluation Module (Component 5)
======================================================================

The query engine runs the held-out query set against an index under controlled
conditions (warm-up + per-query timing to avoid caching / execution-order bias).

The evaluator computes the proposal's metrics:
  * query latency  (mean / p50 / p95 ms per query)
  * recall@K       (vs the Flat ground truth)
  * memory usage   (serialised index size)
  * build time     (index construction cost)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from .indexing import BaseIndex


@dataclass
class QueryResult:
    indices: np.ndarray          # (n_queries, k) retrieved ids
    latencies_ms: np.ndarray     # per-query latency in milliseconds


def run_queries(index: BaseIndex, queries: np.ndarray, k: int,
                warmup: int = 3, per_query: bool = True) -> QueryResult:
    """Execute queries and record per-query latency.

    A few warm-up queries are run first and discarded so that one-off costs
    (lazy allocation, cache priming) do not pollute the measured latencies.
    """
    # Warm-up
    for _ in range(min(warmup, len(queries))):
        index.search(queries[:1], k)

    if per_query:
        lat = np.empty(len(queries), dtype=np.float64)
        all_idx = np.empty((len(queries), k), dtype=np.int64)
        for i in range(len(queries)):
            q = queries[i:i + 1]
            t0 = time.perf_counter()
            _, idx = index.search(q, k)
            lat[i] = (time.perf_counter() - t0) * 1000.0
            all_idx[i] = idx[0]
        return QueryResult(indices=all_idx, latencies_ms=lat)

    # Batch mode (single call) -- latency is amortised across the batch.
    t0 = time.perf_counter()
    _, idx = index.search(queries, k)
    total_ms = (time.perf_counter() - t0) * 1000.0
    lat = np.full(len(queries), total_ms / len(queries))
    return QueryResult(indices=idx, latencies_ms=lat)


def recall_at_k(approx: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Mean recall@K: fraction of true top-K neighbours retrieved, averaged
    over all queries. ``ground_truth`` is the Flat index's top-K result."""
    recalls = []
    for a_row, g_row in zip(approx, ground_truth):
        gt = set(int(x) for x in g_row[:k] if x >= 0)
        if not gt:
            continue
        got = set(int(x) for x in a_row[:k] if x >= 0)
        recalls.append(len(got & gt) / len(gt))
    return float(np.mean(recalls)) if recalls else 0.0


@dataclass
class MethodMetrics:
    method: str
    label: str
    n_vectors: int
    k: int
    build_time_s: float
    memory_mb: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    qps: float                   # queries per second (1000 / mean latency)
    recall_at_k: float

    def as_row(self) -> Dict:
        return asdict(self)


def evaluate(index: BaseIndex, qres: QueryResult, ground_truth: np.ndarray,
             k: int) -> MethodMetrics:
    """Bundle an index's build stats + query results into one metrics record."""
    lat = qres.latencies_ms
    mean_ms = float(np.mean(lat))
    label = getattr(index, "label", index.name)
    bs = index.build_stats
    return MethodMetrics(
        method=index.name,
        label=label,
        n_vectors=bs.n_vectors,
        k=k,
        build_time_s=round(bs.build_time_s, 5),
        memory_mb=round(bs.memory_bytes / (1024 * 1024), 4),
        latency_mean_ms=round(mean_ms, 5),
        latency_p50_ms=round(float(np.percentile(lat, 50)), 5),
        latency_p95_ms=round(float(np.percentile(lat, 95)), 5),
        qps=round(1000.0 / mean_ms, 2) if mean_ms > 0 else float("inf"),
        recall_at_k=round(recall_at_k(qres.indices, ground_truth, k), 4),
    )
