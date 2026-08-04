"""
Tests for the indexing layer (Component 3).

These check the properties the rest of the framework relies on: that all three
index types really do expose the same interface, that Flat is exact (which is
what makes it usable as ground truth), and that the adaptive `nlist` clamp
actually holds at the small end of the scalability sweep -- that last one is
the fix for the k-means training issue recorded in the D.3.1 risk register, so
it gets a test rather than a comment.
"""

import numpy as np
import pytest

from src.embeddings import SyntheticEmbedder
from src.indexing import BaseIndex, FlatIndex, HNSWIndex, IVFIndex, build_index


@pytest.fixture
def vectors():
    """A small clustered dataset, deterministic across runs."""
    return SyntheticEmbedder(dim=64, n_clusters=8, seed=7).generate(2000)


@pytest.fixture
def queries(vectors):
    """Queries drawn from the base set, so each has an obvious neighbour."""
    return vectors[:50]


@pytest.mark.parametrize("method", ["flat", "ivf", "hnsw"])
def test_factory_returns_a_working_index(method, vectors, queries):
    """Every registered method builds, searches, and reports build stats."""
    index = build_index(method, vectors.shape[1], {})
    assert isinstance(index, BaseIndex)

    stats = index.build(vectors)
    assert stats.n_vectors == vectors.shape[0]
    assert stats.build_time_s >= 0
    assert stats.memory_bytes > 0

    distances, indices = index.search(queries, 10)
    assert indices.shape == (len(queries), 10)
    assert distances.shape == (len(queries), 10)


def test_factory_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown index method"):
        build_index("scann", 64, {})


def test_flat_search_is_exact(vectors, queries):
    """Flat must return each query itself as its own nearest neighbour.

    This is the property the whole evaluation rests on: if Flat were not exact,
    every recall@K number in the project would be measured against the wrong
    reference.
    """
    flat = FlatIndex(vectors.shape[1])
    flat.build(vectors)
    _, idx = flat.search(queries, 1)
    assert np.array_equal(idx.ravel(), np.arange(len(queries)))


def test_ivf_clamps_nlist_on_small_datasets():
    """nlist must drop below the requested value when there is too little data.

    FAISS wants roughly 39 training points per cluster. With 200 vectors and a
    requested nlist of 128 the clamp should bring it down to 5.
    """
    small = SyntheticEmbedder(dim=32, n_clusters=4, seed=1).generate(200)
    ivf = IVFIndex(dim=32, nlist=128, nprobe=8)
    ivf.build(small)
    assert ivf.nlist == 200 // 39
    # nprobe can never exceed the number of cells that actually exist.
    assert ivf.index.nprobe <= ivf.nlist


def test_ivf_keeps_requested_nlist_when_data_is_plentiful(vectors):
    """At 2,000 vectors the ceiling is 51, so a request for 32 is honoured."""
    ivf = IVFIndex(dim=vectors.shape[1], nlist=32, nprobe=4)
    ivf.build(vectors)
    assert ivf.nlist == 32


def test_ivf_recall_improves_with_nprobe(vectors, queries):
    """Searching more cells cannot retrieve fewer of the true neighbours."""
    from src.query_engine import recall_at_k

    flat = FlatIndex(vectors.shape[1])
    flat.build(vectors)
    _, truth = flat.search(queries, 10)

    ivf = IVFIndex(dim=vectors.shape[1], nlist=32, nprobe=1)
    ivf.build(vectors)
    _, low = ivf.search(queries, 10)
    ivf.index.nprobe = 32
    _, high = ivf.search(queries, 10)

    assert recall_at_k(high, truth, 10) >= recall_at_k(low, truth, 10)


def test_hnsw_parameters_reach_the_underlying_index(vectors):
    """The YAML knobs must actually be applied, not just stored on the wrapper."""
    hnsw = HNSWIndex(dim=vectors.shape[1], M=16, ef_construction=40, ef_search=48)
    hnsw.build(vectors)
    assert hnsw.index.hnsw.efSearch == 48
    assert "M=16" in hnsw.label and "efS=48" in hnsw.label


def test_labels_carry_the_parameters(vectors):
    """Result rows are identified by label, so the label has to be specific
    enough to tell two settings of the same method apart."""
    ivf = IVFIndex(dim=vectors.shape[1], nlist=32, nprobe=4)
    ivf.build(vectors)
    assert ivf.label == "IVF(nlist=32,nprobe=4)"
    assert FlatIndex(vectors.shape[1]).name == "Flat"
