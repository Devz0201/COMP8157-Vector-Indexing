"""
Tests for ingestion, embedding, the query engine, and the orchestrator.

The recurring theme here is determinism. The project's headline claim is that
a measured difference between two index types comes from the index and nothing
else, and that only holds if every other stage of the pipeline produces the
same output twice. So most of these tests run something twice and assert the
results are identical.
"""

import numpy as np
import pytest

from src.benchmark import run_single_scale
from src.data_ingestion import clean_text, load_corpus, segment
from src.embeddings import HashingEmbedder, SyntheticEmbedder, build_embedder
from src.indexing import FlatIndex, IVFIndex
from src.query_engine import evaluate, recall_at_k, run_queries


# ---------------------------------------------------------------------------
# Component 1: ingestion
# ---------------------------------------------------------------------------
def test_clean_text_normalises_whitespace():
    assert clean_text("  hello \n  world \t") == "hello world"


def test_segment_respects_the_character_budget():
    text = " ".join(f"Sentence number {i} here." for i in range(40))
    segments = segment(text, max_chars=100)
    assert all(len(s) <= 100 for s in segments)
    # Nothing may be dropped: every original sentence survives somewhere.
    assert "Sentence number 39 here." in " ".join(segments)


def test_segment_keeps_short_text_intact():
    assert segment("One short sentence.", max_chars=280) == ["One short sentence."]


def test_text_corpus_holds_out_queries():
    """Queries come from the corpus, so every query has a true neighbour."""
    corpus = load_corpus({"source": "text", "n_query": 5, "seed": 1})
    assert len(corpus.queries) == 5
    assert all(q in corpus.documents for q in corpus.queries)
    assert not corpus.is_vector_source


def test_random_source_produces_normalised_vectors():
    corpus = load_corpus({"source": "random", "n_base": 500, "n_query": 20,
                          "dim": 32, "seed": 3})
    assert corpus.is_vector_source
    assert corpus.base_vectors.shape == (500, 32)
    assert corpus.query_vectors.shape == (20, 32)
    # Inner-product search is only equivalent to cosine similarity on unit
    # vectors, so this is a correctness precondition, not a style check.
    norms = np.linalg.norm(corpus.base_vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_ingestion_is_reproducible():
    a = load_corpus({"source": "text", "n_query": 5, "seed": 42})
    b = load_corpus({"source": "text", "n_query": 5, "seed": 42})
    assert a.queries == b.queries


# ---------------------------------------------------------------------------
# Component 2: embedding
# ---------------------------------------------------------------------------
def test_synthetic_embeddings_are_byte_identical_across_runs():
    """AC-6 from the requirements document, as an automated check."""
    first = SyntheticEmbedder(dim=64, n_clusters=8, seed=11).generate(300)
    second = SyntheticEmbedder(dim=64, n_clusters=8, seed=11).generate(300)
    assert first.tobytes() == second.tobytes()


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=64)
    texts = ["vector databases store embeddings", "news headlines about sport"]
    first, second = embedder.encode(texts), embedder.encode(texts)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)


def test_embedder_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown embedder backend"):
        build_embedder({"backend": "word2vec"})


def test_embeddings_are_float32():
    """FAISS requires float32; a silent float64 array would fail deep inside
    the C++ layer with a far less obvious error."""
    vecs = SyntheticEmbedder(dim=32, seed=5).generate(100)
    assert vecs.dtype == np.float32


# ---------------------------------------------------------------------------
# Components 4 & 5: query engine and evaluation
# ---------------------------------------------------------------------------
def test_recall_is_one_when_results_match_the_truth():
    truth = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(truth.copy(), truth, 3) == 1.0


def test_recall_counts_partial_overlap():
    truth = np.array([[1, 2, 3, 4]])
    approx = np.array([[1, 2, 9, 9]])
    assert recall_at_k(approx, truth, 4) == 0.5


def test_recall_ignores_the_padding_marker():
    """FAISS pads short result rows with -1; those are absences, not ids."""
    truth = np.array([[1, 2, -1, -1]])
    approx = np.array([[1, 2, -1, -1]])
    assert recall_at_k(approx, truth, 4) == 1.0


def test_evaluate_reports_a_complete_metrics_row():
    vectors = SyntheticEmbedder(dim=32, n_clusters=4, seed=9).generate(800)
    queries = vectors[:25]

    flat = FlatIndex(32)
    flat.build(vectors)
    _, truth = flat.search(queries, 5)

    ivf = IVFIndex(dim=32, nlist=16, nprobe=8)
    ivf.build(vectors)
    result = run_queries(ivf, queries, 5, warmup=2)
    metrics = evaluate(ivf, result, truth, 5)

    assert metrics.method == "IVF"
    assert metrics.n_vectors == 800
    assert metrics.k == 5
    assert 0.0 <= metrics.recall_at_k <= 1.0
    assert metrics.latency_p50_ms <= metrics.latency_p95_ms
    assert metrics.qps > 0
    # as_row() feeds both the CSV writer and the database insert, so its key set
    # is effectively the output schema.
    assert set(metrics.as_row()) == {
        "method", "label", "n_vectors", "k", "build_time_s", "memory_mb",
        "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "qps", "recall_at_k",
    }


def test_per_query_timing_returns_one_latency_per_query():
    vectors = SyntheticEmbedder(dim=32, seed=2).generate(400)
    queries = vectors[:30]
    flat = FlatIndex(32)
    flat.build(vectors)

    per_query = run_queries(flat, queries, 5, warmup=1, per_query=True)
    assert per_query.latencies_ms.shape == (30,)
    # Batch mode amortises one measurement across the batch, so every entry is
    # the same number -- the reason the framework defaults to per-query timing.
    batched = run_queries(flat, queries, 5, warmup=1, per_query=False)
    assert len(set(batched.latencies_ms.tolist())) == 1


# ---------------------------------------------------------------------------
# Orchestration: the five layers wired together
# ---------------------------------------------------------------------------
def test_run_single_scale_compares_all_three_methods():
    cfg = {
        "k": 10, "warmup": 2, "per_query": True,
        "data": {"source": "random", "n_base": 1500, "n_query": 30,
                 "dim": 64, "n_clusters": 8, "seed": 42},
        "embedding": {"backend": "synthetic"},
        "methods": ["flat", "ivf", "hnsw"],
        "params": {"ivf": {"nlist": 32, "nprobe": 8},
                   "hnsw": {"M": 16, "ef_construction": 40, "ef_search": 64}},
    }
    df = run_single_scale(cfg)

    assert list(df["method"]) == ["Flat", "IVF", "HNSW"]
    assert len(df) == 3
    # AC-2: Flat is its own ground truth, so its recall is 1.000 by definition.
    assert df.loc[df["method"] == "Flat", "recall_at_k"].iloc[0] == 1.0
    assert (df["recall_at_k"] <= 1.0).all()
    assert (df["memory_mb"] > 0).all()


def test_run_single_scale_is_reproducible():
    cfg = {
        "k": 5, "warmup": 1,
        "data": {"source": "random", "n_base": 800, "n_query": 20,
                 "dim": 32, "n_clusters": 4, "seed": 42},
        "embedding": {"backend": "synthetic"},
        "methods": ["flat", "ivf"],
        "params": {"ivf": {"nlist": 16, "nprobe": 4}},
    }
    first, second = run_single_scale(cfg), run_single_scale(cfg)
    # Timings vary run to run; recall and memory must not.
    assert list(first["recall_at_k"]) == list(second["recall_at_k"])
    assert list(first["memory_mb"]) == list(second["memory_mb"])


def test_log_callback_receives_pipeline_events():
    """The database writes its run log through this callback, so it has to fire."""
    seen = []
    cfg = {
        "k": 5, "warmup": 1,
        "data": {"source": "random", "n_base": 600, "n_query": 10,
                 "dim": 32, "seed": 42},
        "embedding": {"backend": "synthetic"},
        "methods": ["flat", "ivf"],
        "params": {"ivf": {"nlist": 8, "nprobe": 4}},
    }
    run_single_scale(cfg, lambda level, stage, msg: seen.append((level, stage, msg)))
    stages = {stage for _, stage, _ in seen}
    assert {"ingest", "index", "query", "evaluate"} <= stages
