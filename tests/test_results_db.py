"""
Tests for the results database (Component 6).

Two things are being checked. First, that a run round-trips: what the benchmark
writes is what the CLI reads back. Second, that the schema's own constraints
hold -- the CHECK and UNIQUE clauses exist to stop impossible numbers (a recall
above 1.0, a duplicated measurement) from ever being stored, and a constraint
nobody tests is a constraint nobody can rely on.
"""

import sqlite3

import pytest

from src.results_db import ResultsDB

CONFIG = {
    "k": 10,
    "data": {"source": "text", "path": "data/headlines.txt", "n_query": 200},
    "embedding": {"backend": "sbert", "model_name": "all-MiniLM-L6-v2"},
    "methods": ["flat", "ivf", "hnsw"],
}

MEASUREMENTS = [
    {"method": "Flat", "label": "Flat", "n_vectors": 20000, "k": 10,
     "build_time_s": 0.01, "memory_mb": 29.3, "latency_mean_ms": 1.18,
     "latency_p50_ms": 1.10, "latency_p95_ms": 1.44, "qps": 847.5, "recall_at_k": 1.0},
    {"method": "IVF", "label": "IVF(nlist=100,nprobe=8)", "n_vectors": 20000, "k": 10,
     "build_time_s": 0.15, "memory_mb": 29.6, "latency_mean_ms": 0.21,
     "latency_p50_ms": 0.19, "latency_p95_ms": 0.30, "qps": 4761.9, "recall_at_k": 0.98},
]


@pytest.fixture
def db(tmp_path):
    """A throwaway database file per test, so tests cannot see each other."""
    with ResultsDB(str(tmp_path / "test.db")) as handle:
        yield handle


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_schema_creates_every_table_and_view(db):
    names = {r["name"] for r in db.query(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert {"run", "measurement", "tuning_point", "run_log"} <= names
    assert {"v_results", "v_method_summary", "v_scalability", "v_tuning_curve"} <= names


def test_init_schema_is_safe_to_reapply(db):
    """Every connection reapplies the schema, so this must be a no-op."""
    run_id = db.start_run("single", "configs/text_sbert.yaml", CONFIG)
    db.init_schema()
    assert len(db.query("SELECT 1 FROM run WHERE run_id = ?", (run_id,))) == 1


# ---------------------------------------------------------------------------
# Writing and reading back
# ---------------------------------------------------------------------------
def test_run_records_its_config_and_environment(db):
    run_id = db.start_run("single", "configs/text_sbert.yaml", CONFIG, notes="unit test")
    row = db.query("SELECT * FROM run WHERE run_id = ?", (run_id,))[0]

    assert row["status"] == "running"
    assert row["experiment"] == "single"
    assert row["dataset"] == "data/headlines.txt"
    assert row["embedding_backend"] == "sbert"
    assert row["embedding_model"] == "all-MiniLM-L6-v2"
    assert row["k"] == 10
    assert row["notes"] == "unit test"
    # The config is stored verbatim so a result can always be traced back to it.
    assert "all-MiniLM-L6-v2" in row["config_yaml"]
    # Environment columns are what make two runs comparable at all.
    assert row["python_version"] and row["faiss_version"] and row["cpu_count"]


def test_synthetic_run_labels_its_dataset(db):
    run_id = db.start_run("scalability", "configs/scalability.yaml",
                          {"data": {"source": "random"}, "embedding": {"backend": "synthetic"}})
    row = db.query("SELECT dataset FROM run WHERE run_id = ?", (run_id,))[0]
    assert row["dataset"] == "synthetic"


def test_measurements_round_trip(db):
    run_id = db.start_run("single", "configs/text_sbert.yaml", CONFIG)
    assert db.record_measurements(run_id, MEASUREMENTS) == 2
    db.finish_run(run_id)

    rows = db.query("SELECT * FROM v_results WHERE run_id = ? ORDER BY method", (run_id,))
    assert [r["method"] for r in rows] == ["Flat", "IVF"]
    assert rows[0]["recall_at_k"] == 1.0
    assert rows[1]["label"] == "IVF(nlist=100,nprobe=8)"
    # The view joins run context onto each measurement.
    assert rows[0]["dataset"] == "data/headlines.txt"


def test_finish_run_stamps_status_and_time(db):
    run_id = db.start_run("single", "c.yaml", CONFIG)
    db.finish_run(run_id, "completed")
    row = db.query("SELECT status, finished_at FROM run WHERE run_id = ?", (run_id,))[0]
    assert row["status"] == "completed"
    assert row["finished_at"] is not None


def test_tuning_points_round_trip(db):
    run_id = db.start_run("tuning", "configs/tuning.yaml", CONFIG)
    points = [
        {"method": "IVF", "knob": "nprobe", "value": 1, "recall": 0.62, "latency_ms": 0.05},
        {"method": "IVF", "knob": "nprobe", "value": 8, "recall": 1.0, "latency_ms": 0.21},
        {"method": "HNSW", "knob": "efSearch", "value": 64, "recall": 1.0, "latency_ms": 0.11},
    ]
    assert db.record_tuning(run_id, points) == 3
    db.finish_run(run_id)

    curve = db.query("SELECT * FROM v_tuning_curve WHERE run_id = ?", (run_id,))
    assert len(curve) == 3
    assert {r["method"] for r in curve} == {"IVF", "HNSW"}


def test_logs_are_stored_and_filterable(db):
    run_id = db.start_run("single", "c.yaml", CONFIG)
    db.log(run_id, "INFO", "index", "built HNSW(M=32,efS=64) over 20000 vectors")
    db.log(run_id, "WARN", "evaluate", "HNSW recall below 0.95")
    db.log(run_id, "INFO", "report", "wrote 3 measurements")

    assert len(db.query("SELECT 1 FROM run_log WHERE run_id = ?", (run_id,))) == 3
    warns = db.query("SELECT message FROM run_log WHERE run_id = ? AND level = 'WARN'", (run_id,))
    assert len(warns) == 1 and "below 0.95" in warns[0]["message"]
    # The text search behind `db_cli.py logs --grep`.
    hits = db.query("SELECT 1 FROM run_log WHERE message LIKE ?", ("%HNSW%",))
    assert len(hits) == 2


# ---------------------------------------------------------------------------
# Constraints -- the guardrails the schema is there to provide
# ---------------------------------------------------------------------------
def test_impossible_recall_is_rejected(db):
    run_id = db.start_run("single", "c.yaml", CONFIG)
    bad = dict(MEASUREMENTS[0], recall_at_k=1.4)
    with pytest.raises(sqlite3.IntegrityError):
        db.record_measurements(run_id, [bad])


def test_unknown_method_is_rejected(db):
    run_id = db.start_run("single", "c.yaml", CONFIG)
    bad = dict(MEASUREMENTS[0], method="ScaNN")
    with pytest.raises(sqlite3.IntegrityError):
        db.record_measurements(run_id, [bad])


def test_measurements_cannot_be_orphaned(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.record_measurements(9999, MEASUREMENTS)


def test_deleting_a_run_removes_its_children(db):
    """ON DELETE CASCADE: dropping a bad run must not leave its rows behind."""
    run_id = db.start_run("single", "c.yaml", CONFIG)
    db.record_measurements(run_id, MEASUREMENTS)
    db.log(run_id, "INFO", "index", "something")

    db.conn.execute("DELETE FROM run WHERE run_id = ?", (run_id,))
    db.conn.commit()

    assert db.query("SELECT 1 FROM measurement WHERE run_id = ?", (run_id,)) == []
    assert db.query("SELECT 1 FROM run_log WHERE run_id = ?", (run_id,)) == []


def test_a_method_is_measured_once_per_scale(db):
    """Re-recording the same (run, method, scale) replaces rather than duplicates."""
    run_id = db.start_run("single", "c.yaml", CONFIG)
    db.record_measurements(run_id, MEASUREMENTS)
    db.record_measurements(run_id, MEASUREMENTS)
    rows = db.query("SELECT 1 FROM measurement WHERE run_id = ?", (run_id,))
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Query helpers used by the CLI
# ---------------------------------------------------------------------------
def test_latest_run_id_respects_the_experiment_filter(db):
    first = db.start_run("single", "a.yaml", CONFIG)
    second = db.start_run("scalability", "b.yaml", CONFIG)
    assert db.latest_run_id() == second
    assert db.latest_run_id("single") == first
    assert db.latest_run_id("tuning") is None


def test_method_summary_only_counts_completed_runs(db):
    done = db.start_run("single", "a.yaml", CONFIG)
    db.record_measurements(done, MEASUREMENTS)
    db.finish_run(done, "completed")

    failed = db.start_run("single", "b.yaml", CONFIG)
    db.record_measurements(failed, [dict(MEASUREMENTS[0], n_vectors=5000)])
    db.finish_run(failed, "failed")

    summary = {r["method"]: r for r in db.query("SELECT * FROM v_method_summary")}
    # The failed run's extra Flat row must not inflate the count.
    assert summary["Flat"]["n_measurements"] == 1
