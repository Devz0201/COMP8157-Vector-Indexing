"""
Results Persistence Layer (Component 6)
=======================================

Every benchmark run writes its metrics twice: once as the CSV/PNG artefacts
described in the proposal, and once into a small SQLite database.

The CSVs are the per-run deliverable -- easy to open, easy to hand to a
grader. The database is what makes runs *comparable*: once the team started
sweeping dataset sizes and tuning knobs, the questions turned into "how does
HNSW at 100k in today's sweep compare with last week's, on the same
machine", and that is a WHERE clause against one table rather than a manual
diff of a directory full of CSVs.

SQLite was chosen deliberately over a client-server engine: it ships inside
the Python standard library, so there is no server for a grader to install,
no credentials to manage, and the whole database is a single file that can be
committed next to the results it describes. The schema lives in
``db/schema.sql`` -- this module is only the Python side of it.

Typical use (the CLI entry points do exactly this)::

    with ResultsDB() as db:
        run_id = db.start_run("scalability", "configs/scalability.yaml", cfg)
        db.log(run_id, "INFO", "index", "building HNSW at n=100000")
        db.record_measurements(run_id, results_dataframe)
        db.finish_run(run_id, "completed")
"""

from __future__ import annotations

import os
import platform
import socket
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

import yaml

# Default location of the database file, relative to the project root. Kept as
# a module constant so the CLI, the benchmark runners, and the tests all agree
# on where "the database" is without passing the path around everywhere.
DEFAULT_DB_PATH = os.path.join("db", "vecbench_results.db")
SCHEMA_PATH = os.path.join("db", "schema.sql")

# Columns of the MethodMetrics dataclass, in the order the measurement table
# expects them. Declared once here so a change to MethodMetrics fails loudly at
# insert time rather than silently shifting values into the wrong columns.
_MEASUREMENT_COLUMNS: Sequence[str] = (
    "method", "label", "n_vectors", "k", "build_time_s", "memory_mb",
    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "qps", "recall_at_k",
)


def _utcnow() -> str:
    """Current UTC time as an ISO-8601 string (what every ts column stores)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _project_root() -> str:
    """Absolute path to the vecbench folder (the parent of this src/ package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _faiss_version() -> str:
    try:
        import faiss
        return str(getattr(faiss, "__version__", "unknown"))
    except Exception:                       # pragma: no cover - faiss always present in practice
        return "unavailable"


class ResultsDB:
    """A thin, dependency-free wrapper around the SQLite results database.

    Deliberately thin: no ORM, no migrations framework, no connection pool.
    The schema is nine columns of numbers and a log table, the writer is a
    single-threaded CLI process, and anything heavier would be more machinery
    than the problem deserves.

    Parameters
    ----------
    path :
        Location of the SQLite file. Relative paths are resolved against the
        project root, so the same default works no matter which directory the
        benchmark was launched from.
    """

    def __init__(self, path: Optional[str] = None):
        raw = path or DEFAULT_DB_PATH
        self.path = raw if os.path.isabs(raw) else os.path.join(_project_root(), raw)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        # Rows come back as mappings so callers can use column names instead of
        # positional indexes -- worth it purely for readability at the call site.
        self.conn.row_factory = sqlite3.Row
        # SQLite disables foreign keys per-connection by default; without this
        # the ON DELETE CASCADE rules in the schema would silently do nothing.
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.init_schema()

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "ResultsDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- schema ----------------------------------------------------------
    def init_schema(self) -> None:
        """Create tables, indexes, and views if they are not already there.

        Every statement in schema.sql is IF NOT EXISTS, so this is safe to call
        on every connection -- which is exactly what makes the database
        self-creating: a grader who clones the repo and runs a benchmark gets a
        valid database without a separate setup step.
        """
        schema_file = os.path.join(_project_root(), SCHEMA_PATH)
        with open(schema_file, "r", encoding="utf-8") as fh:
            self.conn.executescript(fh.read())
        self.conn.commit()

    # -- writing ---------------------------------------------------------
    def start_run(self, experiment: str, config_path: str, cfg: dict,
                  notes: str = "") -> int:
        """Open a new run row and return its ``run_id``.

        The full config is stored verbatim alongside the environment details,
        so a result row can always be traced back to the exact settings and
        machine that produced it -- the reproducibility requirement from D.4.1,
        enforced by the data model rather than by convention.
        """
        data_cfg = cfg.get("data", {}) or {}
        emb_cfg = cfg.get("embedding", {}) or {}
        dataset = data_cfg.get("path") or (
            "synthetic" if data_cfg.get("source") == "random" else "built-in sample corpus"
        )
        cur = self.conn.execute(
            """
            INSERT INTO run (started_at, status, experiment, config_path, config_yaml,
                             dataset, embedding_backend, embedding_model, k, n_queries,
                             hostname, platform, python_version, faiss_version,
                             cpu_count, notes)
            VALUES (?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utcnow(), experiment, config_path,
                yaml.safe_dump(cfg, sort_keys=False),
                dataset,
                emb_cfg.get("backend"),
                emb_cfg.get("model_name"),
                int(cfg.get("k", 10)),
                int(data_cfg.get("n_query", 0) or 0),
                socket.gethostname(),
                f"{platform.system()} {platform.release()}",
                sys.version.split()[0],
                _faiss_version(),
                os.cpu_count(),
                notes,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str = "completed") -> None:
        """Close a run, stamping its end time and final status."""
        self.conn.execute(
            "UPDATE run SET finished_at = ?, status = ? WHERE run_id = ?",
            (_utcnow(), status, run_id),
        )
        self.conn.commit()

    def log(self, run_id: int, level: str, stage: str, message: str,
            echo: bool = False) -> None:
        """Append one line to the run's log.

        ``echo`` mirrors the line to stdout. The runners use it for the handful
        of messages a user watching the terminal should see; everything else is
        written to the database only, and read back later with ``db_cli.py logs``.
        """
        self.conn.execute(
            "INSERT INTO run_log (run_id, ts, level, stage, message) VALUES (?, ?, ?, ?, ?)",
            (run_id, _utcnow(), level, stage, message),
        )
        self.conn.commit()
        if echo:
            print(f"  [{level}] {message}")

    def record_measurements(self, run_id: int, rows: Iterable[dict]) -> int:
        """Insert one measurement row per (method, dataset size).

        Accepts anything dict-like -- in practice ``df.to_dict("records")`` from
        the benchmark's results DataFrame, so the CSV and the database are
        written from one and the same set of numbers and cannot drift apart.
        """
        payload = [
            tuple(r[c] for c in _MEASUREMENT_COLUMNS)
            for r in rows
        ]
        placeholders = ", ".join("?" * (len(_MEASUREMENT_COLUMNS) + 1))
        self.conn.executemany(
            f"INSERT OR REPLACE INTO measurement "
            f"(run_id, {', '.join(_MEASUREMENT_COLUMNS)}) VALUES ({placeholders})",
            [(run_id, *p) for p in payload],
        )
        self.conn.commit()
        return len(payload)

    def record_tuning(self, run_id: int, rows: Iterable[dict]) -> int:
        """Insert the points of a parameter-tuning sweep."""
        payload = [
            (run_id, r["method"], r["knob"], int(r["value"]),
             float(r["recall"]), float(r["latency_ms"]))
            for r in rows
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO tuning_point (run_id, method, knob, value, recall, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            payload,
        )
        self.conn.commit()
        return len(payload)

    # -- reading ---------------------------------------------------------
    def query(self, sql: str, params: Sequence = ()) -> List[sqlite3.Row]:
        """Run an arbitrary read-only query and return all rows.

        Exposed so the CLI, the tests, and anyone poking at the database from a
        Python shell share one entry point rather than each opening their own
        connection.
        """
        return list(self.conn.execute(sql, params).fetchall())

    def latest_run_id(self, experiment: Optional[str] = None) -> Optional[int]:
        """Most recent run id, optionally restricted to one experiment type.

        Used to make ``--run latest`` work in the CLI, which is what a user
        almost always means when they ask to see "the results".
        """
        if experiment:
            rows = self.query(
                "SELECT run_id FROM run WHERE experiment = ? ORDER BY run_id DESC LIMIT 1",
                (experiment,),
            )
        else:
            rows = self.query("SELECT run_id FROM run ORDER BY run_id DESC LIMIT 1")
        return int(rows[0]["run_id"]) if rows else None
