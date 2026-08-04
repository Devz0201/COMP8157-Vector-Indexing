#!/usr/bin/env python3
"""
db_cli.py -- inspect the benchmark results database
===================================================

The benchmark runners write every run into ``db/vecbench_results.db``. This is
the read side: a small command-line client for browsing runs, filtering
measurements, and searching the execution logs, so nobody has to install a
SQLite GUI or remember the joins.

Commands
--------
    init      create (or upgrade) the database from db/schema.sql
    runs      list benchmark runs, newest first
    show      full detail of one run, including its config
    results   the measurement rows, with filters
    tuning    the parameter-sweep points, with filters
    logs      the execution log, with level / text / stage filters
    summary   per-method rollup across every completed run
    export    write any of the above out as CSV
    sql       run an ad-hoc read-only SELECT

Examples
--------
    python db_cli.py runs
    python db_cli.py results --run latest --method HNSW
    python db_cli.py results --min-recall 0.95 --max-latency 1.0 --sort latency
    python db_cli.py logs --run latest --level ERROR
    python db_cli.py logs --grep "HNSW" --tail 20
    python db_cli.py export results --run latest --out results/from_db.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.results_db import DEFAULT_DB_PATH, ResultsDB  # noqa: E402


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _fmt(value) -> str:
    """Render one cell. Floats are trimmed to something readable at a glance --
    the database keeps full precision, the terminal does not need it."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def print_table(rows: Sequence[sqlite3.Row], columns: Optional[List[str]] = None,
                max_width: int = 60) -> None:
    """Print rows as an aligned text table.

    Written by hand rather than pulled from a library because the only consumer
    is this CLI and the dependency would buy us nothing the eye can see.
    """
    if not rows:
        print("(no rows)")
        return
    cols = columns or list(rows[0].keys())
    cells = [[_fmt(r[c])[:max_width] for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]

    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''})")


def _resolve_run(db: ResultsDB, run: Optional[str],
                 experiment: Optional[str] = None) -> Optional[int]:
    """Turn a --run argument into a run id.

    Accepts a literal id, the word ``latest``, or nothing at all (meaning "do
    not filter by run"). ``latest`` exists because it is what a user means
    almost every time they ask to see the results.
    """
    if run is None:
        return None
    if str(run).lower() == "latest":
        rid = db.latest_run_id(experiment)
        if rid is None:
            raise SystemExit(f"No {'`' + experiment + '` ' if experiment else ''}runs "
                             f"recorded yet. Run a benchmark first.")
        return rid
    try:
        return int(run)
    except ValueError:
        raise SystemExit(f"--run expects a run id or 'latest', got {run!r}")


# ---------------------------------------------------------------------------
# Query builders -- each returns (sql, params) so `export` can reuse them
# ---------------------------------------------------------------------------
def q_runs(args) -> Tuple[str, list]:
    sql = ("SELECT run_id, started_at, status, experiment, dataset, "
           "embedding_backend, k, hostname FROM run")
    params: list = []
    where = []
    if args.experiment:
        where.append("experiment = ?")
        params.append(args.experiment)
    if args.status:
        where.append("status = ?")
        params.append(args.status)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY run_id DESC"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    return sql, params


def q_results(args, db: ResultsDB) -> Tuple[str, list]:
    """Filtered measurement rows, read through the v_results view.

    Every filter is optional and they compose, so a question like "which
    settings answered in under a millisecond while still holding recall above
    0.95" is one command rather than a spreadsheet sort.
    """
    sql = ("SELECT run_id, experiment, method, label, n_vectors, k, "
           "latency_mean_ms, latency_p95_ms, qps, recall_at_k, memory_mb, build_time_s "
           "FROM v_results")
    where, params = [], []

    run_id = _resolve_run(db, args.run)
    if run_id is not None:
        where.append("run_id = ?")
        params.append(run_id)
    if args.method:
        where.append("UPPER(method) = ?")
        params.append(args.method.upper())
    if args.n_vectors:
        where.append("n_vectors = ?")
        params.append(int(args.n_vectors))
    if args.min_recall is not None:
        where.append("recall_at_k >= ?")
        params.append(float(args.min_recall))
    if args.max_latency is not None:
        where.append("latency_mean_ms <= ?")
        params.append(float(args.max_latency))
    if where:
        sql += " WHERE " + " AND ".join(where)

    order = {"latency": "latency_mean_ms ASC",
             "recall": "recall_at_k DESC",
             "memory": "memory_mb ASC",
             "build": "build_time_s ASC",
             "scale": "n_vectors ASC, method ASC"}[args.sort]
    sql += f" ORDER BY {order}"
    return sql, params


def q_tuning(args, db: ResultsDB) -> Tuple[str, list]:
    sql = "SELECT run_id, method, knob, value, recall, latency_ms FROM v_tuning_curve"
    where, params = [], []
    run_id = _resolve_run(db, args.run, "tuning")
    if run_id is not None:
        where.append("run_id = ?")
        params.append(run_id)
    if args.method:
        where.append("UPPER(method) = ?")
        params.append(args.method.upper())
    if args.min_recall is not None:
        where.append("recall >= ?")
        params.append(float(args.min_recall))
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY method, value"
    return sql, params


def q_logs(args, db: ResultsDB) -> Tuple[str, list]:
    """The execution log, filterable by run, severity, stage, and free text."""
    sql = "SELECT log_id, run_id, ts, level, stage, message FROM run_log"
    where, params = [], []
    run_id = _resolve_run(db, args.run)
    if run_id is not None:
        where.append("run_id = ?")
        params.append(run_id)
    if args.level:
        where.append("level = ?")
        params.append(args.level.upper())
    if args.stage:
        where.append("stage = ?")
        params.append(args.stage)
    if args.grep:
        # Substring search over the message text. LIKE is enough here: the log
        # of a benchmark run is thousands of rows, not millions.
        where.append("message LIKE ?")
        params.append(f"%{args.grep}%")
    if where:
        sql += " WHERE " + " AND ".join(where)
    # Newest-last reads like a terminal, which is what a log should look like.
    sql += " ORDER BY log_id ASC"
    return sql, params


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_init(args, db: ResultsDB) -> int:
    # Opening the connection already applied the schema; just report the state.
    tables = db.query("SELECT name, type FROM sqlite_master "
                      "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
                      "ORDER BY type, name")
    print(f"Database ready: {db.path}")
    print_table(tables)
    return 0


def cmd_runs(args, db: ResultsDB) -> int:
    sql, params = q_runs(args)
    print_table(db.query(sql, params))
    return 0


def cmd_show(args, db: ResultsDB) -> int:
    run_id = _resolve_run(db, args.run or "latest")
    rows = db.query("SELECT * FROM run WHERE run_id = ?", (run_id,))
    if not rows:
        raise SystemExit(f"No run with id {run_id}")
    r = rows[0]

    print("=" * 72)
    print(f"Run {r['run_id']}  |  {r['experiment']}  |  {r['status']}")
    print("=" * 72)
    for key in ("started_at", "finished_at", "config_path", "dataset",
                "embedding_backend", "embedding_model", "k", "n_queries",
                "hostname", "platform", "python_version", "faiss_version",
                "cpu_count", "notes"):
        if r[key] not in (None, "", 0):
            print(f"  {key:<18} {r[key]}")

    print("\n-- measurements " + "-" * 56)
    print_table(db.query(
        "SELECT method, label, n_vectors, latency_mean_ms, latency_p95_ms, "
        "recall_at_k, memory_mb, build_time_s FROM measurement "
        "WHERE run_id = ? ORDER BY n_vectors, method", (run_id,)))

    tuning = db.query("SELECT method, knob, value, recall, latency_ms FROM tuning_point "
                      "WHERE run_id = ? ORDER BY method, value", (run_id,))
    if tuning:
        print("\n-- tuning points " + "-" * 55)
        print_table(tuning)

    counts = db.query("SELECT level, COUNT(*) AS n FROM run_log WHERE run_id = ? "
                      "GROUP BY level ORDER BY level", (run_id,))
    if counts:
        print("\n-- log summary " + "-" * 57)
        print_table(counts)
        print(f"Full log:  python db_cli.py logs --run {run_id}")

    if args.config:
        print("\n-- config " + "-" * 62)
        print(r["config_yaml"])
    return 0


def cmd_results(args, db: ResultsDB) -> int:
    sql, params = q_results(args, db)
    print_table(db.query(sql, params))
    return 0


def cmd_tuning(args, db: ResultsDB) -> int:
    sql, params = q_tuning(args, db)
    print_table(db.query(sql, params))
    return 0


def cmd_logs(args, db: ResultsDB) -> int:
    sql, params = q_logs(args, db)
    rows = db.query(sql, params)
    # --tail keeps the last N, matching what `tail` does to a log file.
    if args.tail:
        rows = rows[-int(args.tail):]
    print_table(rows, max_width=100)
    return 0


def cmd_summary(args, db: ResultsDB) -> int:
    print("Per-method rollup across all completed runs")
    print_table(db.query("SELECT * FROM v_method_summary ORDER BY avg_latency_ms"))
    print("\nLatest scalability sweep")
    print_table(db.query("SELECT * FROM v_scalability"))
    return 0


def cmd_export(args, db: ResultsDB) -> int:
    """Write any of the queryable sets out as a CSV file."""
    builders = {
        "runs": lambda: q_runs(args),
        "results": lambda: q_results(args, db),
        "tuning": lambda: q_tuning(args, db),
        "logs": lambda: q_logs(args, db),
    }
    sql, params = builders[args.what]()
    rows = db.query(sql, params)
    if not rows:
        print("(no rows to export)")
        return 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(rows[0].keys())
        writer.writerows([tuple(r) for r in rows])
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


def cmd_sql(args, db: ResultsDB) -> int:
    """Ad-hoc query escape hatch, restricted to reads.

    Handy for questions the fixed commands do not cover. Writes are refused so
    that browsing the results can never damage them -- a recorded measurement
    is evidence, and evidence should not be editable from a convenience CLI.
    """
    stmt = args.query.strip().rstrip(";")
    if not stmt.lower().startswith(("select", "with", "pragma table_info")):
        raise SystemExit("Only SELECT / WITH queries are allowed here.")
    if ";" in stmt:
        raise SystemExit("One statement at a time, please.")
    print_table(db.query(stmt))
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="db_cli.py",
        description="Browse the vector-index benchmark results database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------\n")[-1],
    )
    ap.add_argument("--db", default=DEFAULT_DB_PATH,
                    help=f"path to the SQLite file (default: {DEFAULT_DB_PATH})")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create or upgrade the database")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("runs", help="list benchmark runs, newest first")
    p.add_argument("--experiment", choices=["single", "scalability", "tuning"])
    p.add_argument("--status", choices=["running", "completed", "failed"])
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("show", help="full detail of one run")
    p.add_argument("run", nargs="?", default="latest",
                   help="run id, or 'latest' (default)")
    p.add_argument("--config", action="store_true", help="also print the run's config")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("results", help="measurement rows, with filters")
    p.add_argument("--run", help="run id or 'latest'")
    p.add_argument("--method", help="Flat | IVF | HNSW")
    p.add_argument("--n-vectors", type=int, help="only this dataset size")
    p.add_argument("--min-recall", type=float, help="only rows at or above this recall")
    p.add_argument("--max-latency", type=float, help="only rows at or below this mean latency (ms)")
    p.add_argument("--sort", default="scale",
                   choices=["scale", "latency", "recall", "memory", "build"])
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("tuning", help="parameter-sweep points")
    p.add_argument("--run", help="run id or 'latest'")
    p.add_argument("--method", help="IVF | HNSW")
    p.add_argument("--min-recall", type=float)
    p.set_defaults(func=cmd_tuning)

    p = sub.add_parser("logs", help="execution log, with filters")
    p.add_argument("--run", help="run id or 'latest'")
    p.add_argument("--level", choices=["DEBUG", "INFO", "WARN", "ERROR"])
    p.add_argument("--stage", help="ingest | embed | index | query | evaluate | report")
    p.add_argument("--grep", help="only lines containing this text")
    p.add_argument("--tail", type=int, help="only the last N lines")
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("summary", help="per-method rollup and latest scalability sweep")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("export", help="write a result set to CSV")
    p.add_argument("what", choices=["runs", "results", "tuning", "logs"])
    p.add_argument("--out", default="results/export.csv")
    # The filters below mirror the individual commands so export can reuse them.
    p.add_argument("--run")
    p.add_argument("--method")
    p.add_argument("--n-vectors", type=int)
    p.add_argument("--min-recall", type=float)
    p.add_argument("--max-latency", type=float)
    p.add_argument("--sort", default="scale",
                   choices=["scale", "latency", "recall", "memory", "build"])
    p.add_argument("--experiment", choices=["single", "scalability", "tuning"])
    p.add_argument("--status", choices=["running", "completed", "failed"])
    p.add_argument("--limit", type=int)
    p.add_argument("--level", choices=["DEBUG", "INFO", "WARN", "ERROR"])
    p.add_argument("--stage")
    p.add_argument("--grep")
    p.add_argument("--tail", type=int)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("sql", help="run an ad-hoc SELECT")
    p.add_argument("query")
    p.set_defaults(func=cmd_sql)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    with ResultsDB(args.db) as db:
        return args.func(args, db)


if __name__ == "__main__":
    raise SystemExit(main())
