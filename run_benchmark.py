#!/usr/bin/env python3
"""
Command-line entry point for the Vector Indexing Benchmarking Framework.

Usage
-----
    python run_benchmark.py --config configs/scalability.yaml --mode scalability
    python run_benchmark.py --config configs/text_sbert.yaml

The config file controls the data source, embedding backend, index parameters,
and which metrics/charts to produce. See the configs/ directory for examples.

Every run writes its output twice: the CSV table and PNG charts under
``results/`` (the per-run artefact), and a row set in the SQLite results
database under ``db/`` (the cross-run history, browsable with db_cli.py). Pass
``--no-db`` to skip the database entirely.
"""

from __future__ import annotations

import argparse
import os
import sys

# Silence the harmless Hugging Face "unauthenticated requests" notice (set before
# any HF library is imported, so it never prints). Downloads are unaffected.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import yaml

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import visualization as viz  # noqa: E402
from src.benchmark import run_scalability, run_single_scale  # noqa: E402
from src.results_db import DEFAULT_DB_PATH, ResultsDB  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Vector index benchmarking framework")
    ap.add_argument("--config", required=True, help="path to a YAML config file")
    ap.add_argument("--mode", choices=["single", "scalability"], default=None,
                    help="override the mode set in the config")
    ap.add_argument("--outdir", default=None, help="override the results directory")
    ap.add_argument("--db", default=DEFAULT_DB_PATH,
                    help=f"results database file (default: {DEFAULT_DB_PATH})")
    ap.add_argument("--no-db", action="store_true",
                    help="do not record this run in the results database")
    ap.add_argument("--notes", default="",
                    help="free-text note stored with the run (e.g. why it was run)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.get("mode", "single")
    outdir = args.outdir or cfg.get("outdir", "results")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print(f"Vector Index Benchmark   |   mode={mode}   |   out={outdir}")
    print("=" * 70)

    # Open the results database and register this run before any work starts, so
    # that a run which crashes half way through still leaves a row (marked
    # 'failed') and a log explaining how far it got.
    db = None if args.no_db else ResultsDB(args.db)
    run_id = db.start_run(mode, args.config, cfg, args.notes) if db else None
    log = (lambda level, stage, msg: db.log(run_id, level, stage, msg)) if db else None
    if db:
        print(f"Recording as run {run_id} in {db.path}")

    try:
        if mode == "scalability":
            df = run_scalability(cfg, log)
            csv = viz.write_tables(df, outdir, prefix="scalability_results")
            charts = viz.plot_scalability(df, outdir)
            # Also draw a comparison at the largest scale.
            largest = df["n_vectors"].max()
            charts += viz.plot_comparison(df[df["n_vectors"] == largest].reset_index(drop=True),
                                          outdir)
        else:
            df = run_single_scale(cfg, log)
            csv = viz.write_tables(df, outdir, prefix="single_results")
            charts = viz.plot_comparison(df, outdir)
    except Exception as exc:
        # Record the failure where the partial results already are, then let the
        # traceback surface normally -- swallowing it would hide the real cause.
        if db:
            db.log(run_id, "ERROR", "report", f"{type(exc).__name__}: {exc}")
            db.finish_run(run_id, "failed")
            db.close()
        raise

    if db:
        n = db.record_measurements(run_id, df.to_dict("records"))
        db.log(run_id, "INFO", "report",
               f"wrote {n} measurements, table {csv}, {len(charts)} charts")
        db.finish_run(run_id, "completed")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nTable : {csv}")
    for c in charts:
        print(f"Chart : {c}")
    if db:
        print(f"\nStored as run {run_id} in {db.path}")
        print(f"Inspect with :  python db_cli.py show {run_id}")
        print(f"Run log      :  python db_cli.py logs --run {run_id}")
        db.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
