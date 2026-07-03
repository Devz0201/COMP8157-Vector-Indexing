#!/usr/bin/env python3
"""
Command-line entry point for the Vector Indexing Benchmarking Framework.

Usage
-----
    python run_benchmark.py --config configs/scalability.yaml --mode scalability
    python run_benchmark.py --config configs/text_sbert.yaml

The config file controls the data source, embedding backend, index parameters,
and which metrics/charts to produce. See the configs/ directory for examples.
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

from src.benchmark import run_single_scale, run_scalability  # noqa: E402
from src import visualization as viz  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Vector index benchmarking framework")
    ap.add_argument("--config", required=True, help="path to a YAML config file")
    ap.add_argument("--mode", choices=["single", "scalability"], default=None,
                    help="override the mode set in the config")
    ap.add_argument("--outdir", default=None, help="override the results directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.get("mode", "single")
    outdir = args.outdir or cfg.get("outdir", "results")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print(f"Vector Index Benchmark   |   mode={mode}   |   out={outdir}")
    print("=" * 70)

    if mode == "scalability":
        df = run_scalability(cfg)
        csv = viz.write_tables(df, outdir, prefix="scalability_results")
        charts = viz.plot_scalability(df, outdir)
        # Also draw a comparison at the largest scale.
        largest = df["n_vectors"].max()
        charts += viz.plot_comparison(df[df["n_vectors"] == largest].reset_index(drop=True),
                                      outdir)
    else:
        df = run_single_scale(cfg)
        csv = viz.write_tables(df, outdir, prefix="single_results")
        charts = viz.plot_comparison(df, outdir)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nTable : {csv}")
    for c in charts:
        print(f"Chart : {c}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
