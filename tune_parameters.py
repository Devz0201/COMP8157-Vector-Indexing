#!/usr/bin/env python3
"""
tune_parameters.py
==================

Parameter-tuning experiment (closes the proposal's "parameter tuning" objective).

It embeds the data ONCE, then sweeps each approximate method's main speed/accuracy
knob and records how recall and latency change:

  * IVF  -> sweep `nprobe`   (how many clusters are searched per query)
  * HNSW -> sweep `efSearch` (how wide the graph search is per query)

Both knobs are query-time settings, so each index is built only once and re-queried
at every setting -- fast and fair.

Usage (from the vecbench folder):
    python tune_parameters.py --config configs/tuning.yaml

Outputs a results table (CSV) and two charts into results/tuning/.
"""

from __future__ import annotations

import argparse
import os
import sys

# Silence the harmless Hugging Face "unauthenticated requests" notice (set before
# any HF library is imported, so it never prints). Downloads are unaffected.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import data_ingestion as di  # noqa: E402
from src import embeddings as emb  # noqa: E402
from src.indexing import FlatIndex, HNSWIndex, IVFIndex  # noqa: E402
from src.query_engine import recall_at_k, run_queries  # noqa: E402
from src.results_db import DEFAULT_DB_PATH, ResultsDB  # noqa: E402

COLOR = {"IVF": "#F59E0B", "HNSW": "#10B981"}
INK = "#1f2937"


def materialize(cfg):
    """Build base + query vectors from the data/embedding config (embed once)."""
    corpus = di.load_corpus(cfg["data"])
    if corpus.is_vector_source:
        return corpus.base_vectors, corpus.query_vectors
    embedder = emb.build_embedder(cfg["embedding"])
    print(f"  embedding {len(corpus.documents):,} documents (one-time)...")
    return embedder.encode(corpus.documents), embedder.encode(corpus.queries)


def main():
    ap = argparse.ArgumentParser(description="Parameter-tuning sweep")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--db", default=DEFAULT_DB_PATH,
                    help=f"results database file (default: {DEFAULT_DB_PATH})")
    ap.add_argument("--no-db", action="store_true",
                    help="do not record this sweep in the results database")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    outdir = args.outdir or cfg.get("outdir", "results/tuning")
    os.makedirs(outdir, exist_ok=True)
    k = cfg.get("k", 10)
    warmup = cfg.get("warmup", 3)

    # Register the sweep in the results database up front (see run_benchmark.py
    # for why: a crashed run should still leave a row and a log behind).
    db = None if args.no_db else ResultsDB(args.db)
    run_id = db.start_run("tuning", args.config, cfg, args.notes) if db else None

    def log(level, stage, message):
        if db:
            db.log(run_id, level, stage, message)

    if db:
        print(f"Recording as run {run_id} in {db.path}")

    base, queries = materialize(cfg)
    dim = base.shape[1]
    print(f"  vectors: {base.shape[0]:,} x {dim}   queries: {queries.shape[0]}   k={k}\n")
    log("INFO", "embed", f"embedded {base.shape[0]} documents once for the whole sweep")

    # Ground truth from exact Flat search (computed once)
    flat = FlatIndex(dim)
    flat.build(base)
    _, gt = flat.search(queries, k)
    log("INFO", "evaluate", f"exact top-{k} ground truth computed from the Flat index")

    sweep = cfg["tuning"]
    rows = []

    # ---- IVF: build once, sweep nprobe (query-time knob) ----
    ivf_p = cfg.get("params", {}).get("ivf", {})
    ivf = IVFIndex(dim, nlist=ivf_p.get("nlist", 100), nprobe=1)
    ivf.build(base)
    print(f"IVF sweep (nlist={ivf.nlist}):")
    log("INFO", "index", f"IVF built once with nlist={ivf.nlist}; sweeping nprobe")
    for nprobe in sweep["ivf_nprobe"]:
        ivf.index.nprobe = int(min(nprobe, ivf.nlist))
        qr = run_queries(ivf, queries, k, warmup=warmup)
        rec = round(recall_at_k(qr.indices, gt, k), 4)
        lat = round(float(np.mean(qr.latencies_ms)), 5)
        rows.append({"method": "IVF", "knob": "nprobe", "value": nprobe,
                     "recall": rec, "latency_ms": lat})
        print(f"  nprobe={nprobe:<4}  recall@{k}={rec:.3f}  latency={lat:.4f} ms")
        log("INFO", "query", f"IVF nprobe={nprobe} recall={rec:.4f} latency={lat:.5f}ms")

    # ---- HNSW: build once, sweep efSearch (query-time knob) ----
    hp = cfg.get("params", {}).get("hnsw", {})
    hnsw = HNSWIndex(dim, M=hp.get("M", 32),
                     ef_construction=hp.get("ef_construction", 80), ef_search=16)
    hnsw.build(base)
    print(f"\nHNSW sweep (M={hnsw.M}):")
    log("INFO", "index", f"HNSW built once with M={hnsw.M}; sweeping efSearch")
    for ef in sweep["hnsw_efsearch"]:
        hnsw.index.hnsw.efSearch = int(ef)
        qr = run_queries(hnsw, queries, k, warmup=warmup)
        rec = round(recall_at_k(qr.indices, gt, k), 4)
        lat = round(float(np.mean(qr.latencies_ms)), 5)
        rows.append({"method": "HNSW", "knob": "efSearch", "value": ef,
                     "recall": rec, "latency_ms": lat})
        print(f"  efSearch={ef:<4}  recall@{k}={rec:.3f}  latency={lat:.4f} ms")
        log("INFO", "query", f"HNSW efSearch={ef} recall={rec:.4f} latency={lat:.5f}ms")

    df = pd.DataFrame(rows)
    csv = os.path.join(outdir, "tuning_results.csv")
    df.to_csv(csv, index=False)

    # ================= charts =================
    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold",
                         "figure.facecolor": "white", "axes.grid": True,
                         "grid.color": "#ececec"})

    # Fig 1: recall vs the knob (two panels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (m, knob) in zip(axes, [("IVF", "nprobe"), ("HNSW", "efSearch")]):
        g = df[df.method == m]
        ax.plot(g["value"], g["recall"], marker="o", color=COLOR[m], linewidth=2.4)
        ax.set_title(f"{m}: accuracy improves as {knob} grows", color=INK)
        ax.set_xlabel(f"{knob}  (search more \u2192)")
        ax.set_ylabel("Recall@10")
        ax.set_xscale("log", base=2)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("Tuning the speed/accuracy knob", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, "tuning_recall.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: the trade-off curve (recall vs latency), points labelled by knob value
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    for m in ["IVF", "HNSW"]:
        g = df[df.method == m].sort_values("latency_ms")
        ax.plot(g["latency_ms"], g["recall"], marker="o", markersize=7,
                color=COLOR[m], linewidth=2.4, label=m)
        for _, r in g.iterrows():
            ax.annotate(str(int(r["value"])), (r["latency_ms"], r["recall"]),
                        textcoords="offset points", xytext=(6, 5),
                        fontsize=8, color=INK)
    ax.set_xlabel("Query latency (ms)  \u2192  slower")
    ax.set_ylabel("Recall@10  \u2192  more accurate")
    ax.set_title("The tuning trade-off: more accuracy costs more time\n"
                 "(numbers on points = the knob value)", color=INK)
    ax.legend(frameon=False, title="method")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tuning_tradeoff.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    if db:
        n = db.record_tuning(run_id, rows)
        db.log(run_id, "INFO", "report", f"wrote {n} tuning points and 2 charts")
        db.finish_run(run_id, "completed")

    print(f"\nTable : {csv}")
    print(f"Chart : {outdir}/tuning_recall.png")
    print(f"Chart : {outdir}/tuning_tradeoff.png")
    if db:
        print(f"\nStored as run {run_id} in {db.path}")
        print(f"Inspect with :  python db_cli.py tuning --run {run_id}")
        db.close()
    print("Done.")


if __name__ == "__main__":
    main()
