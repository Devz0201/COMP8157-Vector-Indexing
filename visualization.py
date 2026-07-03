"""
Visualization & Reporting Subsystem
===================================

Generates the comparative graphs and tables described in the proposal:
  * single-scale comparison bars (latency, recall, memory, build time)
  * scalability line charts (each metric vs dataset size)

All figures are written as PNGs to the results directory. Uses a non-interactive
matplotlib backend so it runs headless.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Consistent colour per method across every chart.
_COLORS = {"Flat": "#4C72B0", "IVF": "#DD8452", "HNSW": "#55A868"}


def _color(method: str) -> str:
    return _COLORS.get(method, "#888888")


def plot_comparison(df: pd.DataFrame, outdir: str) -> List[str]:
    """Bar charts comparing methods at a single dataset scale."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    metrics = [
        ("latency_mean_ms", "Mean query latency (ms)  -  lower is better", False),
        ("recall_at_k", "Recall@K  -  higher is better", False),
        ("memory_mb", "Index memory (MB)  -  lower is better", False),
        ("build_time_s", "Index build time (s)  -  lower is better", False),
    ]
    methods = df["method"].tolist()
    colors = [_color(m) for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Vector Index Comparison (single scale)", fontsize=14, fontweight="bold")
    for ax, (col, title, logy) in zip(axes.flat, metrics):
        vals = df[col].tolist()
        bars = ax.bar(methods, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if logy:
            ax.set_yscale("log")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(outdir, "comparison_bars.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # Latency vs recall scatter (the core ANN trade-off).
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in df.iterrows():
        ax.scatter(row["latency_mean_ms"], row["recall_at_k"],
                   s=160, color=_color(row["method"]), zorder=3)
        ax.annotate(row["label"], (row["latency_mean_ms"], row["recall_at_k"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Mean query latency (ms)  ->  slower")
    ax.set_ylabel("Recall@K  ->  more accurate")
    ax.set_title("Accuracy vs Latency trade-off", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "latency_vs_recall.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


def plot_scalability(df: pd.DataFrame, outdir: str) -> List[str]:
    """Line charts: how each metric evolves as the dataset grows."""
    os.makedirs(outdir, exist_ok=True)
    metrics = [
        ("latency_mean_ms", "Mean query latency (ms)"),
        ("recall_at_k", "Recall@K"),
        ("memory_mb", "Index memory (MB)"),
        ("build_time_s", "Index build time (s)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Scalability: metrics vs dataset size", fontsize=14, fontweight="bold")
    for ax, (col, title) in zip(axes.flat, metrics):
        for method, g in df.groupby("method"):
            g = g.sort_values("n_vectors")
            ax.plot(g["n_vectors"], g[col], marker="o",
                    label=method, color=_color(method))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("dataset size (vectors)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(outdir, "scalability.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return [p]


def write_tables(df: pd.DataFrame, outdir: str, prefix: str = "results") -> str:
    """Persist a results table as CSV and return its path."""
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{prefix}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path
