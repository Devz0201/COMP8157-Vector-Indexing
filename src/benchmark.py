"""
Benchmark Orchestrator
======================

Wires the five layers together into a reproducible experiment:

  data ingestion -> embedding -> indexing -> query engine -> evaluation

Provides two entry points:
  * run_single_scale(cfg)  -> compare Flat/IVF/HNSW at one dataset size
  * run_scalability(cfg)   -> sweep dataset sizes and record the trend

Both accept an optional ``log`` callback of the form ``log(level, stage, msg)``.
The CLI passes one that writes into the run's ``run_log`` table, which is how a
run's execution history ends up stored next to the numbers it produced instead
of scrolling off the terminal. When no callback is given the functions run
exactly as before, so the module stays usable from a plain Python shell.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import data_ingestion as di
from . import embeddings as emb
from .indexing import FlatIndex, build_index
from .query_engine import MethodMetrics, evaluate, run_queries

# A log sink: (level, stage, message) -> None.
LogFn = Callable[[str, str, str], None]


def _noop_log(level: str, stage: str, message: str) -> None:
    """Default sink: swallow log lines when no database run is attached."""


def _materialise_vectors(cfg: dict, log: LogFn = _noop_log) -> Tuple[np.ndarray, np.ndarray]:
    """Produce (base_vectors, query_vectors) from the data + embedding config."""
    data_cfg = cfg["data"]
    corpus = di.load_corpus(data_cfg)

    if corpus.is_vector_source:
        log("INFO", "ingest",
            f"synthetic source: {corpus.base_vectors.shape[0]} base vectors, "
            f"{corpus.query_vectors.shape[0]} queries")
        return corpus.base_vectors, corpus.query_vectors

    log("INFO", "ingest",
        f"loaded {len(corpus.documents)} documents from "
        f"{data_cfg.get('path') or 'the built-in sample corpus'}, "
        f"holding out {len(corpus.queries)} queries")
    embedder = emb.build_embedder(cfg["embedding"])
    log("INFO", "embed", f"embedding with {embedder.name}")
    base = embedder.encode(corpus.documents)
    queries = embedder.encode(corpus.queries)
    log("INFO", "embed", f"produced {base.shape[0]} x {base.shape[1]} base embeddings")
    return base, queries


def _ground_truth(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact top-K via a Flat index = the recall reference."""
    flat = FlatIndex(base.shape[1])
    flat.build(base)
    _, gt = flat.search(queries, k)
    return gt


def run_single_scale(cfg: dict, log: Optional[LogFn] = None) -> pd.DataFrame:
    """Build every requested index on the same data and evaluate them."""
    log = log or _noop_log
    k = cfg.get("k", 10)
    base, queries = _materialise_vectors(cfg, log)
    dim = base.shape[1]
    print(f"  vectors: {base.shape[0]:,} x {dim}   queries: {queries.shape[0]}   k={k}")

    ground_truth = _ground_truth(base, queries, k)
    log("INFO", "evaluate", f"exact top-{k} ground truth computed from the Flat index")

    rows: List[Dict] = []
    for method in cfg.get("methods", ["flat", "ivf", "hnsw"]):
        params = cfg.get("params", {}).get(method, {})
        index = build_index(method, dim, params)
        index.build(base)
        log("INFO", "index",
            f"built {getattr(index, 'label', index.name)} over {base.shape[0]} vectors "
            f"in {index.build_stats.build_time_s:.4f}s")
        qres = run_queries(index, queries, k,
                           warmup=cfg.get("warmup", 3),
                           per_query=cfg.get("per_query", True))
        m: MethodMetrics = evaluate(index, qres, ground_truth, k)
        rows.append(m.as_row())
        print(f"    {m.label:<26} latency={m.latency_mean_ms:8.4f} ms  "
              f"recall@{k}={m.recall_at_k:.3f}  mem={m.memory_mb:7.3f} MB  "
              f"build={m.build_time_s:.4f}s")
        log("INFO", "query",
            f"{m.label} n={m.n_vectors} latency={m.latency_mean_ms:.5f}ms "
            f"p95={m.latency_p95_ms:.5f}ms recall@{k}={m.recall_at_k:.4f} "
            f"mem={m.memory_mb:.3f}MB")
        # An approximate index falling well below the Flat baseline is not an
        # error, but it is the single most common reason a result looks wrong,
        # so it is flagged in the log where a user will actually find it.
        if m.method != "Flat" and m.recall_at_k < 0.95:
            log("WARN", "evaluate",
                f"{m.label} recall@{k}={m.recall_at_k:.4f} is below 0.95 at "
                f"n={m.n_vectors}; raise nprobe (IVF) or ef_search (HNSW) to recover it")
    return pd.DataFrame(rows)


def run_scalability(cfg: dict, log: Optional[LogFn] = None) -> pd.DataFrame:
    """Repeat the single-scale benchmark across a list of dataset sizes."""
    log = log or _noop_log
    sizes = cfg["scalability"]["sizes"]
    all_rows: List[Dict] = []
    log("INFO", "ingest", f"scalability sweep over sizes {sizes}")

    for n in sizes:
        print(f"\n[scale] dataset size = {n:,}")
        log("INFO", "index", f"--- scale {n} ---")
        scale_cfg = {**cfg}
        data_cfg = {**cfg["data"]}
        # Resize the dataset depending on the data source.
        if data_cfg.get("source", "text") == "random":
            data_cfg["n_base"] = n
        else:
            data_cfg["target_size"] = n
        scale_cfg["data"] = data_cfg
        df = run_single_scale(scale_cfg, log)
        all_rows.extend(df.to_dict("records"))

    return pd.DataFrame(all_rows)
