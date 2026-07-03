"""
Data Ingestion & Preprocessing Layer (Component 1)
==================================================

Loads raw text, cleans/normalises it, and segments long documents into smaller
coherent units. Also provides a ``random`` corpus source that yields synthetic
vectors directly (bypassing text) for large-scale scalability experiments.

Output: a list of clean text segments (the corpus) and a set of held-out query
texts, OR -- for the random source -- the base/query vectors directly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# A small built-in corpus so the framework runs with zero downloads.
# Each line is one semantic unit. Themes are intentionally mixed so that
# nearest-neighbour retrieval is meaningful.
# ---------------------------------------------------------------------------
_SAMPLE_CORPUS = """
The transformer architecture revolutionised natural language processing.
Attention mechanisms allow models to weigh the importance of different tokens.
Large language models are trained on vast corpora of internet text.
Fine-tuning adapts a pre-trained model to a specific downstream task.
Embeddings map words and sentences into continuous vector spaces.
Cosine similarity measures the angle between two embedding vectors.
Approximate nearest neighbour search trades a little accuracy for speed.
FAISS is a library for efficient similarity search over dense vectors.
HNSW builds a multi-layer proximity graph for fast vector retrieval.
The inverted file index partitions vectors into clusters using k-means.
Product quantization compresses vectors to reduce memory consumption.
Recall measures how many true neighbours an index actually returns.
Query latency is the time taken to answer a single search request.
Scalability describes how performance changes as the dataset grows.
Photosynthesis converts sunlight into chemical energy in plants.
Chlorophyll absorbs light most strongly in the blue and red wavelengths.
The mitochondria is the powerhouse of the eukaryotic cell.
DNA stores genetic information using four nucleotide bases.
Enzymes are biological catalysts that speed up chemical reactions.
The water cycle moves moisture between oceans, air, and land.
Stock markets allow companies to raise capital from investors.
A tax-free savings account shelters investment gains from taxation.
Diversification spreads risk across many different assets.
Compound interest grows wealth exponentially over long horizons.
Inflation erodes the purchasing power of money over time.
Central banks adjust interest rates to manage economic growth.
The Roman Empire spanned three continents at its greatest extent.
The printing press accelerated the spread of knowledge in Europe.
The Industrial Revolution transformed manufacturing and society.
World War II reshaped the political map of the twentieth century.
Ancient Egypt built monumental pyramids along the river Nile.
The Renaissance revived art, science, and classical learning.
Basketball players dribble the ball while moving down the court.
Marathon runners train for months to build their endurance.
A balanced diet includes proteins, carbohydrates, and healthy fats.
Regular exercise strengthens the cardiovascular system.
Sleep is essential for memory consolidation and recovery.
Hydration supports nearly every function in the human body.
Jupiter is the largest planet in our solar system.
Black holes have gravity so strong that light cannot escape.
The speed of light in a vacuum is about 300,000 kilometres per second.
Galaxies contain billions of stars bound together by gravity.
Quantum mechanics describes the behaviour of subatomic particles.
Renewable energy sources include solar, wind, and hydro power.
""".strip()


@dataclass
class Corpus:
    """A loaded corpus ready for embedding."""

    documents: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    # When using the 'random' source, vectors are supplied directly:
    base_vectors: Optional[np.ndarray] = None
    query_vectors: Optional[np.ndarray] = None

    @property
    def is_vector_source(self) -> bool:
        return self.base_vectors is not None


def clean_text(text: str) -> str:
    """Normalise whitespace, strip control characters, collapse repeats."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def segment(text: str, max_chars: int = 280) -> List[str]:
    """Split a long document into sentence-ish segments under ``max_chars``."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments, buf = [], ""
    for s in sentences:
        if not s:
            continue
        if len(buf) + len(s) + 1 <= max_chars:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                segments.append(buf)
            buf = s
    if buf:
        segments.append(buf)
    return segments


def _load_text_lines(path: Optional[str]) -> List[str]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    else:
        raw = _SAMPLE_CORPUS
    lines = [clean_text(l) for l in raw.splitlines() if clean_text(l)]
    return lines


def load_corpus(cfg: dict, embedder=None) -> Corpus:
    """Build a Corpus from a data config block.

    Supported sources:
      * ``text``   -> load lines from a file (or the built-in sample), clean,
                      segment, and hold out a fraction as queries.
      * ``random`` -> generate synthetic base/query vectors directly via the
                      SyntheticEmbedder (for large-scale scalability runs).
    """
    source = cfg.get("source", "text").lower()
    rng = np.random.default_rng(cfg.get("seed", 42))

    if source == "random":
        from .embeddings import SyntheticEmbedder

        n_base = int(cfg.get("n_base", 10000))
        n_query = int(cfg.get("n_query", 200))
        dim = int(cfg.get("dim", 384))
        synth = SyntheticEmbedder(dim=dim, n_clusters=cfg.get("n_clusters", 64),
                                  seed=cfg.get("seed", 42))
        base = synth.generate(n_base)
        # Queries: perturb a random subset of base vectors so they have genuine
        # near neighbours in the index (realistic retrieval scenario).
        idx = rng.integers(0, n_base, size=n_query)
        q = base[idx] + rng.normal(0, 0.05, size=(n_query, dim)).astype(np.float32)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        return Corpus(base_vectors=base, query_vectors=q.astype(np.float32))

    # --- text source ---
    lines = _load_text_lines(cfg.get("path"))
    docs: List[str] = []
    for line in lines:
        docs.extend(segment(line, max_chars=cfg.get("max_chars", 280)))

    # Optionally replicate to reach a target size (useful for quick scaling demos
    # on the tiny built-in corpus). Duplicates get a marker so they differ.
    target = int(cfg.get("target_size", len(docs)))
    if target > len(docs):
        base_docs = list(docs)
        i = 0
        while len(docs) < target:
            docs.append(f"{base_docs[i % len(base_docs)]} (v{i // len(base_docs) + 1})")
            i += 1
    docs = docs[:target]

    # Hold out queries.
    n_query = min(int(cfg.get("n_query", 20)), max(1, len(docs) // 5))
    qidx = rng.choice(len(docs), size=n_query, replace=False)
    queries = [docs[i] for i in qidx]

    return Corpus(documents=docs, queries=queries)
