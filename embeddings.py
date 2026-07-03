"""
Embedding Generation Layer (Component 2)
========================================

Transforms preprocessed text segments into dense, high-dimensional vectors.

Two backends are provided so the framework runs in any environment:

  * ``sbert``    -> Sentence-BERT (production quality, requires the
                   ``sentence-transformers`` package and a one-time model
                   download). This is the embedder described in the proposal.

  * ``hashing``  -> A deterministic, fully-offline TF-IDF/hashing embedder
                   (scikit-learn). It needs no downloads, so it lets the whole
                   pipeline run on real text anywhere. Quality is lower than
                   SBERT; it exists for demos / CI / air-gapped machines.

The embedding step is kept deterministic across runs so that experiments are
reproducible, as required by the benchmarking methodology.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import List, Sequence

import numpy as np

# Keep benchmark output clean by silencing the harmless "unauthenticated requests
# to the HF Hub" notice. The Hugging Face libraries read these environment
# variables when they initialise their own logging, so they must be set before
# those libraries are imported (the sentence-transformers import below is lazy,
# so setting them here at module load is early enough). Downloads are unaffected.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Normalised vectors let us use inner-product
    search as an exact proxy for cosine similarity."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class BaseEmbedder:
    """Common interface for all embedding backends."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SentenceBERTEmbedder(BaseEmbedder):
    """Production embedder using a pre-trained transformer (Sentence-BERT).

    Default model ``all-MiniLM-L6-v2`` produces 384-dimensional embeddings and
    is a strong speed/quality trade-off for semantic retrieval benchmarks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "sentence-transformers is required for the 'sbert' embedder.\n"
                "Install it with:  pip install sentence-transformers\n"
                "Or switch to the offline 'hashing' embedder in your config."
            ) from exc

        # Now that the HF libraries are imported, re-assert quiet logging through
        # their own verbosity setters so no notices are printed during download.
        for _mod in ("huggingface_hub.utils.logging", "transformers.utils.logging"):
            try:
                __import__(_mod, fromlist=["set_verbosity_error"]).set_verbosity_error()
            except Exception:
                pass

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        # Newer sentence-transformers renamed this method; try the new name
        # first and fall back to the old one for compatibility.
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim = int(self._model.get_embedding_dimension())
        else:
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    @property
    def name(self) -> str:
        return f"SBERT[{self.model_name}]"


class HashingEmbedder(BaseEmbedder):
    """Offline deterministic embedder over real text.

    Uses scikit-learn's HashingVectorizer (character + word n-grams) followed by
    L2 normalisation. No model download, fully reproducible. Lower semantic
    quality than SBERT but enough to exercise the full benchmarking pipeline on
    real text without network access.
    """

    def __init__(self, dim: int = 256, ngram_range: tuple = (1, 2)):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = int(dim)
        self._vec = HashingVectorizer(
            n_features=self.dim,
            ngram_range=ngram_range,
            alternate_sign=False,
            norm=None,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = self._vec.transform(list(texts)).toarray().astype(np.float32)
        return _l2_normalize(mat)

    @property
    def name(self) -> str:
        return f"Hashing[d={self.dim}]"


class SyntheticEmbedder(BaseEmbedder):
    """Generates clustered synthetic embeddings directly (no text needed).

    Produces Gaussian clusters in ``dim`` dimensions. This is the standard way
    to stress-test ANN indexes at large scale and gives clean, controllable
    recall behaviour for scalability experiments. Deterministic via ``seed``.
    """

    def __init__(self, dim: int = 384, n_clusters: int = 64, seed: int = 42):
        self.dim = int(dim)
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)

    def generate(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        centers = rng.normal(0, 1, size=(self.n_clusters, self.dim)).astype(np.float32)
        assign = rng.integers(0, self.n_clusters, size=n)
        noise = rng.normal(0, 0.15, size=(n, self.dim)).astype(np.float32)
        vecs = centers[assign] + noise
        return _l2_normalize(vecs)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # Hash each text to a deterministic vector so identical text -> identical
        # embedding, keeping the interface consistent with the text path.
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little")
            rng = np.random.default_rng(seed)
            out[i] = rng.normal(0, 1, size=self.dim)
        return _l2_normalize(out)

    @property
    def name(self) -> str:
        return f"Synthetic[d={self.dim},k={self.n_clusters}]"


def build_embedder(cfg: dict) -> BaseEmbedder:
    """Factory: construct an embedder from a config dict."""
    backend = cfg.get("backend", "hashing").lower()
    if backend == "sbert":
        return SentenceBERTEmbedder(
            model_name=cfg.get("model_name", "all-MiniLM-L6-v2"),
            batch_size=cfg.get("batch_size", 64),
        )
    if backend == "hashing":
        return HashingEmbedder(dim=cfg.get("dim", 256))
    if backend == "synthetic":
        return SyntheticEmbedder(
            dim=cfg.get("dim", 384),
            n_clusters=cfg.get("n_clusters", 64),
            seed=cfg.get("seed", 42),
        )
    raise ValueError(f"Unknown embedder backend: {backend!r}")
