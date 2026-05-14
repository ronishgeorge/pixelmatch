"""Text encoder backed by Sentence-Transformers.

The encoder exposes a small, dependency-light surface that is easy to swap.
At import time we do *not* download any model weights — that happens lazily
on first encode call, which makes the module safe to import in CI/test
environments without network access.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Iterable, Sequence

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    if x.ndim == 1:
        n = np.linalg.norm(x) + eps
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


class _HashingFallback:
    """Deterministic 384-dim hashing encoder used when sentence-transformers
    is unavailable (offline CI, tests).  Produces stable but semantically
    weak embeddings — fine for sanity tests, not for production.
    """

    def __init__(self, dim: int = 384, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for token in t.lower().split():
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                vecs[i, idx] += 1.0
        return vecs


class TextEncoder:
    """Encode text into dense L2-normalized vectors.

    Parameters
    ----------
    model_name : str
        HuggingFace / sentence-transformers model id.
    dim : int
        Expected output dimensionality (used by the offline fallback).
    cache_dir : str, optional
        Directory used by :mod:`joblib.Memory` to persist encoded batches
        keyed by ``sha1(text)``.  ``None`` disables caching.
    device : str
        ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
        cache_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.device = device
        self._model = None
        self._cache_dir = cache_dir
        self._memory = (
            joblib.Memory(location=cache_dir, verbose=0) if cache_dir else None
        )
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Update dim from real model
            self.dim = self._model.get_sentence_embedding_dimension()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load %s (%s) — falling back to hashing encoder.",
                self.model_name,
                exc,
            )
            self._model = _HashingFallback(dim=self.dim)
        return self._model

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode one or many texts.

        Always returns a 2-D ``(n, dim)`` float32 array.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        texts = list(texts)

        model = self._load()
        try:
            vecs = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        except TypeError:
            # _HashingFallback signature
            vecs = model.encode(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        if normalize:
            vecs = _l2_normalize(vecs)
        return vecs

    def encode_iter(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode a (potentially large) iterable, batching internally."""
        buf: list[str] = []
        chunks: list[np.ndarray] = []
        for t in texts:
            buf.append(t)
            if len(buf) >= batch_size:
                chunks.append(self.encode(buf, batch_size=batch_size))
                buf.clear()
        if buf:
            chunks.append(self.encode(buf, batch_size=batch_size))
        if not chunks:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack(chunks)
