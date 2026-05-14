"""BM25 baseline implemented from scratch in numpy.

Reference: Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25
and Beyond*, FnTIR 2009.

We deliberately avoid the ``rank_bm25`` dependency so the module is
self-contained and trivially auditable.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alpha-numeric tokenization."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    r"""Okapi BM25 retriever.

    Score for query ``Q`` and document ``D``:

    .. math::
        \text{score}(D, Q) = \sum_{t \in Q} \text{IDF}(t)
            \cdot \frac{f(t, D) \cdot (k_1 + 1)}
                       {f(t, D) + k_1 \cdot (1 - b + b \cdot |D| / \mathrm{avgdl})}
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._fitted = False

        self._docs: list[list[str]] = []
        self._doc_freqs: list[Counter] = []
        self._doc_lengths: np.ndarray = np.zeros(0, dtype=np.float32)
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._ids: np.ndarray = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------ #
    def fit(self, documents: Iterable[str], ids: Iterable[int] | None = None) -> "BM25Retriever":
        docs = [tokenize(d) for d in documents]
        self._docs = docs
        self._doc_freqs = [Counter(d) for d in docs]
        self._doc_lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self._avgdl = float(self._doc_lengths.mean()) if len(docs) else 0.0
        self._ids = (
            np.asarray(list(ids), dtype=np.int64)
            if ids is not None
            else np.arange(len(docs), dtype=np.int64)
        )

        # IDF (with the +0.5 / +0.5 smoothing of Robertson)
        n = len(docs)
        df: Counter = Counter()
        for tokens in docs:
            for tok in set(tokens):
                df[tok] += 1
        self._idf = {
            tok: float(np.log(1.0 + (n - d + 0.5) / (d + 0.5))) for tok, d in df.items()
        }
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def _score_all(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self._docs), dtype=np.float32)
        for tok in query_tokens:
            idf = self._idf.get(tok)
            if idf is None:
                continue
            for i, freqs in enumerate(self._doc_freqs):
                f = freqs.get(tok)
                if not f:
                    continue
                dl = self._doc_lengths[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                scores[i] += idf * (f * (self.k1 + 1) / denom)
        return scores

    def search(self, query: str, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("BM25Retriever not fitted")
        toks = tokenize(query)
        scores = self._score_all(toks)
        k_eff = min(k, len(scores))
        top = np.argpartition(-scores, k_eff - 1)[:k_eff] if k_eff < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return self._ids[top], scores[top]
