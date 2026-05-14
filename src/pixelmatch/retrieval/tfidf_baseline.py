"""TF-IDF retrieval baseline built on scikit-learn."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfRetriever:
    """Classical TF-IDF cosine-similarity retriever.

    Acts as a non-trivial lexical baseline against which dense embeddings
    should clearly outperform on a retrieval-quality metric like NDCG@10.
    """

    def __init__(self, max_features: int = 50_000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self._vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self._matrix = None
        self._ids: np.ndarray = np.zeros(0, dtype=np.int64)

    def fit(self, documents: Iterable[str], ids: Iterable[int] | None = None) -> "TfidfRetriever":
        documents = list(documents)
        self._matrix = self._vec.fit_transform(documents)
        self._ids = (
            np.asarray(list(ids), dtype=np.int64)
            if ids is not None
            else np.arange(len(documents), dtype=np.int64)
        )
        return self

    def search(self, query: str, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix is None:
            raise RuntimeError("TfidfRetriever not fitted")
        q = self._vec.transform([query])
        sims = linear_kernel(q, self._matrix).ravel().astype(np.float32)
        k_eff = min(k, sims.shape[0])
        top = np.argpartition(-sims, k_eff - 1)[:k_eff] if k_eff < sims.shape[0] else np.arange(sims.shape[0])
        top = top[np.argsort(-sims[top])]
        return self._ids[top], sims[top]
