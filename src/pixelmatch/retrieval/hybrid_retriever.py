"""Multi-stage hybrid retriever: dense ANN recall + lexical signal + re-ranker.

The retriever follows a two-stage architecture:

1. **Recall stage** — Combine candidates from a FAISS dense index and an
   optional BM25 lexical index using reciprocal-rank fusion (RRF).
2. **Re-rank stage** — Optional callable (e.g. LightGBM LambdaMART) re-orders
   the top ``rerank_window`` candidates with richer feature signals.

This mirrors the architecture used in production search systems (e.g.
Pinterest's PinSage retrieval + Lambda-rank cascade).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pixelmatch.retrieval.bm25_baseline import BM25Retriever
from pixelmatch.retrieval.faiss_index import FaissIndex


@dataclass
class HybridConfig:
    rrf_k: int = 60                   # RRF constant; 60 is the value from Cormack 2009
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    rerank_window: int = 100


class HybridRetriever:
    """Combine dense + lexical recall, optionally followed by a re-ranker."""

    def __init__(
        self,
        dense: FaissIndex,
        lexical: BM25Retriever | None = None,
        config: HybridConfig | None = None,
        reranker: Callable[[list[int], dict], list[int]] | None = None,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.config = config or HybridConfig()
        self.reranker = reranker

    # ------------------------------------------------------------------ #
    def _rrf(
        self,
        dense_ids: np.ndarray,
        lexical_ids: np.ndarray,
    ) -> list[int]:
        scores: dict[int, float] = {}
        for rank, doc_id in enumerate(dense_ids.tolist()):
            scores[doc_id] = scores.get(doc_id, 0.0) + self.config.dense_weight / (
                self.config.rrf_k + rank + 1
            )
        for rank, doc_id in enumerate(lexical_ids.tolist()):
            scores[doc_id] = scores.get(doc_id, 0.0) + self.config.lexical_weight / (
                self.config.rrf_k + rank + 1
            )
        return [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    # ------------------------------------------------------------------ #
    def search(
        self,
        text_query: str | None,
        query_vec: np.ndarray | None,
        k: int = 10,
        rerank_context: dict | None = None,
    ) -> list[int]:
        """Return up to ``k`` document ids."""
        window = self.config.rerank_window
        dense_ids = np.zeros(0, dtype=np.int64)
        lex_ids = np.zeros(0, dtype=np.int64)

        if query_vec is not None:
            ids, _ = self.dense.search(query_vec, k=window)
            dense_ids = ids.ravel()
        if text_query and self.lexical is not None:
            lex_ids, _ = self.lexical.search(text_query, k=window)

        fused = self._rrf(dense_ids, lex_ids)
        if self.reranker is not None and rerank_context is not None:
            fused = self.reranker(fused[:window], rerank_context)
        return fused[:k]
