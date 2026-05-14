"""Hybrid recommender: collaborative + content-based weighted blend.

For warm users with rich interaction history we rely on the collaborative
model (ALS or two-tower).  For cold users / cold items we lean on the
content recommender.  We blend at the score level with a confidence
weight that decreases as user interaction count drops.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class _Scorer(Protocol):
    def score_items(self, user_id: int) -> np.ndarray: ...  # noqa: E704


class HybridRecommender:
    """Weighted blend of two recommenders.

    Parameters
    ----------
    collab : object with ``user_factors``/``item_factors`` (ALS) or
             ``item_embeddings``/``_user_vec`` (two-tower) attributes.
    content : ContentBasedRecommender
    item_ids : (M,) int array — global item id ordering used by both stores.
    warm_threshold : int
        Below this number of past interactions a user is considered cold and
        content weight is set to ~1.0.
    """

    def __init__(
        self,
        collab,
        content,
        item_ids: np.ndarray,
        warm_threshold: int = 5,
    ) -> None:
        self.collab = collab
        self.content = content
        self.item_ids = item_ids.astype(np.int64)
        self.warm_threshold = warm_threshold

    # ------------------------------------------------------------------ #
    def _collab_scores(self, user_id: int) -> np.ndarray:
        # ALS path
        if hasattr(self.collab, "user_factors") and self.collab.user_factors is not None:
            return self.collab.item_factors @ self.collab.user_factors[user_id]
        # TwoTower path
        if hasattr(self.collab, "item_embeddings") and self.collab.item_embeddings is not None:
            u = self.collab._user_vec(user_id)  # noqa: SLF001
            return self.collab.item_embeddings @ u
        return np.zeros(len(self.item_ids), dtype=np.float32)

    def _content_scores(self, history: list[int]) -> np.ndarray:
        if not history:
            return np.zeros(len(self.item_ids), dtype=np.float32)
        positions = np.array(
            [np.where(self.content.item_ids == i)[0][0] for i in history if (self.content.item_ids == i).any()]
        )
        if positions.size == 0:
            return np.zeros(len(self.item_ids), dtype=np.float32)
        centroid = self.content.item_features[positions].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        return self.content.item_features @ centroid

    # ------------------------------------------------------------------ #
    def _content_weight(self, n_history: int) -> float:
        if n_history <= 0:
            return 1.0
        if n_history >= self.warm_threshold:
            return 0.2
        # Linear ramp between cold and warm
        return 1.0 - 0.8 * (n_history / self.warm_threshold)

    # ------------------------------------------------------------------ #
    def recommend(
        self,
        user_id: int,
        history: list[int] | None = None,
        k: int = 10,
        exclude_history: bool = True,
    ) -> list[int]:
        history = history or []
        w_c = self._content_weight(len(history))
        w_b = 1.0 - w_c

        s_collab = self._collab_scores(user_id)
        s_content = self._content_scores(history)

        # Bring to comparable scale via z-score
        def _z(x: np.ndarray) -> np.ndarray:
            std = x.std() + 1e-9
            return (x - x.mean()) / std

        scores = w_b * _z(s_collab) + w_c * _z(s_content)
        if exclude_history and history:
            mask = np.isin(self.item_ids, history)
            scores[mask] = -np.inf
        return self.item_ids[np.argsort(-scores)[:k]].tolist()
