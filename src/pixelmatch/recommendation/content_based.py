"""Content-based recommendation for cold-start scenarios.

When a user has no interaction history, or when scoring brand-new products
with no clicks, we fall back to ranking items by their content-vector
similarity to either:

* the user's previously-interacted item centroid, or
* a single seed product (also useful as a "more like this" endpoint).
"""

from __future__ import annotations

import numpy as np


class ContentBasedRecommender:
    """Score items by cosine similarity over content features."""

    def __init__(self, item_features: np.ndarray, item_ids: np.ndarray) -> None:
        if item_features.shape[0] != item_ids.shape[0]:
            raise ValueError("item_features and item_ids row count must match")
        # L2-normalize for cosine similarity
        norms = np.linalg.norm(item_features, axis=1, keepdims=True) + 1e-12
        self.item_features = (item_features / norms).astype(np.float32)
        self.item_ids = item_ids.astype(np.int64)

    # ------------------------------------------------------------------ #
    def similar_to_item(self, item_id: int, k: int = 10) -> list[int]:
        """Top-k most similar items to a given seed item."""
        pos = np.where(self.item_ids == item_id)[0]
        if pos.size == 0:
            raise KeyError(f"item_id {item_id} not in catalog")
        q = self.item_features[pos[0]]
        scores = self.item_features @ q
        scores[pos[0]] = -np.inf
        return self.item_ids[np.argsort(-scores)[:k]].tolist()

    # ------------------------------------------------------------------ #
    def recommend_from_history(
        self,
        history_item_ids: list[int],
        k: int = 10,
        exclude_history: bool = True,
    ) -> list[int]:
        """Score items by cosine similarity to the centroid of the user's history."""
        if not history_item_ids:
            # No history → return globally popular-by-norm? We return random sample
            # in deterministic order to avoid empty responses.
            return self.item_ids[:k].tolist()
        positions = np.array(
            [np.where(self.item_ids == i)[0][0] for i in history_item_ids if (self.item_ids == i).any()]
        )
        if positions.size == 0:
            return self.item_ids[:k].tolist()
        centroid = self.item_features[positions].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        scores = self.item_features @ centroid
        if exclude_history:
            scores[positions] = -np.inf
        return self.item_ids[np.argsort(-scores)[:k]].tolist()

    # ------------------------------------------------------------------ #
    def cold_start_rank(self, seed_vec: np.ndarray, k: int = 10) -> list[int]:
        """Rank items against an arbitrary query/seed vector."""
        seed_vec = seed_vec / (np.linalg.norm(seed_vec) + 1e-12)
        scores = self.item_features @ seed_vec
        return self.item_ids[np.argsort(-scores)[:k]].tolist()
