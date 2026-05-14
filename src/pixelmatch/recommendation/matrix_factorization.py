"""Alternating Least Squares (ALS) for implicit-feedback CF.

Implements Hu, Koren, Volinsky 2008 — *Collaborative Filtering for Implicit
Feedback Datasets*.  Written from scratch in numpy/scipy to keep this
project free of the ``implicit`` dependency.

Notation
--------
Given a user-item interaction matrix :math:`R` of shape :math:`(U, I)`
with confidence weights :math:`C = 1 + \\alpha R`, we minimise

.. math::
    \\sum_{u, i} c_{u i} (p_{u i} - x_u^\\top y_i)^2
    + \\lambda \\left( \\sum_u \\|x_u\\|^2 + \\sum_i \\|y_i\\|^2 \\right)

where :math:`p_{u i} = \\mathbb{1}[r_{u i} > 0]` is the binary preference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


@dataclass
class ALSConfig:
    factors: int = 64
    regularization: float = 0.01
    iterations: int = 15
    alpha: float = 40.0
    seed: int = 42


class ALSRecommender:
    """Implicit-feedback ALS.

    Attributes
    ----------
    user_factors : np.ndarray  (U, F)
    item_factors : np.ndarray  (I, F)
    """

    def __init__(self, config: ALSConfig | None = None) -> None:
        self.config = config or ALSConfig()
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self._n_users = 0
        self._n_items = 0

    # ------------------------------------------------------------------ #
    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray | None = None,
        n_users: int | None = None,
        n_items: int | None = None,
    ) -> "ALSRecommender":
        """Fit on a triple of (user, item, rating) arrays."""
        if ratings is None:
            ratings = np.ones_like(user_ids, dtype=np.float32)
        n_users = n_users or int(user_ids.max()) + 1
        n_items = n_items or int(item_ids.max()) + 1
        self._n_users = n_users
        self._n_items = n_items

        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        # Confidence matrix C = 1 + alpha * r
        confidence = sparse.csr_matrix(
            (cfg.alpha * ratings.astype(np.float32), (user_ids, item_ids)),
            shape=(n_users, n_items),
        )
        preference = (confidence > 0).astype(np.float32)
        confidence = confidence + preference  # adds the +1 baseline confidence

        self.user_factors = rng.standard_normal((n_users, cfg.factors)).astype(np.float32) * 0.01
        self.item_factors = rng.standard_normal((n_items, cfg.factors)).astype(np.float32) * 0.01

        reg_eye = cfg.regularization * np.eye(cfg.factors, dtype=np.float32)

        for it in range(cfg.iterations):
            # ---- Solve for user factors ----
            YtY = self.item_factors.T @ self.item_factors
            for u in range(n_users):
                start, end = confidence.indptr[u], confidence.indptr[u + 1]
                if start == end:
                    continue
                items_u = confidence.indices[start:end]
                conf_u = confidence.data[start:end]
                Y_u = self.item_factors[items_u]
                # A = Y^T (C^u - I) Y  + YtY + lambda I
                A = YtY + Y_u.T @ ((conf_u[:, None] - 1.0) * Y_u) + reg_eye
                b = (conf_u[:, None] * Y_u).sum(axis=0)
                self.user_factors[u] = np.linalg.solve(A, b)

            # ---- Solve for item factors ----
            XtX = self.user_factors.T @ self.user_factors
            confidence_T = confidence.T.tocsr()
            for i in range(n_items):
                start, end = confidence_T.indptr[i], confidence_T.indptr[i + 1]
                if start == end:
                    continue
                users_i = confidence_T.indices[start:end]
                conf_i = confidence_T.data[start:end]
                X_i = self.user_factors[users_i]
                A = XtX + X_i.T @ ((conf_i[:, None] - 1.0) * X_i) + reg_eye
                b = (conf_i[:, None] * X_i).sum(axis=0)
                self.item_factors[i] = np.linalg.solve(A, b)
            logger.info("ALS iter %d/%d", it + 1, cfg.iterations)
        return self

    # ------------------------------------------------------------------ #
    def recommend(self, user_id: int, k: int = 10, exclude: set[int] | None = None) -> list[int]:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("ALSRecommender not fitted")
        scores = self.item_factors @ self.user_factors[user_id]
        if exclude:
            scores[list(exclude)] = -np.inf
        return np.argsort(-scores)[:k].tolist()
