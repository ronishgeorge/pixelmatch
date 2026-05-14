"""LambdaMART learning-to-rank re-ranker.

Re-ranks the top-100 candidates returned by the dense+lexical retriever
using LightGBM's ``lambdarank`` objective (Burges 2010 — "From RankNet to
LambdaRank to LambdaMART").

Feature set (intentionally small and inspectable)
-------------------------------------------------
* ``text_sim``       — cosine sim between query text and item text vector
* ``image_sim``      — cosine sim between query image and item image vector
* ``popularity``     — log(1 + interactions) prior
* ``category_match`` — indicator: query category == item category
* ``price_proximity``— exp(-|log(p_q) - log(p_d)| / 0.5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    _HAS_LGB = True
except Exception:  # noqa: BLE001
    lgb = None  # type: ignore
    _HAS_LGB = False


FEATURE_NAMES = ("text_sim", "image_sim", "popularity", "category_match", "price_proximity")


@dataclass
class LTRConfig:
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_data_in_leaf: int = 20
    label_gain: tuple[int, ...] = (0, 1, 3, 7, 15)
    objective: str = "lambdarank"
    metric: str = "ndcg"
    eval_at: tuple[int, ...] = (5, 10)
    seed: int = 42


class _LinearFallback:
    """Tiny logistic-regression-style fallback if LightGBM is unavailable."""

    def __init__(self, dim: int) -> None:
        self.weights = np.array([0.5, 0.3, 0.1, 0.1, 0.05])[:dim]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights


class LambdaMARTRanker:
    """LightGBM-backed LambdaMART re-ranker."""

    def __init__(self, config: LTRConfig | None = None) -> None:
        self.config = config or LTRConfig()
        self.model = None

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        X_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
        group_eval: np.ndarray | None = None,
    ) -> "LambdaMARTRanker":
        """Train LambdaMART.

        Parameters
        ----------
        X : (N, F) features
        y : (N,) integer relevance grades (>=0)
        group : (G,) sizes of each query's candidate list, summing to N
        """
        if not _HAS_LGB:
            logger.warning("LightGBM unavailable — using linear-weight fallback")
            self.model = _LinearFallback(X.shape[1])
            return self

        cfg = self.config
        train = lgb.Dataset(X, label=y, group=group)
        eval_sets = []
        if X_eval is not None and y_eval is not None and group_eval is not None:
            eval_sets.append(lgb.Dataset(X_eval, label=y_eval, group=group_eval, reference=train))

        params = {
            "objective": cfg.objective,
            "metric": cfg.metric,
            "num_leaves": cfg.num_leaves,
            "learning_rate": cfg.learning_rate,
            "min_data_in_leaf": cfg.min_data_in_leaf,
            "label_gain": list(cfg.label_gain),
            "eval_at": list(cfg.eval_at),
            "seed": cfg.seed,
            "verbose": -1,
        }
        self.model = lgb.train(
            params,
            train,
            num_boost_round=cfg.n_estimators,
            valid_sets=eval_sets if eval_sets else None,
        )
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LambdaMARTRanker not fitted")
        return self.model.predict(X)

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_features(
        text_sims: np.ndarray,
        image_sims: np.ndarray,
        popularity: np.ndarray,
        category_match: np.ndarray,
        price_proximity: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack(
            [text_sims, image_sims, popularity, category_match, price_proximity]
        ).astype(np.float32)

    def rerank(
        self,
        candidate_ids: list[int],
        features: np.ndarray,
    ) -> list[int]:
        """Return ``candidate_ids`` reordered by predicted score (desc)."""
        if len(candidate_ids) != features.shape[0]:
            raise ValueError("candidate_ids and features must have same length")
        scores = self.predict(features)
        order = np.argsort(-scores)
        return [candidate_ids[i] for i in order]
