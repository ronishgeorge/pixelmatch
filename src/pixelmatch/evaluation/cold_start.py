"""Cold-start evaluation helpers.

We define a *cold product* as one that has fewer than ``min_interactions``
events in the training partition.  The evaluator measures recall@k / MRR
on queries whose ground-truth targets are cold products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from pixelmatch.evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k


@dataclass
class ColdStartSplit:
    train_interactions: pd.DataFrame
    test_interactions: pd.DataFrame
    cold_product_ids: np.ndarray


def make_cold_start_split(
    interactions: pd.DataFrame,
    min_interactions: int = 5,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> ColdStartSplit:
    """Split interactions so that ``cold_product_ids`` carry zero training rows.

    The procedure:
    1. Identify items with < ``min_interactions`` events — these are *cold*.
    2. Remove all rows referencing cold items from the training partition.
    3. Place those rows in the held-out test partition so we can measure
       whether retrieval surfaces cold items.
    """
    rng = np.random.default_rng(seed)
    counts = interactions.groupby("product_id").size()
    cold_ids = counts[counts < min_interactions].index.to_numpy()

    cold_mask = interactions["product_id"].isin(cold_ids)
    cold_rows = interactions[cold_mask]
    warm_rows = interactions[~cold_mask]

    # Sample additional warm rows into test so test isn't 100% cold
    n_warm_test = int(len(warm_rows) * test_fraction)
    if n_warm_test > 0:
        idx = rng.choice(warm_rows.index.to_numpy(), size=n_warm_test, replace=False)
        warm_test = warm_rows.loc[idx]
        warm_train = warm_rows.drop(index=idx)
    else:
        warm_test = warm_rows.iloc[:0]
        warm_train = warm_rows

    train = warm_train
    test = pd.concat([warm_test, cold_rows], ignore_index=True)
    return ColdStartSplit(train_interactions=train, test_interactions=test, cold_product_ids=cold_ids)


# ---------------------------------------------------------------------- #
def evaluate_cold_start(
    query_fn: Callable[[int, int], list[int]],
    test_pairs: list[tuple[int, int]],
    cold_product_ids: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a retrieval function on cold queries.

    Parameters
    ----------
    query_fn : ``(user_id, k) -> list[product_id]``
    test_pairs : list of ``(user_id, gold_product_id)`` tuples.
    cold_product_ids : ids considered cold.
    """
    cold_set = set(cold_product_ids.tolist())
    cold_pairs = [(u, p) for (u, p) in test_pairs if p in cold_set]
    if not cold_pairs:
        return {"n_cold_pairs": 0, f"recall@{k}": 0.0, "mrr": 0.0, f"ndcg@{k}": 0.0}

    preds = [query_fn(u, k) for (u, _) in cold_pairs]
    relevants = [[p] for (_, p) in cold_pairs]

    recall = float(np.mean([recall_at_k(pr, r, k=k) for pr, r in zip(preds, relevants)]))
    mrr = mean_reciprocal_rank(preds, relevants)
    ndcg = float(np.mean([ndcg_at_k(pr, r, k=k) for pr, r in zip(preds, relevants)]))
    return {
        "n_cold_pairs": len(cold_pairs),
        f"recall@{k}": recall,
        "mrr": mrr,
        f"ndcg@{k}": ndcg,
    }
