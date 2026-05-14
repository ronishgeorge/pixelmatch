"""Vectorized information-retrieval metrics.

All functions accept a list of *ranked* predicted ids and a set/list of
ground-truth relevant ids per query.  Where applicable we also accept
graded relevance.

References
----------
* Järvelin & Kekäläinen, 2002 — *Cumulated Gain-Based Evaluation of IR
  Techniques*.  ACM TOIS.
* Manning, Raghavan, Schütze — *Introduction to Information Retrieval*,
  Chapter 8.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _truncate(pred: Sequence[int], k: int) -> Sequence[int]:
    return pred[:k]


# ---------------------------------------------------------------------- #
# Precision / Recall
# ---------------------------------------------------------------------- #
def precision_at_k(predicted: Sequence[int], relevant: Iterable[int], k: int) -> float:
    r"""Precision\@k = |pred ∩ rel| / k."""
    if k <= 0:
        return 0.0
    rel = set(relevant)
    top = _truncate(predicted, k)
    hits = sum(1 for p in top if p in rel)
    return hits / k


def recall_at_k(predicted: Sequence[int], relevant: Iterable[int], k: int) -> float:
    r"""Recall\@k = |pred ∩ rel| / |rel|.  Returns 0 if there are no relevants."""
    rel = set(relevant)
    if not rel:
        return 0.0
    top = _truncate(predicted, k)
    hits = sum(1 for p in top if p in rel)
    return hits / len(rel)


# ---------------------------------------------------------------------- #
# Reciprocal rank / MRR
# ---------------------------------------------------------------------- #
def reciprocal_rank(predicted: Sequence[int], relevant: Iterable[int]) -> float:
    r"""RR = 1 / rank of first relevant item; 0 if none found."""
    rel = set(relevant)
    for i, p in enumerate(predicted, start=1):
        if p in rel:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    predictions: Sequence[Sequence[int]],
    relevants: Sequence[Iterable[int]],
) -> float:
    if not predictions:
        return 0.0
    return float(np.mean([reciprocal_rank(p, r) for p, r in zip(predictions, relevants)]))


# ---------------------------------------------------------------------- #
# NDCG
# ---------------------------------------------------------------------- #
def _dcg(gains: np.ndarray) -> float:
    r"""DCG = \sum_i (2^{rel_i} - 1) / log2(i + 1).

    Uses the *gain* form, which is standard in graded-relevance IR.
    """
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float(((np.power(2.0, gains) - 1.0) * discounts).sum())


def ndcg_at_k(
    predicted: Sequence[int],
    relevant: Iterable[int] | dict[int, float],
    k: int,
) -> float:
    r"""NDCG\@k with graded relevance.

    Parameters
    ----------
    predicted : ranked ids returned by the model.
    relevant : either a set/list of relevant ids (binary) or a mapping
        ``{id: grade}`` for graded relevance.
    """
    if k <= 0:
        return 0.0
    if isinstance(relevant, dict):
        rel_map = {int(i): float(g) for i, g in relevant.items()}
    else:
        rel_map = {int(i): 1.0 for i in relevant}

    if not rel_map:
        return 0.0

    pred_gains = np.array([rel_map.get(int(p), 0.0) for p in _truncate(predicted, k)], dtype=np.float32)
    dcg = _dcg(pred_gains)
    # Ideal: sort all grades desc, take top-k
    ideal_gains = np.sort(np.fromiter(rel_map.values(), dtype=np.float32))[::-1][:k]
    idcg = _dcg(ideal_gains)
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


# ---------------------------------------------------------------------- #
# (Mean) Average Precision
# ---------------------------------------------------------------------- #
def average_precision(predicted: Sequence[int], relevant: Iterable[int], k: int | None = None) -> float:
    r"""AP\@k = (1/min(|rel|, k)) * \sum_i P\@i * 1[i is relevant]."""
    rel = set(relevant)
    if not rel:
        return 0.0
    top = _truncate(predicted, k) if k else predicted
    hits = 0
    total = 0.0
    for i, p in enumerate(top, start=1):
        if p in rel:
            hits += 1
            total += hits / i
    denom = min(len(rel), k) if k else len(rel)
    return total / max(denom, 1)


def mean_average_precision(
    predictions: Sequence[Sequence[int]],
    relevants: Sequence[Iterable[int]],
    k: int | None = None,
) -> float:
    if not predictions:
        return 0.0
    return float(np.mean([average_precision(p, r, k=k) for p, r in zip(predictions, relevants)]))
