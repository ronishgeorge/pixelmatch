"""Evaluation: information-retrieval metrics, cold-start splits, benchmarking."""

from pixelmatch.evaluation.metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "ndcg_at_k",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "average_precision",
    "mean_average_precision",
]
