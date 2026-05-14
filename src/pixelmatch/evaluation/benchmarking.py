"""End-to-end benchmarking harness.

Runs every retrieval / recommendation variant on a held-out evaluation set
and produces a comparison table suitable for the README and the docs
``benchmark_results.md``.

This module is intentionally framework-agnostic: each method is wrapped in
a callable that returns a ranked list of ids.  The harness only needs that
callable + the ground truth.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from pixelmatch.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    name: str
    ndcg_10: float
    mrr: float
    recall_10: float
    precision_5: float
    p95_latency_ms: float
    extras: dict[str, float]


def benchmark_method(
    name: str,
    query_fn: Callable[[str], list[int]],
    queries: list[str],
    relevants: list[set[int]],
    k: int = 10,
) -> BenchmarkResult:
    """Run one method against the provided queries."""
    preds: list[list[int]] = []
    latencies: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        preds.append(query_fn(q))
        latencies.append((time.perf_counter() - t0) * 1000.0)

    ndcg = float(np.mean([ndcg_at_k(p, r, k=k) for p, r in zip(preds, relevants)]))
    mrr = mean_reciprocal_rank(preds, relevants)
    rec = float(np.mean([recall_at_k(p, r, k=k) for p, r in zip(preds, relevants)]))
    prec5 = float(np.mean([precision_at_k(p, r, k=5) for p, r in zip(preds, relevants)]))
    p95 = float(np.percentile(np.asarray(latencies), 95))
    logger.info(
        "%s: NDCG@%d=%.3f MRR=%.3f Recall@%d=%.3f Prec@5=%.3f p95=%.2fms",
        name, k, ndcg, mrr, k, rec, prec5, p95,
    )
    return BenchmarkResult(
        name=name,
        ndcg_10=ndcg,
        mrr=mrr,
        recall_10=rec,
        precision_5=prec5,
        p95_latency_ms=p95,
        extras={},
    )


def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "method": r.name,
            "NDCG@10": round(r.ndcg_10, 4),
            "MRR": round(r.mrr, 4),
            "Recall@10": round(r.recall_10, 4),
            "Precision@5": round(r.precision_5, 4),
            "p95_latency_ms": round(r.p95_latency_ms, 2),
        }
        row.update({k: round(v, 4) for k, v in r.extras.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def relative_uplift(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    metric: str = "ndcg_10",
) -> float:
    """Compute (candidate - baseline) / baseline for the given metric."""
    base = getattr(baseline, metric)
    cand = getattr(candidate, metric)
    if base <= 0:
        return 0.0
    return (cand - base) / base
