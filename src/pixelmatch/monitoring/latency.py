"""Lightweight latency tracking with periodic CSV flushing.

Designed to be embedded in request handlers without taking a dependency on
Prometheus / OpenTelemetry.  For real production we recommend swapping
this for an OpenTelemetry exporter — the API is intentionally compatible
(``with LatencyTracker("op"):``).
"""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

import numpy as np


class LatencyTracker:
    """Context manager that records elapsed milliseconds.

    Example
    -------
    >>> with LatencyTracker("search.text"):  # doctest: +SKIP
    ...     run_query()
    """

    _buffers: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=10_000))
    _lock = threading.Lock()
    _flush_path: str | None = None

    def __init__(self, name: str) -> None:
        self.name = name
        self._t0 = 0.0
        self.elapsed_ms = 0.0

    # ------------------------------------------------------------------ #
    def __enter__(self) -> "LatencyTracker":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        with LatencyTracker._lock:
            LatencyTracker._buffers[self.name].append(self.elapsed_ms)

    # ------------------------------------------------------------------ #
    @classmethod
    def configure_flush(cls, path: str) -> None:
        """Attach a CSV file path that will be appended to on each flush."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cls._flush_path = path

    @classmethod
    def flush(cls) -> None:
        """Persist the current buffers to CSV (append-only) and clear them."""
        if cls._flush_path is None:
            return
        with cls._lock:
            rows = []
            ts = time.time()
            for name, vals in cls._buffers.items():
                if not vals:
                    continue
                arr = np.asarray(vals)
                rows.append(
                    {
                        "ts": ts,
                        "name": name,
                        "n": len(arr),
                        "p50_ms": float(np.percentile(arr, 50)),
                        "p95_ms": float(np.percentile(arr, 95)),
                        "p99_ms": float(np.percentile(arr, 99)),
                        "mean_ms": float(arr.mean()),
                    }
                )
                cls._buffers[name].clear()

            if not rows:
                return
            new_file = not os.path.exists(cls._flush_path)
            with open(cls._flush_path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                if new_file:
                    writer.writeheader()
                writer.writerows(rows)

    # ------------------------------------------------------------------ #
    @classmethod
    def snapshot(cls) -> dict[str, dict[str, float]]:
        """Return a percentile snapshot of all current buffers (without clearing)."""
        out: dict[str, dict[str, float]] = {}
        with cls._lock:
            for name, vals in cls._buffers.items():
                if not vals:
                    continue
                arr = np.asarray(vals)
                out[name] = {
                    "n": float(len(arr)),
                    "p50_ms": float(np.percentile(arr, 50)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "p99_ms": float(np.percentile(arr, 99)),
                    "mean_ms": float(arr.mean()),
                }
        return out

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._buffers.clear()


def compute_percentiles(values: list[float] | np.ndarray) -> dict[str, float]:
    """Convenience for one-shot percentile reporting."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }
