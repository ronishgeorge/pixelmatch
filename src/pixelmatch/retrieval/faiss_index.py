"""FAISS HNSW / IVF index wrapper.

Provides a thin, testable surface around :mod:`faiss` with an in-memory
fallback (brute-force) for environments where the FAISS wheel is not
available (e.g. CI on unsupported platforms).
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore
    _HAS_FAISS = False
    logger.warning("faiss not available — using brute-force numpy fallback")


@dataclass
class IndexConfig:
    index_type: str = "hnsw"       # hnsw | ivf | flat
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    ivf_nlist: int = 1024
    ivf_nprobe: int = 16
    metric: str = "ip"             # ip (inner product) | l2


@dataclass
class IndexStats:
    n_vectors: int = 0
    dim: int = 0
    build_seconds: float = 0.0
    bytes_in_memory: int = 0
    last_query_ms: list[float] = field(default_factory=list)


class _BruteForceIndex:
    """Pure numpy fallback used when FAISS is missing.  Uses inner product."""

    def __init__(self, dim: int, metric: str = "ip") -> None:
        self.dim = dim
        self.metric = metric
        self._vecs = np.zeros((0, dim), dtype=np.float32)

    def add(self, vecs: np.ndarray) -> None:
        self._vecs = np.vstack([self._vecs, vecs.astype(np.float32)])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.metric == "ip":
            sims = queries @ self._vecs.T
        else:
            # L2 squared
            sims = -((queries[:, None, :] - self._vecs[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(-sims, axis=1)[:, :k]
        rows = np.arange(queries.shape[0])[:, None]
        dist = sims[rows, idx]
        return dist, idx


class FaissIndex:
    """High-level FAISS wrapper that owns the underlying index, ids and stats.

    Notes
    -----
    *  Inner-product is used by default — embeddings should be L2-normalized
       so IP is equivalent to cosine similarity.
    *  ``ids`` may be arbitrary 64-bit integers (e.g. product ids).
    """

    def __init__(self, dim: int, config: IndexConfig | None = None) -> None:
        self.dim = dim
        self.config = config or IndexConfig()
        self._index = None
        self._ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self.stats = IndexStats(dim=dim)

    # ------------------------------------------------------------------ #
    def _make_index(self):
        if not _HAS_FAISS:
            return _BruteForceIndex(self.dim, self.config.metric)

        cfg = self.config
        if cfg.index_type == "flat":
            idx = (
                faiss.IndexFlatIP(self.dim) if cfg.metric == "ip" else faiss.IndexFlatL2(self.dim)
            )
        elif cfg.index_type == "hnsw":
            metric = faiss.METRIC_INNER_PRODUCT if cfg.metric == "ip" else faiss.METRIC_L2
            idx = faiss.IndexHNSWFlat(self.dim, cfg.hnsw_m, metric)
            idx.hnsw.efConstruction = cfg.hnsw_ef_construction
            idx.hnsw.efSearch = cfg.hnsw_ef_search
        elif cfg.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dim)
            idx = faiss.IndexIVFFlat(quantizer, self.dim, cfg.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"unknown index_type: {cfg.index_type}")
        return idx

    # ------------------------------------------------------------------ #
    def build(self, embeddings: np.ndarray, ids: Iterable[int] | None = None) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"dim mismatch: index dim={self.dim} but embeddings dim={embeddings.shape[1]}"
            )
        ids_arr = (
            np.asarray(list(ids), dtype=np.int64)
            if ids is not None
            else np.arange(embeddings.shape[0], dtype=np.int64)
        )
        if ids_arr.shape[0] != embeddings.shape[0]:
            raise ValueError("ids length must match embeddings rows")

        t0 = time.perf_counter()
        self._index = self._make_index()
        if _HAS_FAISS and self.config.index_type == "ivf":
            self._index.train(embeddings)
            self._index.nprobe = self.config.ivf_nprobe
        self._index.add(embeddings)
        self._ids = ids_arr
        elapsed = time.perf_counter() - t0

        self.stats.n_vectors = embeddings.shape[0]
        self.stats.build_seconds = elapsed
        self.stats.bytes_in_memory = embeddings.nbytes
        logger.info(
            "Built %s index: n=%d dim=%d in %.2fs",
            self.config.index_type,
            embeddings.shape[0],
            self.dim,
            elapsed,
        )

    # ------------------------------------------------------------------ #
    def search(
        self, query: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the index.  Returns ``(ids, scores)`` of shape ``(n_q, k)``."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")
        q = np.atleast_2d(np.ascontiguousarray(query, dtype=np.float32))
        k_eff = min(k, max(1, self.stats.n_vectors))
        t0 = time.perf_counter()
        dist, idx = self._index.search(q, k_eff)
        self.stats.last_query_ms.append((time.perf_counter() - t0) * 1000.0)
        # Map internal positions to external ids
        ids = np.where(idx >= 0, self._ids[np.clip(idx, 0, len(self._ids) - 1)], -1)
        return ids, dist

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = {
            "dim": self.dim,
            "config": self.config,
            "ids": self._ids,
        }
        with open(path + ".meta", "wb") as fh:
            pickle.dump(meta, fh)
        if _HAS_FAISS and not isinstance(self._index, _BruteForceIndex):
            faiss.write_index(self._index, path)
        else:
            with open(path, "wb") as fh:
                pickle.dump(self._index, fh)
        logger.info("Saved index to %s", path)

    def load(self, path: str) -> None:
        with open(path + ".meta", "rb") as fh:
            meta = pickle.load(fh)
        self.dim = meta["dim"]
        self.config = meta["config"]
        self._ids = meta["ids"]
        if _HAS_FAISS and os.path.exists(path) and self.config.index_type in {"hnsw", "ivf", "flat"}:
            try:
                self._index = faiss.read_index(path)
            except Exception:
                with open(path, "rb") as fh:
                    self._index = pickle.load(fh)
        else:
            with open(path, "rb") as fh:
                self._index = pickle.load(fh)
        self.stats.n_vectors = len(self._ids)

    # ------------------------------------------------------------------ #
    def benchmark(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> dict[str, float]:
        """Run ``len(queries)`` searches and return p50/p95/p99 latency in ms."""
        latencies: list[float] = []
        for q in queries:
            t0 = time.perf_counter()
            self.search(q, k=k)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(latencies)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean()),
            "n_queries": len(latencies),
            "build_seconds": float(self.stats.build_seconds),
            "bytes_in_memory": int(self.stats.bytes_in_memory),
        }
