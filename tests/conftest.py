"""Shared pytest fixtures.

These keep the test suite snappy by building only tiny in-memory artefacts —
no model downloads, no real PNG renders, no full catalog generation.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# Make the ``src`` layout importable without an editable install
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Reset RNGs before every test."""
    random.seed(42)
    np.random.seed(42)


# ---------------------------------------------------------------------- #
@pytest.fixture
def tiny_catalog() -> pd.DataFrame:
    """100-row catalog with realistic-ish columns."""
    rng = np.random.default_rng(42)
    n = 100
    categories = ["apparel", "electronics", "kitchen", "outdoor"]
    brands = ["Atlas", "Nimbus", "Vertex", "Lumen"]
    rows = []
    for pid in range(n):
        cat = categories[pid % len(categories)]
        brand = brands[pid % len(brands)]
        rows.append(
            {
                "product_id": pid,
                "title": f"{brand} {cat} item {pid}",
                "description": f"A {cat} product made by {brand}. " * 5,
                "category": cat,
                "brand": brand,
                "color": "black",
                "price": float(rng.uniform(5, 200)),
                "has_image": False,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_interactions(tiny_catalog: pd.DataFrame) -> pd.DataFrame:
    """500 fake interactions over the tiny catalog."""
    rng = np.random.default_rng(0)
    n_users = 50
    n = 500
    df = pd.DataFrame(
        {
            "user_id": rng.integers(0, n_users, size=n),
            "product_id": rng.integers(0, len(tiny_catalog), size=n),
            "interaction_type": rng.choice(["view", "click", "purchase"], size=n, p=[0.7, 0.25, 0.05]),
            "rating": rng.choice([1.0, 2.0, 5.0], size=n, p=[0.7, 0.25, 0.05]),
            "timestamp": rng.integers(0, 1_000_000, size=n),
        }
    )
    return df


# ---------------------------------------------------------------------- #
@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Single 32x32 RGB PNG used by image-encoder tests."""
    arr = (np.random.RandomState(0).rand(32, 32, 3) * 255).astype("uint8")
    img = Image.fromarray(arr, mode="RGB")
    p = tmp_path / "sample.png"
    img.save(p)
    return p


@pytest.fixture
def sample_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """``(text_vec, image_vec)`` row-pair for the multimodal tests."""
    rng = np.random.default_rng(123)
    t = rng.standard_normal((4, 16)).astype("float32")
    i = rng.standard_normal((4, 32)).astype("float32")
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    i /= np.linalg.norm(i, axis=1, keepdims=True)
    return t, i


# ---------------------------------------------------------------------- #
@pytest.fixture
def faiss_test_index():
    """Build a 64-dim brute-force / FAISS index over 200 random vectors."""
    from pixelmatch.retrieval.faiss_index import FaissIndex

    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((200, 64)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    idx = FaissIndex(dim=64)
    idx.build(vecs, ids=np.arange(200))
    return idx, vecs


# ---------------------------------------------------------------------- #
@pytest.fixture
def env_no_proxy(monkeypatch):
    """Strip HTTP proxy env vars that can blow up offline test runs."""
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    yield
