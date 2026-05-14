"""Tests for the recommendation modules."""

from __future__ import annotations

import numpy as np
import pytest

from pixelmatch.recommendation.content_based import ContentBasedRecommender
from pixelmatch.recommendation.hybrid import HybridRecommender
from pixelmatch.recommendation.matrix_factorization import ALSConfig, ALSRecommender


def _make_synth_implicit(n_users: int = 30, n_items: int = 50, density: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = int(n_users * n_items * density)
    u = rng.integers(0, n_users, size=n)
    i = rng.integers(0, n_items, size=n)
    r = rng.choice([1.0, 2.0, 5.0], size=n)
    return u, i, r, n_users, n_items


# ---------------- ALS ---------------- #
def test_als_fits_and_predicts_shape():
    u, i, r, nu, ni = _make_synth_implicit()
    als = ALSRecommender(ALSConfig(factors=8, iterations=2))
    als.fit(u, i, r, n_users=nu, n_items=ni)
    assert als.user_factors.shape == (nu, 8)
    assert als.item_factors.shape == (ni, 8)
    rec = als.recommend(user_id=0, k=5)
    assert len(rec) == 5


def test_als_recommend_excludes():
    u, i, r, nu, ni = _make_synth_implicit()
    als = ALSRecommender(ALSConfig(factors=4, iterations=1))
    als.fit(u, i, r, n_users=nu, n_items=ni)
    rec = als.recommend(user_id=1, k=10, exclude={0, 1, 2})
    assert all(x not in {0, 1, 2} for x in rec)


def test_als_unfitted_raises():
    als = ALSRecommender()
    with pytest.raises(RuntimeError):
        als.recommend(0)


# ---------------- Content-based ---------------- #
def test_content_based_similar_to_item():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((20, 8)).astype(np.float32)
    ids = np.arange(20)
    cb = ContentBasedRecommender(feats, ids)
    top = cb.similar_to_item(5, k=3)
    assert 5 not in top
    assert len(top) == 3


def test_content_based_history_recommendation():
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((30, 8)).astype(np.float32)
    cb = ContentBasedRecommender(feats, np.arange(30))
    top = cb.recommend_from_history([0, 1, 2], k=5, exclude_history=True)
    assert all(x not in {0, 1, 2} for x in top)


def test_content_based_unknown_item_raises():
    cb = ContentBasedRecommender(np.zeros((5, 4), dtype=np.float32), np.arange(5))
    with pytest.raises(KeyError):
        cb.similar_to_item(999)


def test_content_based_empty_history_returns_k():
    feats = np.eye(10, dtype=np.float32)
    cb = ContentBasedRecommender(feats, np.arange(10))
    top = cb.recommend_from_history([], k=4)
    assert len(top) == 4


def test_content_based_cold_start_rank():
    feats = np.eye(8, dtype=np.float32)
    cb = ContentBasedRecommender(feats, np.arange(8))
    seed = np.zeros(8, dtype=np.float32)
    seed[3] = 1.0
    top = cb.cold_start_rank(seed, k=1)
    assert top == [3]


# ---------------- Hybrid ---------------- #
def test_hybrid_recommender_cold_user_uses_content():
    u, i, r, nu, ni = _make_synth_implicit()
    als = ALSRecommender(ALSConfig(factors=4, iterations=1)).fit(u, i, r, n_users=nu, n_items=ni)
    feats = np.random.default_rng(2).standard_normal((ni, 4)).astype(np.float32)
    cb = ContentBasedRecommender(feats, np.arange(ni))
    hyb = HybridRecommender(als, cb, item_ids=np.arange(ni), warm_threshold=5)
    # Cold user: no history
    rec_cold = hyb.recommend(user_id=0, history=[], k=5)
    assert len(rec_cold) == 5
    # Warm user: > threshold history
    rec_warm = hyb.recommend(user_id=0, history=[0, 1, 2, 3, 4, 5, 6], k=5)
    assert len(rec_warm) == 5


def test_hybrid_excludes_history():
    u, i, r, nu, ni = _make_synth_implicit()
    als = ALSRecommender(ALSConfig(factors=4, iterations=1)).fit(u, i, r, n_users=nu, n_items=ni)
    feats = np.random.default_rng(3).standard_normal((ni, 4)).astype(np.float32)
    cb = ContentBasedRecommender(feats, np.arange(ni))
    hyb = HybridRecommender(als, cb, item_ids=np.arange(ni))
    rec = hyb.recommend(user_id=0, history=[0, 1], exclude_history=True, k=5)
    assert 0 not in rec and 1 not in rec
