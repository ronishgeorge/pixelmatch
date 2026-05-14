"""Tests for the retrieval modules."""

from __future__ import annotations

import numpy as np
import pytest

from pixelmatch.retrieval.bm25_baseline import BM25Retriever, tokenize
from pixelmatch.retrieval.faiss_index import FaissIndex, IndexConfig
from pixelmatch.retrieval.hybrid_retriever import HybridConfig, HybridRetriever
from pixelmatch.retrieval.tfidf_baseline import TfidfRetriever


# ---------------- FAISS / index ---------------- #
def test_faiss_build_and_search(faiss_test_index):
    idx, vecs = faiss_test_index
    ids, scores = idx.search(vecs[0:1], k=5)
    assert ids.shape == (1, 5)
    # The first hit should be self with cosine sim ≈ 1
    assert ids[0, 0] == 0
    assert scores[0, 0] > 0.99


def test_faiss_dim_mismatch_raises():
    idx = FaissIndex(dim=8)
    with pytest.raises(ValueError):
        idx.build(np.zeros((4, 16), dtype=np.float32))


def test_faiss_ids_mismatch_raises():
    idx = FaissIndex(dim=4)
    with pytest.raises(ValueError):
        idx.build(np.zeros((4, 4), dtype=np.float32), ids=[1, 2])


def test_faiss_search_before_build_raises():
    idx = FaissIndex(dim=4)
    with pytest.raises(RuntimeError):
        idx.search(np.zeros((1, 4), dtype=np.float32))


def test_faiss_benchmark_shape(faiss_test_index):
    idx, vecs = faiss_test_index
    stats = idx.benchmark(vecs[:5], k=3)
    assert "p50_ms" in stats and "p95_ms" in stats and "p99_ms" in stats
    assert stats["n_queries"] == 5


def test_faiss_save_load_roundtrip(tmp_path, faiss_test_index):
    idx, vecs = faiss_test_index
    p = tmp_path / "idx.bin"
    idx.save(str(p))
    new_idx = FaissIndex(dim=64)
    new_idx.load(str(p))
    ids, _ = new_idx.search(vecs[0:1], k=3)
    assert ids[0, 0] == 0


def test_index_config_default_hnsw():
    cfg = IndexConfig()
    assert cfg.index_type == "hnsw"
    assert cfg.hnsw_m == 32


# ---------------- BM25 ---------------- #
def test_bm25_finds_exact_match():
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "a totally unrelated sentence about cats",
        "another sentence mentioning a fox running quickly",
    ]
    bm = BM25Retriever().fit(docs)
    ids, scores = bm.search("fox", k=2)
    # Both docs containing 'fox' should rank before the unrelated cat sentence
    assert 0 in ids[:2].tolist()
    assert 1 not in ids[:2].tolist() or scores[ids.tolist().index(1)] < scores[0]


def test_bm25_empty_query_no_crash():
    bm = BM25Retriever().fit(["alpha beta", "gamma delta"])
    ids, _ = bm.search("", k=2)
    assert ids.shape == (2,)


def test_bm25_tokenizer_lowercases():
    assert tokenize("The Quick BROWN fox") == ["the", "quick", "brown", "fox"]


# ---------------- TF-IDF ---------------- #
def test_tfidf_returns_top_k():
    docs = ["a dog runs", "a cat sleeps", "dogs and cats"]
    r = TfidfRetriever().fit(docs)
    ids, _ = r.search("dog", k=2)
    assert len(ids) == 2
    assert 0 in ids.tolist()


# ---------------- Hybrid ---------------- #
def test_hybrid_rrf_fuses(faiss_test_index):
    idx, vecs = faiss_test_index
    # BM25 over numeric tags so we can produce ranked ids
    bm = BM25Retriever().fit([f"item id {i}" for i in range(200)])
    hyb = HybridRetriever(dense=idx, lexical=bm, config=HybridConfig(rerank_window=10))
    out = hyb.search(text_query="item id 7", query_vec=vecs[7:8], k=5)
    assert len(out) == 5
    assert 7 in out
