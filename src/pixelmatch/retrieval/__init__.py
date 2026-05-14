"""Retrieval: dense (FAISS) + lexical (BM25, TF-IDF) + hybrid."""

from pixelmatch.retrieval.bm25_baseline import BM25Retriever
from pixelmatch.retrieval.faiss_index import FaissIndex
from pixelmatch.retrieval.hybrid_retriever import HybridRetriever
from pixelmatch.retrieval.tfidf_baseline import TfidfRetriever

__all__ = ["FaissIndex", "BM25Retriever", "TfidfRetriever", "HybridRetriever"]
