"""Run PixelMatch benchmark end-to-end and produce real measured numbers.

Measures, on the generated 10K-SKU catalog + 100K interactions:
  - NDCG@10, MRR, Recall@10, Precision@5  for each retrieval method
  - Cold-start Recall@10                   on held-out zero-interaction products
  - p50 / p95 / p99 query latency          per method

Methods evaluated:
  1. BM25 (text-only sparse)
  2. TF-IDF (text-only sparse)
  3. SBERT (text-only dense via sentence-transformers)
  4. Multimodal hybrid (SBERT text + color-histogram image, late-avg fusion)

Output: reports/benchmark_output.json (machine-readable)
        reports/benchmark_summary.md   (human-readable)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- metrics
def ndcg_at_k(predicted: Sequence[int], relevant: set[int], k: int = 10) -> float:
    pred_k = list(predicted)[:k]
    if not pred_k:
        return 0.0
    gains = np.array([1.0 if pid in relevant else 0.0 for pid in pred_k])
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal_gains = np.ones(min(len(relevant), k))
    if len(ideal_gains) == 0:
        return 0.0
    ideal_disc = 1.0 / np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = float((ideal_gains * ideal_disc).sum())
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(predicted: Sequence[int], relevant: set[int]) -> float:
    for i, pid in enumerate(predicted, start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(predicted: Sequence[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(list(predicted)[:k]) & relevant) / len(relevant)


def precision_at_k(predicted: Sequence[int], relevant: set[int], k: int = 5) -> float:
    pred_k = list(predicted)[:k]
    if not pred_k:
        return 0.0
    return len(set(pred_k) & relevant) / len(pred_k)


# ----------------------------------------------------------------- BM25 (lite)
class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b

    def fit(self, docs: list[list[str]]) -> "BM25":
        self.docs = docs
        self.N = len(docs)
        self.doc_lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if self.N else 0.0
        df: dict[str, int] = defaultdict(int)
        for d in docs:
            for term in set(d):
                df[term] += 1
        self.idf = {t: float(np.log((self.N - f + 0.5) / (f + 0.5) + 1.0)) for t, f in df.items()}
        self.tf = [defaultdict(int) for _ in docs]
        for i, d in enumerate(docs):
            for t in d:
                self.tf[i][t] += 1
        return self

    def score(self, query_terms: list[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        for t in query_terms:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i in range(self.N):
                tf = self.tf[i].get(t, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / max(self.avgdl, 1e-9))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores


# --------------------------------------------------------------------- runner
def main() -> None:
    catalog = pd.read_csv("data/catalog.csv")
    interactions = pd.read_csv("data/interactions.csv")
    logger.info("Catalog: %d, Interactions: %d", len(catalog), len(interactions))

    # Build per-product corpus text
    if "title" not in catalog.columns:
        # fallback to any text columns
        text_cols = [c for c in catalog.columns if catalog[c].dtype == object][:2]
        catalog["text"] = catalog[text_cols].fillna("").agg(" ".join, axis=1)
    else:
        desc_col = "description" if "description" in catalog.columns else None
        catalog["text"] = catalog["title"].fillna("")
        if desc_col is not None:
            catalog["text"] = catalog["text"] + " " + catalog[desc_col].fillna("")

    product_ids = catalog["product_id"].tolist() if "product_id" in catalog.columns else list(range(len(catalog)))
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    texts = catalog["text"].fillna("").astype(str).tolist()

    # ------- Build query/relevant ground truth (category-based) -------
    # Principled retrieval eval: for each query product P (sampled), the relevance set
    # is the set of OTHER products in the SAME category as P. The query is the title
    # of P with brand/category name redacted to force semantic retrieval (not exact-match).
    rng = np.random.default_rng(42)
    if "category" in catalog.columns:
        cat_col = "category"
    else:
        cat_col = catalog.columns[catalog.dtypes == object].tolist()[0]
    cat_to_pids: dict = catalog.groupby(cat_col)["product_id"].apply(list).to_dict()
    logger.info("Categories: %d", len(cat_to_pids))

    # Sample 1000 query products distributed across categories
    sampled = []
    per_cat = max(1, 1000 // len(cat_to_pids))
    for cat, pids in cat_to_pids.items():
        if len(pids) < 2:
            continue
        n = min(per_cat, len(pids))
        sampled.extend(rng.choice(pids, n, replace=False).tolist())
    rng.shuffle(sampled)
    sampled = sampled[:1000]
    logger.info("Sampled %d query products", len(sampled))

    # Build (query, relevant) pairs.
    # Query = title of P with stop-categorical words removed (forces semantic match).
    # Relevant = set of other products in same category.
    def make_query(text: str, cat: str) -> str:
        words = text.split()
        # Drop category name and overly generic words
        drop = {cat.lower(), "the", "a", "an", "and", "with", "for", "in"}
        kept = [w for w in words if w.lower() not in drop]
        return " ".join(kept[:8])  # first 8 tokens

    held_out: list[tuple[str, set[int]]] = []
    cat_of_pid = dict(zip(catalog["product_id"], catalog[cat_col]))
    for pid in sampled:
        if pid not in pid_to_idx:
            continue
        cat = cat_of_pid[pid]
        text = texts[pid_to_idx[pid]]
        query = make_query(text, str(cat))
        same_cat = set(cat_to_pids[cat]) - {pid}
        held_out.append((query, same_cat))
    logger.info("Held-out evaluation queries: %d (category-based relevance)", len(held_out))

    # Cold-start eval: queries whose query product is itself a cold product
    cold_pids = set(interactions[interactions.get("is_cold_product", False) == True]["product_id"].unique().tolist())
    cold_sampled = [pid for pid in sampled if pid in cold_pids]
    cold_queries = []
    for pid in cold_sampled:
        if pid not in pid_to_idx:
            continue
        cat = cat_of_pid[pid]
        text = texts[pid_to_idx[pid]]
        query = make_query(text, str(cat))
        same_cat = set(cat_to_pids[cat]) - {pid}
        cold_queries.append((query, same_cat))
    logger.info("Cold-start queries: %d", len(cold_queries))

    # ----------- Build retrieval indices -----------
    logger.info("Building TF-IDF index")
    tfidf = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), stop_words="english")
    X_tfidf = tfidf.fit_transform(texts)

    logger.info("Building BM25 index")
    tok = [t.lower().split() for t in texts]
    bm25 = BM25().fit(tok)

    logger.info("Building SBERT embeddings (this may take a few minutes on CPU)")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    t0 = time.perf_counter()
    sbert_embs = sbert.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    logger.info("SBERT encoding done in %.1fs (shape=%s)", time.perf_counter() - t0, sbert_embs.shape)

    # FAISS-cpu segfaults on macOS; use direct numpy cosine for 10K vectors instead.
    # For a 100K+ catalog, swap in faiss.IndexHNSWFlat — same API surface.
    logger.info("Using numpy direct cosine for dense retrieval (10K corpus, exact)")
    sbert_embs_f32 = sbert_embs.astype(np.float32)

    # Multimodal: SBERT text + simple "image" feature (color histogram on a deterministic seed)
    # If we have images, encode them with a simple color-histogram extractor.
    image_feat = None
    images_dir = Path("data/images")
    if images_dir.exists():
        logger.info("Computing color-histogram image features (subset)")
        from PIL import Image
        feats = np.zeros((len(catalog), 48), dtype=np.float32)
        for i, pid in enumerate(product_ids):
            img_path = images_dir / f"{pid}.png"
            if img_path.exists():
                img = np.array(Image.open(img_path).convert("RGB"))
                hist = []
                for ch in range(3):
                    h, _ = np.histogram(img[:, :, ch], bins=16, range=(0, 256), density=True)
                    hist.append(h)
                feats[i] = np.concatenate(hist).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1
        image_feat = (feats / norms).astype(np.float32)
        logger.info("Image features: %s (non-zero rows: %d)", image_feat.shape, int(np.any(image_feat != 0, axis=1).sum()))

    # ----------- Methods to benchmark -----------
    def search_bm25(q: str, k: int = 10) -> list[int]:
        scores = bm25.score(q.lower().split())
        top = np.argsort(-scores)[:k]
        return [product_ids[i] for i in top]

    def search_tfidf(q: str, k: int = 10) -> list[int]:
        qv = tfidf.transform([q])
        sims = cosine_similarity(qv, X_tfidf).ravel()
        top = np.argsort(-sims)[:k]
        return [product_ids[i] for i in top]

    def search_sbert(q: str, k: int = 10) -> list[int]:
        qv = sbert.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        sims = (sbert_embs_f32 @ qv.T).ravel()
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [product_ids[i] for i in top]

    def search_multimodal(q: str, k: int = 10) -> list[int]:
        # SBERT score + image score (cosine vs mean of relevant images)
        qv = sbert.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        text_sim = (sbert_embs @ qv.T).ravel()
        if image_feat is not None and image_feat.any():
            # use mean image feat from BM25 top-50 as "visual prior"
            seed_top = np.argsort(-bm25.score(q.lower().split()))[:50]
            visual_prior = image_feat[seed_top].mean(axis=0, keepdims=True)
            visual_prior /= max(np.linalg.norm(visual_prior), 1e-9)
            img_sim = (image_feat @ visual_prior.T).ravel()
            combined = 0.6 * text_sim + 0.4 * img_sim
        else:
            combined = text_sim
        top = np.argsort(-combined)[:k]
        return [product_ids[i] for i in top]

    methods: dict[str, Callable[[str, int], list[int]]] = {
        "bm25": search_bm25,
        "tfidf": search_tfidf,
        "sbert_text": search_sbert,
        "multimodal_hybrid": search_multimodal,
    }

    # ----------- Run benchmark -----------
    results: dict[str, dict] = {}
    for name, fn in methods.items():
        logger.info("Benchmarking method: %s", name)
        ndcgs, mrrs, recs, precs, lats = [], [], [], [], []
        for q, rel in held_out:
            t0 = time.perf_counter()
            preds = fn(q, 10)
            lats.append((time.perf_counter() - t0) * 1000)
            ndcgs.append(ndcg_at_k(preds, rel, 10))
            mrrs.append(reciprocal_rank(preds, rel))
            recs.append(recall_at_k(preds, rel, 10))
            precs.append(precision_at_k(preds, rel, 5))

        # Cold-start subset
        cold_recs = []
        for q, rel in cold_queries:
            preds = fn(q, 10)
            cold_recs.append(recall_at_k(preds, rel, 10))

        lats_arr = np.asarray(lats)
        results[name] = {
            "ndcg_10": float(np.mean(ndcgs)),
            "mrr": float(np.mean(mrrs)),
            "recall_10": float(np.mean(recs)),
            "precision_5": float(np.mean(precs)),
            "cold_recall_10": float(np.mean(cold_recs)) if cold_recs else None,
            "p50_latency_ms": float(np.percentile(lats_arr, 50)),
            "p95_latency_ms": float(np.percentile(lats_arr, 95)),
            "p99_latency_ms": float(np.percentile(lats_arr, 99)),
            "n_queries": len(held_out),
            "n_cold_queries": len(cold_queries),
        }
        logger.info(
            "%s: NDCG@10=%.3f MRR=%.3f Recall@10=%.3f Cold-Recall@10=%.3f p95=%.1fms",
            name, results[name]["ndcg_10"], results[name]["mrr"],
            results[name]["recall_10"], results[name]["cold_recall_10"] or 0.0,
            results[name]["p95_latency_ms"],
        )

    # ----------- Headline computations -----------
    baseline = "bm25"
    best = max(results, key=lambda m: results[m]["ndcg_10"])
    ndcg_improvement_pct = (
        (results[best]["ndcg_10"] / results[baseline]["ndcg_10"] - 1.0) * 100
        if results[baseline]["ndcg_10"] > 0 else None
    )

    headline = {
        "catalog_size": len(catalog),
        "interactions_total": len(interactions),
        "evaluation_queries": len(held_out),
        "cold_start_queries": len(cold_queries),
        "baseline_method": baseline,
        "best_method": best,
        "best_ndcg_10": results[best]["ndcg_10"],
        "baseline_ndcg_10": results[baseline]["ndcg_10"],
        "ndcg_10_improvement_pct_vs_bm25": ndcg_improvement_pct,
        "best_cold_recall_10": results[best]["cold_recall_10"],
        "best_p95_latency_ms": results[best]["p95_latency_ms"],
        "seed": 42,
    }

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    (reports / "benchmark_output.json").write_text(
        json.dumps({"headline": headline, "per_method": results}, indent=2)
    )

    # Human-readable markdown
    lines = [
        "# PixelMatch Benchmark Output (measured)",
        "",
        "## Headline",
        f"- Catalog: **{len(catalog):,} products**, **{len(interactions):,} interactions**",
        f"- Evaluation queries: **{len(held_out):,}** (cold-start subset: **{len(cold_queries):,}**)",
        f"- Best method: **{best}**",
        f"- Best NDCG@10: **{results[best]['ndcg_10']:.4f}**",
        f"- Baseline (BM25) NDCG@10: **{results[baseline]['ndcg_10']:.4f}**",
        f"- NDCG@10 improvement vs BM25: **{ndcg_improvement_pct:+.1f}%**" if ndcg_improvement_pct is not None else "",
        f"- Best cold-start Recall@10: **{results[best]['cold_recall_10']:.4f}**",
        f"- Best p95 latency: **{results[best]['p95_latency_ms']:.1f}ms**",
        "",
        "## Per-method results",
        "",
        "| Method | NDCG@10 | MRR | Recall@10 | Precision@5 | Cold-Recall@10 | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|--------|--------:|----:|----------:|------------:|---------------:|---------:|---------:|---------:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['ndcg_10']:.4f} | {r['mrr']:.4f} | {r['recall_10']:.4f} | "
            f"{r['precision_5']:.4f} | {r['cold_recall_10']:.4f} | "
            f"{r['p50_latency_ms']:.2f} | {r['p95_latency_ms']:.2f} | {r['p99_latency_ms']:.2f} |"
        )
    (reports / "benchmark_summary.md").write_text("\n".join(lines))

    print("\n=== HEADLINE NUMBERS ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
