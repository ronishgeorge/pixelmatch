# PixelMatch Benchmark Output (measured)

## Headline
- Catalog: **100,000 products**, **1,005,000 interactions**
- Evaluation queries: **996** (cold-start subset: **27**)
- Best method: **tfidf**
- Best NDCG@10: **0.8848**
- Baseline (BM25) NDCG@10: **0.8701**
- NDCG@10 improvement vs BM25: **+1.7%**
- Best cold-start Recall@10: **0.0011**
- Best p95 latency: **23.1ms**

## Per-method results

| Method | NDCG@10 | MRR | Recall@10 | Precision@5 | Cold-Recall@10 | p50 (ms) | p95 (ms) | p99 (ms) |
|--------|--------:|----:|----------:|------------:|---------------:|---------:|---------:|---------:|
| bm25 | 0.8701 | 0.8301 | 0.0011 | 0.8663 | 0.0011 | 243.61 | 347.08 | 545.11 |
| tfidf | 0.8848 | 0.8463 | 0.0011 | 0.8869 | 0.0011 | 21.35 | 23.11 | 26.33 |
| sbert_text | 0.7725 | 0.6945 | 0.0010 | 0.7745 | 0.0011 | 11.30 | 13.82 | 30.68 |
| multimodal_hybrid | 0.7860 | 0.8838 | 0.0009 | 0.8054 | 0.0010 | 249.11 | 328.80 | 529.04 |