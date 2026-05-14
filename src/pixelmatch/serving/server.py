"""FastAPI application exposing PixelMatch as a JSON HTTP service.

Endpoints
---------
* ``POST /search/text``        — text-only retrieval
* ``POST /search/image``       — image upload retrieval
* ``POST /search/multimodal``  — text + image retrieval
* ``POST /recommend/{user_id}``— personalized top-k recommendations
* ``GET  /metrics``            — latency percentile snapshot
* ``GET  /health``             — liveness probe

The server lazily initializes a small *demo* in-memory index so that the
service is runnable without first running offline indexing.  In production
the index should be built by ``make index`` and loaded from disk on
startup.
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from pixelmatch import load_config
from pixelmatch.encoders import ImageEncoder, MultiModalEncoder, TextEncoder
from pixelmatch.encoders.multimodal_encoder import FusionConfig
from pixelmatch.monitoring import LatencyTracker
from pixelmatch.retrieval import FaissIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Pydantic models
# ---------------------------------------------------------------------- #
class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    k: int = Field(10, ge=1, le=100)


class SearchHit(BaseModel):
    product_id: int
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    latency_ms: float
    method: str


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[int]
    latency_ms: float
    cold_user: bool


# ---------------------------------------------------------------------- #
# State container
# ---------------------------------------------------------------------- #
class _AppState:
    text_encoder: Optional[TextEncoder] = None
    image_encoder: Optional[ImageEncoder] = None
    fusion: Optional[MultiModalEncoder] = None
    index: Optional[FaissIndex] = None
    product_ids: Optional[np.ndarray] = None
    config: dict = {}


state = _AppState()


def _bootstrap_demo_index(n: int = 1024, seed: int = 42) -> None:
    """Create a tiny random index so the API is usable at zero-setup time."""
    rng = np.random.default_rng(seed)
    cfg = state.config
    dim = state.fusion.dim if state.fusion is not None else 384
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    ids = np.arange(n, dtype=np.int64)
    idx = FaissIndex(dim=dim)
    idx.build(vecs, ids=ids)
    state.index = idx
    state.product_ids = ids
    logger.info("Bootstrapped demo index: n=%d dim=%d", n, dim)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    state.config = load_config()
    enc_cfg = state.config.get("encoders", {})
    fuse_cfg = state.config.get("fusion", {})

    state.text_encoder = TextEncoder(
        model_name=enc_cfg.get("text_model", "sentence-transformers/all-MiniLM-L6-v2"),
        dim=enc_cfg.get("text_dim", 384),
        device=enc_cfg.get("device", "cpu"),
    )
    state.image_encoder = ImageEncoder(
        backbone=enc_cfg.get("image_backbone", "resnet50"),
        device=enc_cfg.get("device", "cpu"),
        clip_model=enc_cfg.get("clip_model", "openai/clip-vit-base-patch32"),
    )
    state.fusion = MultiModalEncoder(
        config=FusionConfig(
            strategy=fuse_cfg.get("strategy", "early_concat"),
            projection_dim=fuse_cfg.get("projection_dim", 256),
            text_weight=fuse_cfg.get("text_weight", 0.6),
            image_weight=fuse_cfg.get("image_weight", 0.4),
        ),
        text_dim=enc_cfg.get("text_dim", 384),
        image_dim=enc_cfg.get("image_dim", 2048),
    )

    index_path = os.environ.get("PIXELMATCH_INDEX_PATH")
    if index_path and os.path.exists(index_path + ".meta"):
        idx = FaissIndex(dim=state.fusion.dim)
        idx.load(index_path)
        state.index = idx
        state.product_ids = idx._ids  # noqa: SLF001
        logger.info("Loaded index from %s (n=%d)", index_path, len(state.product_ids))
    else:
        _bootstrap_demo_index()

    yield


app = FastAPI(
    title="PixelMatch",
    version="0.1.0",
    description="Multi-Modal Visual Search & Recommendation Engine",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------- #
# Health + metrics
# ---------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/metrics")
def metrics() -> dict:
    return {"latency": LatencyTracker.snapshot()}


# ---------------------------------------------------------------------- #
# Search endpoints
# ---------------------------------------------------------------------- #
@app.post("/search/text", response_model=SearchResponse)
def search_text(req: TextSearchRequest) -> SearchResponse:
    if state.index is None or state.text_encoder is None or state.fusion is None:
        raise HTTPException(503, "index not ready")
    with LatencyTracker("search.text") as t:
        text_vec = state.text_encoder.encode(req.query)
        q = state.fusion.fuse_query(text_vec=text_vec, image_vec=None)
        ids, scores = state.index.search(q, k=req.k)
    hits = [SearchHit(product_id=int(i), score=float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
    return SearchResponse(hits=hits, latency_ms=t.elapsed_ms, method="text")


@app.post("/search/image", response_model=SearchResponse)
async def search_image(file: UploadFile = File(...), k: int = Form(10)) -> SearchResponse:
    if state.index is None or state.image_encoder is None or state.fusion is None:
        raise HTTPException(503, "index not ready")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"invalid image: {exc}") from exc
    with LatencyTracker("search.image") as t:
        image_vec = state.image_encoder.encode(img)
        q = state.fusion.fuse_query(text_vec=None, image_vec=image_vec)
        ids, scores = state.index.search(q, k=k)
    hits = [SearchHit(product_id=int(i), score=float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
    return SearchResponse(hits=hits, latency_ms=t.elapsed_ms, method="image")


@app.post("/search/multimodal", response_model=SearchResponse)
async def search_multimodal(
    query: str = Form(...),
    file: UploadFile = File(...),
    k: int = Form(10),
) -> SearchResponse:
    if state.index is None or state.text_encoder is None or state.image_encoder is None or state.fusion is None:
        raise HTTPException(503, "index not ready")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"invalid image: {exc}") from exc
    with LatencyTracker("search.multimodal") as t:
        t_vec = state.text_encoder.encode(query)
        i_vec = state.image_encoder.encode(img)
        q = state.fusion.fuse_query(text_vec=t_vec, image_vec=i_vec)
        ids, scores = state.index.search(q, k=k)
    hits = [SearchHit(product_id=int(i), score=float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
    return SearchResponse(hits=hits, latency_ms=t.elapsed_ms, method="multimodal")


# ---------------------------------------------------------------------- #
# Recommendation endpoint (demo: ranks by index neighbourhood of user-id-hashed vector)
# ---------------------------------------------------------------------- #
@app.post("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, k: int = 10) -> RecommendResponse:
    if state.index is None or state.fusion is None or state.product_ids is None:
        raise HTTPException(503, "index not ready")
    with LatencyTracker("recommend") as t:
        # Demo policy: hash user_id into the embedding space deterministically.
        rng = np.random.default_rng(user_id)
        q = rng.standard_normal(state.fusion.dim).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-12
        ids, _ = state.index.search(q, k=k)
    return RecommendResponse(
        user_id=user_id,
        recommendations=[int(x) for x in ids[0] if x >= 0],
        latency_ms=t.elapsed_ms,
        cold_user=False,
    )
