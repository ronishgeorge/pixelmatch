"""Tests for the encoder modules."""

from __future__ import annotations

import numpy as np
import pytest

from pixelmatch.encoders.feature_extractor import FeatureExtractor
from pixelmatch.encoders.image_encoder import ImageEncoder
from pixelmatch.encoders.multimodal_encoder import FusionConfig, MultiModalEncoder
from pixelmatch.encoders.text_encoder import TextEncoder, _HashingFallback, _l2_normalize


# ---------------- TextEncoder ---------------- #
def test_text_encoder_returns_2d_normalized():
    enc = TextEncoder(model_name="__forced_fallback__", dim=64)
    # Force the fallback by ensuring the model loader raises
    enc._model = _HashingFallback(dim=64)
    out = enc.encode(["hello world", "another"])
    assert out.shape == (2, 64)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_text_encoder_single_string():
    enc = TextEncoder(model_name="__forced_fallback__", dim=32)
    enc._model = _HashingFallback(dim=32)
    out = enc.encode("a single sentence")
    assert out.shape == (1, 32)


def test_text_encoder_encode_iter_handles_empty():
    enc = TextEncoder(model_name="__forced_fallback__", dim=16)
    enc._model = _HashingFallback(dim=16)
    out = enc.encode_iter([])
    assert out.shape == (0, 16)


def test_text_encoder_deterministic_with_fallback():
    enc = TextEncoder(model_name="__forced_fallback__", dim=32)
    enc._model = _HashingFallback(dim=32)
    a = enc.encode(["the quick brown fox"])
    b = enc.encode(["the quick brown fox"])
    assert np.allclose(a, b)


def test_l2_normalize_handles_zero_vec():
    z = np.zeros((1, 4), dtype=np.float32)
    out = _l2_normalize(z)
    assert np.isfinite(out).all()


# ---------------- ImageEncoder ---------------- #
def test_image_encoder_fallback_runs(sample_image):
    enc = ImageEncoder(backbone="resnet50")
    # Force fallback to avoid downloading torchvision weights in tests
    enc._mode = "fallback"
    enc._model = None
    out = enc.encode(str(sample_image))
    assert out.ndim == 2
    assert out.shape[0] == 1
    assert np.isfinite(out).all()


def test_image_encoder_batch(sample_image):
    enc = ImageEncoder(backbone="resnet50")
    enc._mode = "fallback"
    out = enc.encode([str(sample_image), str(sample_image)])
    assert out.shape[0] == 2


def test_image_encoder_normalizes(sample_image):
    enc = ImageEncoder(backbone="resnet50")
    enc._mode = "fallback"
    out = enc.encode(str(sample_image), normalize=True)
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-4


def test_image_encoder_invalid_backbone_rejected():
    with pytest.raises(ValueError):
        ImageEncoder(backbone="garbage")


# ---------------- MultiModalEncoder ---------------- #
def test_fusion_early_concat(sample_embeddings):
    t, i = sample_embeddings
    enc = MultiModalEncoder(FusionConfig(strategy="early_concat"), text_dim=16, image_dim=32)
    out = enc.fuse(t, i)
    assert out.shape == (4, 16 + 32)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)


def test_fusion_late_avg(sample_embeddings):
    t, i = sample_embeddings
    enc = MultiModalEncoder(FusionConfig(strategy="late_avg"), text_dim=16, image_dim=32)
    out = enc.fuse(t, i)
    assert out.shape == (4, max(16, 32))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)


def test_fusion_learned_projection(sample_embeddings):
    t, i = sample_embeddings
    enc = MultiModalEncoder(
        FusionConfig(strategy="learned_projection", projection_dim=24),
        text_dim=16,
        image_dim=32,
    )
    out = enc.fuse(t, i)
    assert out.shape == (4, 24)


def test_fusion_query_with_only_text(sample_embeddings):
    t, _ = sample_embeddings
    enc = MultiModalEncoder(FusionConfig(strategy="early_concat"), text_dim=16, image_dim=32)
    out = enc.fuse_query(text_vec=t[:1], image_vec=None)
    assert out.shape == (1, 48)


def test_fusion_query_requires_modality():
    enc = MultiModalEncoder(FusionConfig(strategy="early_concat"), text_dim=16, image_dim=32)
    with pytest.raises(ValueError):
        enc.fuse_query(text_vec=None, image_vec=None)


def test_invalid_fusion_strategy():
    with pytest.raises(ValueError):
        MultiModalEncoder(FusionConfig(strategy="not_a_strategy"))


# ---------------- FeatureExtractor ---------------- #
def test_feature_extractor_fits_and_transforms(tiny_catalog):
    fe = FeatureExtractor()
    fe.fit(tiny_catalog)
    X = fe.transform_batch(tiny_catalog)
    assert X.shape[0] == len(tiny_catalog)
    assert X.shape[1] > 0
    assert np.isfinite(X).all()


def test_feature_extractor_requires_fit(tiny_catalog):
    fe = FeatureExtractor()
    with pytest.raises(RuntimeError):
        fe.transform_text(["x"])
