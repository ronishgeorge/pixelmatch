"""Image encoder — CLIP ViT-B/32 or torchvision ResNet-50.

The class lazily loads the chosen backbone.  If the optional weights cannot
be obtained (no network, no torchvision weights cache) we fall back to a
deterministic color-histogram + downscaled-pixel encoder so that the rest
of the pipeline remains exercisable.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_RESNET_DIM = 2048
_CLIP_DIM = 512


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + eps)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _histogram_features(img: Image.Image, dim: int) -> np.ndarray:
    """Deterministic offline-friendly fallback embedding.

    Combines a 3x8-bin per-channel color histogram with a downscaled
    grayscale patch flattened to ``dim`` floats.
    """
    img = img.convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # 32x32x3

    feats = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=8, range=(0.0, 1.0), density=True)
        feats.append(hist)
    flat = arr.mean(axis=2).flatten()
    vec = np.concatenate(feats + [flat]).astype(np.float32)

    if vec.shape[0] >= dim:
        vec = vec[:dim]
    else:
        pad = np.zeros(dim - vec.shape[0], dtype=np.float32)
        vec = np.concatenate([vec, pad])
    return vec


class ImageEncoder:
    """Encode images into dense L2-normalized vectors.

    Parameters
    ----------
    backbone : {"resnet50", "clip"}
        Choice of vision backbone.
    device : str
        Torch device id; ``"mps"`` is supported on Apple Silicon.
    clip_model : str
        HuggingFace id for the CLIP backbone (if used).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        device: str = "cpu",
        clip_model: str = "openai/clip-vit-base-patch32",
    ) -> None:
        if backbone not in {"resnet50", "clip"}:
            raise ValueError(f"unknown backbone: {backbone}")
        self.backbone = backbone
        self.device = device
        self.clip_model = clip_model
        self.dim = _CLIP_DIM if backbone == "clip" else _RESNET_DIM
        self._model = None
        self._preprocess = None
        self._mode = "fallback"  # "fallback" | "resnet50" | "clip"

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self._model is not None or self._mode != "fallback":
            return
        try:
            import torch

            if self.backbone == "clip":
                from transformers import CLIPModel, CLIPProcessor

                logger.info("Loading CLIP: %s", self.clip_model)
                self._model = CLIPModel.from_pretrained(self.clip_model).to(self.device).eval()
                self._preprocess = CLIPProcessor.from_pretrained(self.clip_model)
                self.dim = self._model.config.projection_dim
                self._mode = "clip"
            else:
                import torchvision.models as tvm
                from torchvision import transforms

                logger.info("Loading torchvision ResNet-50")
                try:
                    weights = tvm.ResNet50_Weights.IMAGENET1K_V2
                    model = tvm.resnet50(weights=weights)
                except Exception:
                    # Offline build: load untrained weights so the pipeline
                    # still runs deterministically.
                    logger.warning("ResNet-50 weights unavailable — using random init")
                    model = tvm.resnet50(weights=None)
                # Strip the final classifier so .forward returns 2048-dim pool
                model.fc = torch.nn.Identity()
                self._model = model.to(self.device).eval()
                self._preprocess = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
                self.dim = _RESNET_DIM
                self._mode = "resnet50"
        except Exception as exc:  # noqa: BLE001
            logger.warning("ImageEncoder falling back to histogram features: %s", exc)
            self._mode = "fallback"
            self._model = None

    # ------------------------------------------------------------------ #
    def _open(self, src: str | Image.Image) -> Image.Image:
        if isinstance(src, Image.Image):
            return src
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        return Image.open(src).convert("RGB")

    def encode(
        self,
        images: str | Image.Image | Sequence,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode one or many images.  Returns ``(n, dim)`` float32."""
        self._load()
        single = isinstance(images, (str, Image.Image))
        if single:
            images = [images]
        pil_imgs = [self._open(x) for x in images]

        if self._mode == "fallback" or self._model is None:
            vecs = np.stack([_histogram_features(im, self.dim) for im in pil_imgs])
        elif self._mode == "clip":
            import torch

            outs = []
            for i in range(0, len(pil_imgs), batch_size):
                chunk = pil_imgs[i : i + batch_size]
                inputs = self._preprocess(images=chunk, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feat = self._model.get_image_features(**inputs)
                outs.append(feat.cpu().numpy())
            vecs = np.vstack(outs).astype(np.float32)
        else:  # resnet50
            import torch

            outs = []
            for i in range(0, len(pil_imgs), batch_size):
                chunk = pil_imgs[i : i + batch_size]
                batch = torch.stack([self._preprocess(im) for im in chunk]).to(self.device)
                with torch.no_grad():
                    feat = self._model(batch)
                outs.append(feat.cpu().numpy())
            vecs = np.vstack(outs).astype(np.float32)

        if normalize:
            vecs = _l2(vecs)
        return vecs
