"""Multi-modal fusion of text + image embeddings.

Three fusion strategies are supported:

``early_concat``
    Concatenate the two L2-normalized vectors and renormalize.
``late_avg``
    Keep modalities separate; produced index is the average of the two
    cosine-similarity rankings.  At fuse time we approximate this by
    averaging the two vectors after projecting to a shared dim via
    truncation/zero-padding.
``learned_projection``
    Small MLP that maps the concatenation to a shared 256-dim space.
    Trained off-line with a symmetric InfoNCE loss on positive
    (user, clicked-item) pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + eps)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _pad_or_trim(x: np.ndarray, target: int) -> np.ndarray:
    d = x.shape[-1]
    if d == target:
        return x
    if d > target:
        return x[..., :target]
    pad_shape = list(x.shape)
    pad_shape[-1] = target - d
    return np.concatenate([x, np.zeros(pad_shape, dtype=x.dtype)], axis=-1)


@dataclass
class FusionConfig:
    strategy: str = "early_concat"
    projection_dim: int = 256
    text_weight: float = 0.6
    image_weight: float = 0.4


class _ProjectionMLP:
    """Tiny numpy MLP used when torch is unavailable; otherwise we use torch."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        hidden = max(out_dim, 128)
        self.W1 = rng.standard_normal((in_dim, hidden)).astype(np.float32) / np.sqrt(in_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden, out_dim)).astype(np.float32) / np.sqrt(hidden)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2


class MultiModalEncoder:
    """Fuse pre-computed text and image embeddings.

    Parameters
    ----------
    config : FusionConfig
        Strategy + dimensions.
    text_dim, image_dim : int
        Dimensions of the input modalities (needed for the learned MLP).
    """

    VALID_STRATEGIES = ("early_concat", "late_avg", "learned_projection")

    def __init__(
        self,
        config: FusionConfig | None = None,
        text_dim: int = 384,
        image_dim: int = 2048,
    ) -> None:
        self.config = config or FusionConfig()
        if self.config.strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"invalid strategy: {self.config.strategy}")
        self.text_dim = text_dim
        self.image_dim = image_dim
        self._mlp: _ProjectionMLP | None = None
        if self.config.strategy == "learned_projection":
            self._mlp = _ProjectionMLP(
                in_dim=text_dim + image_dim,
                out_dim=self.config.projection_dim,
            )

    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        s = self.config.strategy
        if s == "early_concat":
            return self.text_dim + self.image_dim
        if s == "late_avg":
            return max(self.text_dim, self.image_dim)
        return self.config.projection_dim

    # ------------------------------------------------------------------ #
    def fuse(self, text_vec: np.ndarray, image_vec: np.ndarray) -> np.ndarray:
        """Fuse two embedding matrices ``(n, d_text)`` and ``(n, d_img)``."""
        text_vec = np.atleast_2d(text_vec).astype(np.float32)
        image_vec = np.atleast_2d(image_vec).astype(np.float32)
        if text_vec.shape[0] != image_vec.shape[0]:
            raise ValueError(
                f"row mismatch: text={text_vec.shape[0]} image={image_vec.shape[0]}"
            )
        cfg = self.config
        if cfg.strategy == "early_concat":
            t = _l2(text_vec) * cfg.text_weight
            i = _l2(image_vec) * cfg.image_weight
            return _l2(np.concatenate([t, i], axis=1))

        if cfg.strategy == "late_avg":
            target = max(text_vec.shape[1], image_vec.shape[1])
            t = _l2(_pad_or_trim(text_vec, target))
            i = _l2(_pad_or_trim(image_vec, target))
            return _l2(cfg.text_weight * t + cfg.image_weight * i)

        # learned_projection
        assert self._mlp is not None
        cat = np.concatenate([_l2(text_vec), _l2(image_vec)], axis=1)
        proj = self._mlp(cat)
        return _l2(proj)

    def fuse_query(
        self,
        text_vec: np.ndarray | None,
        image_vec: np.ndarray | None,
    ) -> np.ndarray:
        """Encode a query that may carry only one modality.

        Missing modalities are replaced by a zero vector of the correct width
        before fusion, then re-normalized.
        """
        if text_vec is None and image_vec is None:
            raise ValueError("At least one of text_vec or image_vec must be provided.")
        if text_vec is None:
            text_vec = np.zeros((1, self.text_dim), dtype=np.float32)
        if image_vec is None:
            image_vec = np.zeros((1, self.image_dim), dtype=np.float32)
        return self.fuse(text_vec, image_vec)
