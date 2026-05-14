"""Two-tower neural recommender (a la YouTubeDNN, Covington et al. 2016).

The model embeds users and items into a shared space and is trained with
**in-batch sampled softmax**: positive (user, item) pairs from a mini-batch
are scored against every item in the same batch as in-batch negatives.

Item representations may be initialized from pre-computed content
embeddings (text + image fused) so that the model generalizes to cold
items at inference time — a key requirement of this project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


@dataclass
class TwoTowerConfig:
    num_users: int = 50_000
    item_input_dim: int = 384         # content-embedding dim feeding the item tower
    user_emb_dim: int = 64
    hidden_dim: int = 256
    output_dim: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 1024
    epochs: int = 3
    temperature: float = 0.07


if _HAS_TORCH:

    class TwoTowerModel(nn.Module):
        """PyTorch two-tower model.  User tower is an embedding + MLP,
        item tower is an MLP over pre-computed content embeddings."""

        def __init__(self, config: TwoTowerConfig) -> None:
            super().__init__()
            self.config = config
            self.user_embedding = nn.Embedding(config.num_users, config.user_emb_dim)
            self.user_mlp = nn.Sequential(
                nn.Linear(config.user_emb_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim),
            )
            self.item_mlp = nn.Sequential(
                nn.Linear(config.item_input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim),
            )

        def encode_user(self, user_ids: "torch.Tensor") -> "torch.Tensor":
            e = self.user_embedding(user_ids)
            v = self.user_mlp(e)
            return nn.functional.normalize(v, dim=-1)

        def encode_item(self, item_feats: "torch.Tensor") -> "torch.Tensor":
            v = self.item_mlp(item_feats)
            return nn.functional.normalize(v, dim=-1)

        def forward(self, user_ids, item_feats):  # noqa: D401
            return self.encode_user(user_ids), self.encode_item(item_feats)

else:  # pragma: no cover

    class TwoTowerModel:  # type: ignore
        """Placeholder used when torch is unavailable (CI without torch wheel)."""

        def __init__(self, config: TwoTowerConfig) -> None:
            raise RuntimeError("PyTorch is required for TwoTowerModel")


class TwoTowerRecommender:
    """Train + serve a two-tower recommender.

    The class works in two modes:

    * **Trained** — call :meth:`fit` with ``(user_ids, item_ids)`` positives
      and a matrix of item content features.
    * **Lookup-only** — initialize from already computed user/item
      embeddings via :meth:`load_embeddings` and just call :meth:`recommend`.
    """

    def __init__(self, config: TwoTowerConfig | None = None, device: str = "cpu") -> None:
        self.config = config or TwoTowerConfig()
        self.device = device
        self.model: TwoTowerModel | None = None
        self.user_embeddings: np.ndarray | None = None
        self.item_embeddings: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def fit(
        self,
        user_ids: np.ndarray,
        item_indices: np.ndarray,
        item_features: np.ndarray,
        seed: int = 42,
    ) -> "TwoTowerRecommender":
        """Train the two-tower model with in-batch sampled softmax.

        Parameters
        ----------
        user_ids : (N,) int array
            User ids for each interaction.
        item_indices : (N,) int array
            Row index into ``item_features`` for each interaction's item.
        item_features : (M, D) float array
            Pre-computed content embeddings for all items.
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required to fit TwoTowerRecommender")

        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = self.config
        cfg.item_input_dim = item_features.shape[1]
        cfg.num_users = int(user_ids.max()) + 1
        self.model = TwoTowerModel(cfg).to(self.device)

        item_feats_t = torch.tensor(item_features, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        n = len(user_ids)
        idx = np.arange(n)
        for epoch in range(cfg.epochs):
            np.random.shuffle(idx)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, cfg.batch_size):
                batch = idx[start : start + cfg.batch_size]
                u = torch.tensor(user_ids[batch], dtype=torch.long, device=self.device)
                i = torch.tensor(item_indices[batch], dtype=torch.long, device=self.device)
                item_in = item_feats_t[i]

                u_emb, i_emb = self.model(u, item_in)
                logits = (u_emb @ i_emb.T) / cfg.temperature
                target = torch.arange(u_emb.shape[0], device=self.device)
                loss = nn.functional.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
            logger.info("epoch %d/%d  loss=%.4f", epoch + 1, cfg.epochs, total_loss / max(n_batches, 1))

        # Cache full item embedding matrix for inference
        with torch.no_grad():
            self.item_embeddings = self.model.encode_item(item_feats_t).cpu().numpy()
        return self

    # ------------------------------------------------------------------ #
    def load_embeddings(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        item_ids: np.ndarray,
    ) -> None:
        self.user_embeddings = user_embeddings.astype(np.float32)
        self.item_embeddings = item_embeddings.astype(np.float32)
        self.item_ids = item_ids.astype(np.int64)

    # ------------------------------------------------------------------ #
    def _user_vec(self, user_id: int) -> np.ndarray:
        if self.user_embeddings is not None:
            return self.user_embeddings[user_id]
        if not _HAS_TORCH or self.model is None:
            raise RuntimeError("Model not trained and no embeddings loaded.")
        with torch.no_grad():
            v = self.model.encode_user(torch.tensor([user_id], device=self.device))
        return v.cpu().numpy().ravel()

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        if self.item_embeddings is None:
            raise RuntimeError("Call fit() or load_embeddings() first")
        u = self._user_vec(user_id)
        scores = self.item_embeddings @ u
        top = np.argsort(-scores)[:k]
        if self.item_ids is not None:
            return self.item_ids[top].tolist()
        return top.tolist()

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        if not _HAS_TORCH or self.model is None:
            raise RuntimeError("Nothing to save — model not trained")
        torch.save(
            {"state_dict": self.model.state_dict(), "config": self.config.__dict__},
            path,
        )

    def load(self, path: str) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("torch required to load checkpoint")
        ckpt = torch.load(path, map_location=self.device)
        self.config = TwoTowerConfig(**ckpt["config"])
        self.model = TwoTowerModel(self.config).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
