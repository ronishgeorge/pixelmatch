"""Generate synthetic user-product interactions for PixelMatch.

The distribution is intentionally heavy-tailed (Zipfian over products) so
the collaborative-filtering baselines behave realistically, *and* a fixed
fraction of products is held cold (< 5 interactions) for cold-start eval.

Output: ``data/interactions.csv`` with columns
``[user_id, product_id, interaction_type, rating, timestamp]``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INTERACTION_TYPES = ("view", "click", "add_to_cart", "purchase")
TYPE_WEIGHTS = (0.70, 0.20, 0.07, 0.03)  # rough funnel
RATING_MAP = {"view": 1.0, "click": 2.0, "add_to_cart": 3.5, "purchase": 5.0}


def _zipf_sample(num_items: int, n_draws: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Sample item ids with a Zipf-like power-law: P(rank=r) ∝ 1 / r^alpha."""
    ranks = np.arange(1, num_items + 1, dtype=np.float64)
    probs = 1.0 / np.power(ranks, alpha)
    probs /= probs.sum()
    # Item id = rank - 1 mapped through a fixed permutation so popular ids
    # are not contiguous (more realistic).
    perm = rng.permutation(num_items)
    chosen_ranks = rng.choice(num_items, size=n_draws, replace=True, p=probs)
    return perm[chosen_ranks]


def generate_interactions(
    num_users: int = 50_000,
    num_products: int = 100_000,
    num_interactions: int = 1_000_000,
    cold_product_fraction: float = 0.10,
    zipf_alpha: float = 1.15,
    output_dir: str = "data",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the interactions table and persist ``interactions.csv``."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    n_cold = int(num_products * cold_product_fraction)
    cold_ids = rng.choice(num_products, size=n_cold, replace=False)
    cold_set = set(cold_ids.tolist())
    warm_pool = np.setdiff1d(np.arange(num_products), cold_ids, assume_unique=True)
    logger.info("Reserved %d cold products, %d warm", n_cold, len(warm_pool))

    # User activity distribution: also Zipfian — a few heavy users
    user_activity = rng.zipf(a=1.5, size=num_interactions) - 1
    user_ids = (user_activity % num_users).astype(np.int64)

    # Sample products from warm pool only (cold by definition has <5 interactions).
    # We deliberately allow up to 4 interactions per cold product later.
    warm_draws = _zipf_sample(len(warm_pool), num_interactions, alpha=zipf_alpha, rng=rng)
    product_ids = warm_pool[warm_draws]

    # Sprinkle in <5 interactions for some cold products (so they're not literally zero).
    cold_extra = min(int(num_interactions * 0.005), n_cold * 4)
    if cold_extra > 0:
        cold_pids = rng.choice(cold_ids, size=cold_extra, replace=True)
        cold_uids = rng.integers(0, num_users, size=cold_extra)
        user_ids = np.concatenate([user_ids, cold_uids])
        product_ids = np.concatenate([product_ids, cold_pids])

    n = len(user_ids)
    types_idx = rng.choice(len(INTERACTION_TYPES), size=n, p=TYPE_WEIGHTS)
    types = np.asarray(INTERACTION_TYPES)[types_idx]
    ratings = np.asarray([RATING_MAP[t] for t in types], dtype=np.float32)

    now = int(time.time())
    # Spread timestamps across the last 180 days
    offsets = rng.integers(0, 180 * 24 * 3600, size=n)
    timestamps = now - offsets

    df = pd.DataFrame(
        {
            "user_id": user_ids.astype(np.int32),
            "product_id": product_ids.astype(np.int32),
            "interaction_type": types,
            "rating": ratings,
            "timestamp": timestamps.astype(np.int64),
        }
    )
    # Mark cold products
    df["is_cold_product"] = df["product_id"].isin(cold_set)

    out_path = os.path.join(output_dir, "interactions.csv")
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d interactions to %s", len(df), out_path)

    # Save the cold-id list for downstream cold-start evaluation
    np.save(os.path.join(output_dir, "cold_product_ids.npy"), cold_ids)
    return df


# ---------------------------------------------------------------------- #
def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate PixelMatch synthetic interactions")
    p.add_argument("--users", type=int, default=50_000)
    p.add_argument("--products", type=int, default=100_000)
    p.add_argument("--num", type=int, default=1_000_000)
    p.add_argument("--cold-fraction", type=float, default=0.10)
    p.add_argument("--zipf-alpha", type=float, default=1.15)
    p.add_argument("--output-dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_interactions(
        num_users=args.users,
        num_products=args.products,
        num_interactions=args.num,
        cold_product_fraction=args.cold_fraction,
        zipf_alpha=args.zipf_alpha,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
