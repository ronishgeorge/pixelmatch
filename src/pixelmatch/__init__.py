"""PixelMatch — Multi-Modal Visual Search & Recommendation Engine.

A research-grade reference implementation of:
  * Multi-modal (text + image) product embeddings
  * Approximate nearest neighbor retrieval via FAISS HNSW
  * Two-tower neural recommendation with cold-start fallback
  * Learning-to-rank re-ranking via LambdaMART (LightGBM)
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__", "load_config"]


def load_config(path: str | None = None) -> dict:
    """Load the project YAML config (``params.yaml``) as a dict.

    Parameters
    ----------
    path : str, optional
        Path to a YAML config file.  If ``None``, the project default
        (``params.yaml`` at the repo root) is used.
    """
    import os

    import yaml

    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        # src/pixelmatch -> repo root is two levels up
        path = os.path.normpath(os.path.join(here, "..", "..", "params.yaml"))
    with open(path) as fh:
        return yaml.safe_load(fh)
