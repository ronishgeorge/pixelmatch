"""Content-based feature extractor used for cold-start fallback.

When a product has no behavioural signal (no clicks, no purchases), we can
still rank it by visual + textual + categorical content alone.  This module
extracts a compact concatenated feature vector for that purpose.

The implementation is intentionally lightweight (numpy + scikit-learn) so
it can run inside unit tests with no external models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


def _color_histogram(img: Image.Image, bins: int = 8) -> np.ndarray:
    arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    feats = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0.0, 1.0), density=True)
        feats.append(h)
    return np.concatenate(feats).astype(np.float32)


@dataclass
class FeatureExtractorConfig:
    tfidf_max_features: int = 512
    color_bins: int = 8
    categorical_cols: tuple[str, ...] = ("category", "brand")


class FeatureExtractor:
    """Build a numeric content vector from a row of catalog metadata.

    Workflow
    --------
    1. ``fit(catalog_df, image_lookup)`` learns the TF-IDF vocab and the
       one-hot categorical encoder.
    2. ``transform_row(row, image)`` returns a 1-D float32 feature vector.
    3. ``transform_batch(...)`` is the vectorized equivalent.
    """

    def __init__(self, config: FeatureExtractorConfig | None = None) -> None:
        self.config = config or FeatureExtractorConfig()
        self._tfidf: TfidfVectorizer | None = None
        self._ohe: OneHotEncoder | None = None
        self._numeric_cols = ("price",)
        self._fitted = False

    # ------------------------------------------------------------------ #
    def fit(self, catalog: pd.DataFrame) -> "FeatureExtractor":
        text_corpus = (catalog["title"].fillna("") + " " + catalog["description"].fillna("")).tolist()
        self._tfidf = TfidfVectorizer(max_features=self.config.tfidf_max_features)
        self._tfidf.fit(text_corpus)

        cat_cols = [c for c in self.config.categorical_cols if c in catalog.columns]
        if cat_cols:
            try:
                self._ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:
                # older sklearn
                self._ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self._ohe.fit(catalog[cat_cols].fillna("UNK"))
            self._cat_cols = cat_cols
        else:
            self._cat_cols = []

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def _ensure_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("FeatureExtractor not fitted — call .fit() first")

    # ------------------------------------------------------------------ #
    def transform_text(self, texts: list[str]) -> np.ndarray:
        self._ensure_fit()
        assert self._tfidf is not None
        return self._tfidf.transform(texts).toarray().astype(np.float32)

    def transform_categorical(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_fit()
        if not self._cat_cols or self._ohe is None:
            return np.zeros((len(df), 0), dtype=np.float32)
        return self._ohe.transform(df[self._cat_cols].fillna("UNK")).astype(np.float32)

    def transform_numeric(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self._numeric_cols if c in df.columns]
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        # Log-scale price to compress its range
        arr = np.log1p(df[cols].fillna(0.0).to_numpy(dtype=np.float32))
        return arr

    def transform_image(self, images: list[Image.Image]) -> np.ndarray:
        return np.stack([_color_histogram(im, self.config.color_bins) for im in images])

    # ------------------------------------------------------------------ #
    def transform_batch(
        self,
        catalog: pd.DataFrame,
        images: list[Image.Image] | None = None,
    ) -> np.ndarray:
        """Return a 2-D feature matrix ``(n, d)``."""
        self._ensure_fit()
        text = self.transform_text(
            (catalog["title"].fillna("") + " " + catalog["description"].fillna("")).tolist()
        )
        cat = self.transform_categorical(catalog)
        num = self.transform_numeric(catalog)
        if images is not None:
            img = self.transform_image(images)
        else:
            img = np.zeros((len(catalog), 3 * self.config.color_bins), dtype=np.float32)
        return np.concatenate([text, cat, num, img], axis=1).astype(np.float32)
