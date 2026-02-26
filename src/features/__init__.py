"""Feature extractors with optional image dependencies."""

from __future__ import annotations

import pandas as pd

from .text_features import TextFeatureExtractor, extract_all_text_features

try:
    from .image_features import extract_label_vector
except Exception:  # pragma: no cover - optional dependency (google-cloud-vision)
    def extract_label_vector(features_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        del features_df, top_n
        raise RuntimeError("Image feature utilities are unavailable in this environment.")

# Backward compatibility: image feature extractor class was removed.
ImageFeatureExtractor = None

__all__ = [
    "TextFeatureExtractor",
    "extract_all_text_features",
    "ImageFeatureExtractor",
    "extract_label_vector",
]
