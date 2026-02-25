"""Feature extractors with optional image dependencies."""

from .text_features import TextFeatureExtractor, extract_all_text_features

try:
    from .image_features import ImageFeatureExtractor, extract_label_vector
except Exception:  # pragma: no cover - optional dependency (google-cloud-vision)
    ImageFeatureExtractor = None  # type: ignore[assignment]
    extract_label_vector = None  # type: ignore[assignment]

__all__ = [
    "TextFeatureExtractor",
    "extract_all_text_features",
    "ImageFeatureExtractor",
    "extract_label_vector",
]
