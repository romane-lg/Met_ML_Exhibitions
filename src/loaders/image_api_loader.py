from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VisionAPILoader:
    """Stub – Google Vision API support has been removed.

    This class is kept for backward compatibility with any code that imports it,
    but instantiation will raise a RuntimeError.  The pure data-conversion helper
    ``_to_raw_dict`` is preserved for tests that use mock response objects.
    """

    def __init__(self, credentials_path: str | None = None, max_retries: int = 2, retry_delay: float = 0.5):
        raise RuntimeError(
            "Google Vision API support has been removed from this project. "
            "Remove any code that constructs VisionAPILoader."
        )

    def load_image_features(self, image_path: str, max_results: int = 10) -> dict[str, Any]:
        """Always returns an empty dict (no API calls are made)."""
        path = Path(image_path)
        if not path.exists():
            logger.warning("Image not found: %s", path)
            return {}
        return {}

    @staticmethod
    def _to_raw_dict(response: Any) -> dict[str, Any]:
        labels = [{"description": item.description, "score": item.score} for item in response.label_annotations]
        objects = [
            {
                "name": item.name,
                "score": item.score,
                "bbox": [(vertex.x, vertex.y) for vertex in item.bounding_poly.normalized_vertices],
            }
            for item in response.localized_object_annotations
        ]
        colors = []
        if response.image_properties_annotation:
            for item in response.image_properties_annotation.dominant_colors.colors[:5]:
                colors.append(
                    {
                        "color": {
                            "red": item.color.red,
                            "green": item.color.green,
                            "blue": item.color.blue,
                        },
                        "score": item.score,
                        "pixel_fraction": item.pixel_fraction,
                    }
                )
        web_entities = []
        if response.web_detection and response.web_detection.web_entities:
            for item in response.web_detection.web_entities:
                if item.description:
                    web_entities.append({"entity": item.description, "score": item.score})
        text = response.text_annotations[0].description if response.text_annotations else ""
        return {
            "labels": labels,
            "objects": objects,
            "colors": colors,
            "web_entities": web_entities,
            "text": text,
        }
