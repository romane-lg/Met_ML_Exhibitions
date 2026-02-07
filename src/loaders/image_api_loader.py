from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from google.cloud import vision
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


class VisionAPILoader:
    """Load raw image analysis from Google Vision API."""

    def __init__(self, credentials_path: str | None = None, max_retries: int = 2, retry_delay: float = 0.5):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            self.client = vision.ImageAnnotatorClient()

    def load_image_features(self, image_path: str, max_results: int = 10) -> dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            logger.error("Image not found: %s", path)
            return {}

        content = path.read_bytes()
        image = vision.Image(content=content)
        features = [
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=max_results),
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=max_results),
            vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            vision.Feature(type_=vision.Feature.Type.WEB_DETECTION, max_results=max_results),
            vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
        ]
        request = vision.AnnotateImageRequest(image=image, features=features)

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.annotate_image(request)
                if response.error.message:
                    logger.error("Vision API error for %s: %s", path, response.error.message)
                    return {}
                return self._to_raw_dict(response)
            except Exception as exc:
                if attempt >= self.max_retries:
                    logger.error("Vision request failed for %s: %s", path, exc)
                    return {}
                time.sleep(self.retry_delay * (attempt + 1))
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
