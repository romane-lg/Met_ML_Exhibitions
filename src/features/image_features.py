from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.nlp_utils import tokenize_text
from src.loaders import VisionAPILoader


def normalize_vision_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize_clean(text: str) -> list[str]:
    return tokenize_text(normalize_vision_text(text), min_len=1)


def clean_vision_response(features: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {
        "labels": [],
        "objects": [],
        "colors": [],
        "web_entities": [],
        "text": "",
    }

    for item in features.get("labels", []) or []:
        description = " ".join(tokenize_clean(item.get("description", "")))
        if description:
            cleaned["labels"].append({"description": description, "score": float(item.get("score", 0.0))})

    for item in features.get("objects", []) or []:
        name = " ".join(tokenize_clean(item.get("name", "")))
        if name:
            out = {"name": name, "score": float(item.get("score", 0.0))}
            if "bbox" in item:
                out["bbox"] = item["bbox"]
            cleaned["objects"].append(out)

    for item in features.get("web_entities", []) or []:
        entity = " ".join(tokenize_clean(item.get("entity", "")))
        if entity:
            cleaned["web_entities"].append({"entity": entity, "score": float(item.get("score", 0.0))})

    cleaned["text"] = " ".join(tokenize_clean(features.get("text", "")))

    for item in features.get("colors", []) or []:
        color = item.get("color", {}) or {}
        try:
            red = int(color.get("red"))
            green = int(color.get("green"))
            blue = int(color.get("blue"))
        except (TypeError, ValueError):
            continue
        cleaned["colors"].append(
            {
                "color": {"red": red, "green": green, "blue": blue},
                "score": float(item.get("score", 0.0)),
                "pixel_fraction": float(item.get("pixel_fraction", 0.0)),
            }
        )

    return cleaned


def vision_tokens_from_features(features: dict[str, Any], include_rgb: bool = True) -> list[str]:
    tokens: list[str] = []
    for item in features.get("labels", []):
        tokens.extend(tokenize_clean(item.get("description", "")))
    for item in features.get("objects", []):
        tokens.extend(tokenize_clean(item.get("name", "")))
    for item in features.get("web_entities", []):
        tokens.extend(tokenize_clean(item.get("entity", "")))
    tokens.extend(tokenize_clean(features.get("text", "")))
    if include_rgb:
        for item in features.get("colors", []):
            color = item.get("color", {})
            red = color.get("red")
            green = color.get("green")
            blue = color.get("blue")
            if red is None or green is None or blue is None:
                continue
            tokens.append(f"rgb_{int(red)}_{int(green)}_{int(blue)}")
    return list(dict.fromkeys(tokens))


def extract_numeric_features(features: dict[str, Any]) -> dict[str, float]:
    """Extract structured numeric features from raw or cleaned Vision response."""
    colors = features.get("colors", []) or []
    color_rows: list[tuple[float, float, float, float]] = []
    for item in colors:
        color = item.get("color", {}) or {}
        try:
            red = float(color.get("red"))
            green = float(color.get("green"))
            blue = float(color.get("blue"))
            weight = float(item.get("pixel_fraction", 0.0))
        except (TypeError, ValueError):
            continue
        if math.isnan(red) or math.isnan(green) or math.isnan(blue):
            continue
        color_rows.append((red, green, blue, weight))

    def _weighted_avg(idx: int) -> float:
        if not color_rows:
            return 0.0
        total_w = sum(max(row[3], 0.0) for row in color_rows)
        if total_w <= 0:
            return float(sum(row[idx] for row in color_rows) / len(color_rows))
        return float(sum(row[idx] * max(row[3], 0.0) for row in color_rows) / total_w)

    label_scores = [float(item.get("score", 0.0)) for item in (features.get("labels", []) or [])]
    object_scores = [float(item.get("score", 0.0)) for item in (features.get("objects", []) or [])]
    web_scores = [float(item.get("score", 0.0)) for item in (features.get("web_entities", []) or [])]
    text_norm = normalize_vision_text(features.get("text", ""))
    numbers = [float(token) for token in text_norm.split() if token.isdigit()]

    return {
        "vision_num_labels": float(len(label_scores)),
        "vision_num_objects": float(len(object_scores)),
        "vision_num_web_entities": float(len(web_scores)),
        "vision_label_score_mean": float(sum(label_scores) / len(label_scores)) if label_scores else 0.0,
        "vision_object_score_mean": float(sum(object_scores) / len(object_scores)) if object_scores else 0.0,
        "vision_web_score_mean": float(sum(web_scores) / len(web_scores)) if web_scores else 0.0,
        "vision_avg_red": _weighted_avg(0),
        "vision_avg_green": _weighted_avg(1),
        "vision_avg_blue": _weighted_avg(2),
        "vision_ocr_number_count": float(len(numbers)),
        "vision_ocr_number_mean": float(sum(numbers) / len(numbers)) if numbers else 0.0,
    }


class ImageFeatureExtractor:
    """Adapter that loads raw Vision response and returns cleaned features."""

    def __init__(self, credentials_path: str | None = None):
        self.loader = VisionAPILoader(credentials_path=credentials_path)

    def extract_features(self, image_path: str, max_results: int = 10) -> dict[str, Any]:
        raw = self.loader.load_image_features(image_path, max_results=max_results)
        return clean_vision_response(raw)

    def batch_extract(
        self,
        image_paths: list[str],
        max_results: int = 10,
        save_path: str | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for path in image_paths:
            item = self.extract_features(path, max_results=max_results)
            item["image_path"] = str(path)
            rows.append(item)
        frame = pd.DataFrame(rows)
        if save_path:
            frame.to_pickle(save_path)
        return frame


def extract_label_vector(features_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    from collections import Counter

    all_labels: list[str] = []
    for labels in features_df["labels"]:
        if labels:
            all_labels.extend([item["description"] for item in labels])
    top_labels = [label for label, _ in Counter(all_labels).most_common(top_n)]

    vectors = []
    for labels in features_df["labels"]:
        vector = {label: 0 for label in top_labels}
        if labels:
            for item in labels:
                if item["description"] in top_labels:
                    vector[item["description"]] = item["score"]
        vectors.append(vector)
    return pd.DataFrame(vectors)


if __name__ == "__main__":
    extractor = ImageFeatureExtractor()
    sample = Path("data/raw/images/398746.jpg")
    if sample.exists():
        out = extractor.extract_features(str(sample))
        print("labels", out.get("labels", [])[:3])
