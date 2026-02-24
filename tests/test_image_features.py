from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.image_features import (
    ImageFeatureExtractor,
    clean_vision_response,
    extract_numeric_features,
    extract_label_vector,
    normalize_vision_text,
    vision_tokens_from_features,
)


def test_normalize_vision_text() -> None:
    out = normalize_vision_text("Thé MET—Museum! 1880–1890")
    assert out == "th met museum 1880 1890"


def test_clean_vision_response_removes_stopwords_and_invalid_colors() -> None:
    raw = {
        "labels": [{"description": "The Ancient Art", "score": 0.9}],
        "objects": [{"name": "A Human Figure", "score": 0.8, "bbox": [(0.1, 0.2)]}],
        "web_entities": [{"entity": "The Metropolitan Museum of Art", "score": 0.7}],
        "text": "The king of Egypt",
        "colors": [
            {"color": {"red": "12", "green": 34, "blue": 56}, "score": 0.4, "pixel_fraction": 0.3},
            {"color": {"red": None, "green": 1, "blue": 2}, "score": 0.2, "pixel_fraction": 0.1},
        ],
    }
    cleaned = clean_vision_response(raw)
    assert cleaned["labels"][0]["description"] == "ancient art"
    assert cleaned["objects"][0]["name"] == "human figure"
    assert cleaned["web_entities"][0]["entity"] == "metropolitan museum art"
    assert cleaned["text"] == "king egypt"
    assert len(cleaned["colors"]) == 1


def test_vision_tokens_from_features() -> None:
    features = {
        "labels": [{"description": "ancient sculpture"}],
        "objects": [{"name": "human face"}],
        "web_entities": [{"entity": "egypt art"}],
        "text": "dynasty xix",
        "colors": [{"color": {"red": 12, "green": 34, "blue": 56}}],
    }
    out = vision_tokens_from_features(features)
    assert "ancient" in out
    assert "sculpture" in out
    assert "human" in out
    assert "face" in out
    assert "rgb_12_34_56" in out


def test_image_feature_extractor_uses_loader(monkeypatch) -> None:
    class _FakeLoader:
        def __init__(self, credentials_path: str | None = None):
            self.credentials_path = credentials_path

        def load_image_features(self, image_path: str, max_results: int = 10) -> dict:
            return {"labels": [{"description": "The Statue", "score": 0.9}]}

    monkeypatch.setattr("src.features.image_features.VisionAPILoader", _FakeLoader)
    extractor = ImageFeatureExtractor(credentials_path="x.json")
    out = extractor.extract_features("dummy.jpg")
    assert out["labels"][0]["description"] == "statue"


def test_batch_extract_and_save(tmp_path: Path, monkeypatch) -> None:
    class _FakeLoader:
        def __init__(self, credentials_path: str | None = None):
            pass

        def load_image_features(self, image_path: str, max_results: int = 10) -> dict:
            return {"labels": [{"description": Path(image_path).stem, "score": 0.9}]}

    monkeypatch.setattr("src.features.image_features.VisionAPILoader", _FakeLoader)
    extractor = ImageFeatureExtractor()
    out_path = tmp_path / "features.pkl"
    frame = extractor.batch_extract(["a.jpg", "b.jpg"], save_path=str(out_path))
    assert len(frame) == 2
    assert out_path.exists()


def test_extract_label_vector() -> None:
    frame = pd.DataFrame(
        {
            "labels": [
                [{"description": "statue", "score": 0.9}, {"description": "stone", "score": 0.8}],
                [{"description": "stone", "score": 0.7}],
            ]
        }
    )
    vec = extract_label_vector(frame, top_n=2)
    assert "statue" in vec.columns
    assert "stone" in vec.columns
    assert vec.shape == (2, 2)


def test_extract_numeric_features_keeps_rgb_and_scores() -> None:
    raw = {
        "labels": [{"description": "statue", "score": 0.8}],
        "objects": [{"name": "person", "score": 0.7}],
        "web_entities": [{"entity": "egypt", "score": 0.6}],
        "text": "dynasty 1234",
        "colors": [{"color": {"red": 10, "green": 20, "blue": 30}, "pixel_fraction": 1.0}],
    }
    out = extract_numeric_features(raw)
    assert out["vision_avg_red"] == 10.0
    assert out["vision_avg_green"] == 20.0
    assert out["vision_avg_blue"] == 30.0
    assert out["vision_label_score_mean"] == 0.8
    assert out["vision_ocr_number_count"] == 1.0
    assert out["vision_ocr_number_mean"] == 1234.0
