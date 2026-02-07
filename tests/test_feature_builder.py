import pandas as pd

from scripts.build_features import (
    build_text,
    is_supported_image_file,
    resolve_image_path,
    tokenize_local,
)
from src.features.image_features import vision_tokens_from_features


def test_build_text_includes_fields():
    row = pd.Series(
        {
            "title": "Head",
            "artist": "Unknown",
            "department": "Egyptian Art",
            "objectDate": "100",
            "medium": "Stone",
            "description": "Ancient sculpture",
        }
    )
    text = build_text(row)
    assert "Head" in text
    assert "Ancient sculpture" in text


def test_tokenize_local_basic():
    out = tokenize_local("Ancient/Egypt;Stone")
    assert out == ["ancient", "egypt", "stone"]


def test_vision_tokens_from_all_feature_types():
    features = {
        "labels": [{"description": "Sculpture"}],
        "objects": [{"name": "Human face"}],
        "web_entities": [{"entity": "Ancient Egypt"}],
        "text": "Dynasty XIX",
        "colors": [{"color": {"red": 12, "green": 34, "blue": 56}}],
    }
    out = vision_tokens_from_features(features)
    assert "sculpture" in out
    assert "human" in out
    assert "face" in out
    assert "ancient" in out
    assert "egypt" in out
    assert "dynasty" in out
    assert "xix" in out
    assert "rgb_12_34_56" in out


def test_resolve_image_path_from_images_prefix():
    path = resolve_image_path("images/123.jpg", "data/raw/images")
    assert str(path).replace("\\", "/").endswith("data/raw/images/123.jpg")


def test_is_supported_image_file(tmp_path):
    jpg = tmp_path / "ok.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 20)
    html = tmp_path / "bad.jpg"
    html.write_text("<!DOCTYPE html>", encoding="utf-8")
    assert is_supported_image_file(jpg) is True
    assert is_supported_image_file(html) is False
