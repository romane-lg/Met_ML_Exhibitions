from __future__ import annotations

from pathlib import Path

from src.loaders.image_api_loader import VisionAPILoader


def test_missing_image_returns_empty(tmp_path: Path) -> None:
    loader = VisionAPILoader.__new__(VisionAPILoader)
    loader.max_retries = 0
    loader.retry_delay = 0.0
    out = loader.load_image_features(str(tmp_path / "missing.jpg"))
    assert out == {}


def test_to_raw_dict_maps_fields() -> None:
    label = type("L", (), {"description": "statue", "score": 0.9})()
    vertex = type("V", (), {"x": 0.1, "y": 0.2})()
    obj = type(
        "O",
        (),
        {
            "name": "person",
            "score": 0.8,
            "bounding_poly": type("B", (), {"normalized_vertices": [vertex]})(),
        },
    )()
    color_item = type(
        "C",
        (),
        {
            "color": type("Rgb", (), {"red": 1, "green": 2, "blue": 3})(),
            "score": 0.5,
            "pixel_fraction": 0.4,
        },
    )()
    response = type(
        "R",
        (),
        {
            "label_annotations": [label],
            "localized_object_annotations": [obj],
            "image_properties_annotation": type(
                "A", (), {"dominant_colors": type("D", (), {"colors": [color_item]})()}
            )(),
            "web_detection": type(
                "W",
                (),
                {"web_entities": [type("E", (), {"description": "egypt", "score": 0.7})()]},
            )(),
            "text_annotations": [type("T", (), {"description": "Dynasty XIX"})()],
        },
    )()

    out = VisionAPILoader._to_raw_dict(response)
    assert out["labels"][0]["description"] == "statue"
    assert out["objects"][0]["name"] == "person"
    assert out["colors"][0]["color"]["red"] == 1
    assert out["web_entities"][0]["entity"] == "egypt"
    assert out["text"] == "Dynasty XIX"
