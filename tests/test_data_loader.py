import pandas as pd
import pytest

from src.data import (
    filter_by_department,
    get_data_summary,
    get_image_path,
    load_met_data,
    validate_data,
)


def test_validate_data_removes_duplicates():
    frame = pd.DataFrame(
        {
            "objectID": [1, 1, 2],
            "title": ["a", "a", "b"],
            "image_path": ["images/1.jpg", "images/1.jpg", "images/2.jpg"],
        }
    )
    out = validate_data(frame)
    assert len(out) == 2


def test_summary_counts_fields():
    frame = pd.DataFrame(
        {
            "objectID": [1, 2],
            "title": ["a", "b"],
            "artist": ["x", None],
            "department": ["d1", "d2"],
            "objectDate": ["1900", "1901"],
            "medium": ["oil", "ink"],
            "image_path": ["images/1.jpg", "images/2.jpg"],
        }
    )
    out = get_data_summary(frame)
    assert out["total_artworks"] == 2
    assert out["departments"] == 2


def test_validate_data_missing_required_columns_raises():
    frame = pd.DataFrame({"objectID": [1], "title": ["a"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(frame)


def test_load_met_data_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_met_data(data_path="does/not/exist.csv")


def test_load_met_data_validate_false_keeps_duplicates(tmp_path):
    csv_path = tmp_path / "met.csv"
    pd.DataFrame(
        {
            "objectID": [1, 1],
            "title": ["a", "a"],
            "image_path": ["images/1.jpg", "images/1.jpg"],
            "artist": ["x", "x"],
            "department": ["d1", "d1"],
            "objectDate": ["1900", "1900"],
        }
    ).to_csv(csv_path, index=False)
    out = load_met_data(data_path=str(csv_path), validate=False)
    assert len(out) == 2


def test_get_image_path_exists_and_missing(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    found = images_dir / "10.jpg"
    found.write_bytes(b"\xff\xd8\xff")
    assert get_image_path(10, images_dir=str(images_dir)) == found
    assert get_image_path(999, images_dir=str(images_dir)) is None


def test_filter_by_department_returns_subset():
    frame = pd.DataFrame(
        {
            "objectID": [1, 2, 3],
            "title": ["a", "b", "c"],
            "image_path": ["images/1.jpg", "images/2.jpg", "images/3.jpg"],
            "department": ["d1", "d2", "d1"],
        }
    )
    out = filter_by_department(frame, ["d1"])
    assert set(out["objectID"]) == {1, 3}
