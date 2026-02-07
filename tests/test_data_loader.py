import pandas as pd

from src.data import get_data_summary, validate_data


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
