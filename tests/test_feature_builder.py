import pickle

import numpy as np
import pandas as pd
import pytest

from scripts.build_features import (
    atomic_pickle_dump,  # wrapper import preserved for CLI compatibility
)
from src.features.build_pipeline import (
    build_combined_embeddings_payload,
    build_numeric_feature_matrix,
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


def test_resolve_image_path_from_project_root_prefix():
    """CSV stores full relative paths like 'data/raw/images/398746.jpg'."""
    from pathlib import Path
    path = resolve_image_path("data/raw/images/123.jpg", "/some/project/data/raw/images")
    assert path == Path.cwd() / "data" / "raw" / "images" / "123.jpg"


def test_resolve_image_path_windows_backslashes_on_any_platform():
    """CSV written on Windows uses backslashes; must resolve correctly on Mac/Linux too."""
    from pathlib import Path
    path = resolve_image_path("data\\raw\\images\\123.jpg", "/some/project/data/raw/images")
    assert path == Path.cwd() / "data" / "raw" / "images" / "123.jpg"


def test_resolve_image_path_bare_filename():
    path = resolve_image_path("123.jpg", "data/raw/images")
    assert str(path).replace("\\", "/").endswith("data/raw/images/123.jpg")


def test_is_supported_image_file(tmp_path):
    jpg = tmp_path / "ok.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 20)
    html = tmp_path / "bad.jpg"
    html.write_text("<!DOCTYPE html>", encoding="utf-8")
    assert is_supported_image_file(jpg) is True
    assert is_supported_image_file(html) is False


def test_build_numeric_feature_matrix_aligns_and_fills_missing():
    object_ids = np.array([2, 1, 3], dtype=np.int64)
    numeric_rows = [
        {"objectID": 1, "meta_has_year": 1.0, "vision_num_labels": 3.0},
        {"objectID": 3, "meta_has_year": 0.0, "vision_num_labels": 1.0},
    ]
    matrix, cols = build_numeric_feature_matrix(object_ids, numeric_rows)
    col_to_idx = {name: idx for idx, name in enumerate(cols)}

    assert matrix.shape == (3, 2)
    assert np.isclose(matrix[0, col_to_idx["meta_has_year"]], 0.0)
    assert np.isclose(matrix[1, col_to_idx["meta_has_year"]], 1.0)
    assert np.isclose(matrix[2, col_to_idx["vision_num_labels"]], 1.0)


def test_build_combined_embeddings_payload_pca_and_l2_normalization():
    payload = build_combined_embeddings_payload(
        object_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        text_docs=["ancient stone", "portrait oil", "egypt statue", "roman bronze"],
        vision_docs=["face statue", "canvas brush", "artifact hieroglyph", "helmet metal"],
        numeric_rows=[
            {"objectID": 1, "meta_has_year": 1.0, "vision_num_labels": 2.0},
            {"objectID": 2, "meta_has_year": 1.0, "vision_num_labels": 1.0},
            {"objectID": 3, "meta_has_year": 0.0, "vision_num_labels": 3.0},
            {"objectID": 4, "meta_has_year": 1.0, "vision_num_labels": 4.0},
        ],
        pca_variance=0.95,
        pca_max_components=3,
    )
    emb = payload["embeddings"]
    norms = np.linalg.norm(emb, axis=1)

    assert emb.shape[0] == 4
    assert emb.shape[1] <= 3
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)
    assert payload["metrics"]["explained_variance_ratio_sum"] >= 0.0


def test_build_combined_embeddings_payload_with_missing_vision_docs():
    payload = build_combined_embeddings_payload(
        object_ids=np.array([10, 11, 12], dtype=np.int64),
        text_docs=["head sculpture", "portrait", "vase clay"],
        vision_docs=["", " ", ""],
        numeric_rows=[
            {"objectID": 10, "meta_has_year": 1.0},
            {"objectID": 11, "meta_has_year": 0.0},
            {"objectID": 12, "meta_has_year": 1.0},
        ],
    )
    emb = payload["embeddings"]
    assert emb.shape[0] == 3
    assert np.isfinite(emb).all()
    assert payload["vision_vectorizer"] is None
    assert payload["metrics"]["vision_feature_dim"] == 0


from typing import Any, cast
import numpy as np
from numpy.typing import NDArray

def test_combined_embeddings_payload_is_deterministic_shape_and_components():
    object_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    text_docs = ["aa bb", "aa cc", "dd ee", "ff gg", "hh ii"]
    vision_docs = ["xx yy", "yy zz", "xx", "zz", "xx zz"]
    numeric_rows = [
        {"objectID": 1, "meta_has_year": 1.0},
        {"objectID": 2, "meta_has_year": 0.0},
        {"objectID": 3, "meta_has_year": 1.0},
        {"objectID": 4, "meta_has_year": 0.0},
        {"objectID": 5, "meta_has_year": 1.0},
    ]

    one = build_combined_embeddings_payload(
        object_ids=object_ids,
        text_docs=text_docs,
        vision_docs=vision_docs,
        numeric_rows=numeric_rows,
        pca_variance=0.95,
        pca_max_components=4,
    )
    two = build_combined_embeddings_payload(
        object_ids=object_ids,
        text_docs=text_docs,
        vision_docs=vision_docs,
        numeric_rows=numeric_rows,
        pca_variance=0.95,
        pca_max_components=4,
    )

    one_embeddings = cast(NDArray[np.floating], one["embeddings"])
    two_embeddings = cast(NDArray[np.floating], two["embeddings"])
    one_metrics = cast(dict[str, Any], one["metrics"])
    two_metrics = cast(dict[str, Any], two["metrics"])

    assert one_embeddings.shape == two_embeddings.shape
    assert one_metrics["selected_components"] == two_metrics["selected_components"]

def test_atomic_pickle_dump_writes_expected_schema(tmp_path):
    payload = build_combined_embeddings_payload(
        object_ids=np.array([1, 2], dtype=np.int64),
        text_docs=["ancient", "portrait"],
        vision_docs=["stone", "canvas"],
        numeric_rows=[
            {"objectID": 1, "meta_has_year": 1.0},
            {"objectID": 2, "meta_has_year": 0.0},
        ],
    )
    out_path = tmp_path / "combined_embeddings.pkl"
    atomic_pickle_dump(payload, out_path)
    with open(out_path, "rb") as file:
        loaded = pickle.load(file)

    expected = {
        "object_ids",
        "embeddings",
        "pca_model",
        "numeric_scaler",
        "text_vectorizer",
        "vision_vectorizer",
        "numeric_feature_columns",
        "config",
        "metrics",
    }
    assert expected.issubset(set(loaded.keys()))


def test_build_combined_embeddings_payload_rejects_invalid_pca_variance():
    with pytest.raises(ValueError, match="pca_variance"):
        build_combined_embeddings_payload(
            object_ids=np.array([1, 2], dtype=np.int64),
            text_docs=["a", "b"],
            vision_docs=["a", "b"],
            numeric_rows=[],
            pca_variance=1.2,
        )


def test_build_combined_embeddings_payload_rejects_negative_weight():
    with pytest.raises(ValueError, match="text_weight"):
        build_combined_embeddings_payload(
            object_ids=np.array([1, 2], dtype=np.int64),
            text_docs=["a", "b"],
            vision_docs=["a", "b"],
            numeric_rows=[],
            text_weight=-0.1,
        )
