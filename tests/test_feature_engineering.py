from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer, create_feature_pipeline


def test_combine_features_shape() -> None:
    engineer = FeatureEngineer(use_pca=False)
    vision = np.array([[1.0, 2.0], [2.0, 3.0]])
    text = np.array([[3.0, 4.0], [4.0, 5.0]])
    out = engineer.combine_features(vision, text, normalize=False)
    assert out.shape == (2, 4)


def test_combine_features_mismatch_raises() -> None:
    engineer = FeatureEngineer(use_pca=False)
    with pytest.raises(ValueError, match="Number of samples must match"):
        engineer.combine_features(np.zeros((2, 2)), np.zeros((3, 2)))


def test_fit_transform_with_pca() -> None:
    engineer = FeatureEngineer(use_pca=True, n_components=2)
    vision = np.random.RandomState(0).rand(6, 3)
    text = np.random.RandomState(1).rand(6, 4)
    out = engineer.fit_transform(vision, text)
    assert out.shape == (6, 2)


def test_transform_without_fitted_pca_raises() -> None:
    engineer = FeatureEngineer(use_pca=True, n_components=2)
    with pytest.raises(ValueError, match="PCA not fitted yet"):
        engineer.transform(np.zeros((4, 2)), np.zeros((4, 2)))


def test_save_load(tmp_path: Path) -> None:
    path = tmp_path / "engineer.pkl"
    engineer = FeatureEngineer(use_pca=False, scaler_type="minmax")
    engineer.save(str(path))
    loaded = FeatureEngineer.load(str(path))
    assert isinstance(loaded, FeatureEngineer)
    assert loaded.scaler_type == "minmax"


def test_create_feature_pipeline_defaults() -> None:
    df = pd.DataFrame({"objectID": [1, 2, 3]})
    features, engineer = create_feature_pipeline(df, config={"use_pca": False})
    assert features.shape == (3, 20)
    assert isinstance(engineer, FeatureEngineer)


def test_create_feature_pipeline_from_pickles(tmp_path: Path) -> None:
    df = pd.DataFrame({"objectID": [1, 2, 3]})
    vision = np.ones((3, 5))
    text = {"tfidf": np.ones((3, 4)), "topics": np.ones((3, 2))}
    vision_path = tmp_path / "vision.pkl"
    text_path = tmp_path / "text.pkl"
    save_path = tmp_path / "combined.pkl"
    with open(vision_path, "wb") as file:
        pickle.dump(vision, file)
    with open(text_path, "wb") as file:
        pickle.dump(text, file)

    features, _ = create_feature_pipeline(
        df,
        vision_features_path=str(vision_path),
        text_features_path=str(text_path),
        save_path=str(save_path),
        config={"use_pca": False},
    )
    assert features.shape == (3, 11)
    assert save_path.exists()
