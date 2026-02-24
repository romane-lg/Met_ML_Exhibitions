from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.text_features import TextFeatureExtractor, extract_all_text_features


def test_preprocess_text() -> None:
    extractor = TextFeatureExtractor(min_df=1, max_df=1.0)
    extractor.stop_words = {"the"}
    out = extractor.preprocess_text("The Ancient! Egypt 123")
    assert out == "ancient egypt"


def test_combine_text_fields() -> None:
    df = pd.DataFrame(
        {
            "title": ["Head"],
            "artist": ["Unknown"],
            "medium": ["Stone"],
            "department": ["Egyptian Art"],
            "objectDate": ["100 BCE"],
        }
    )
    extractor = TextFeatureExtractor(min_df=1, max_df=1.0)
    combined = extractor.combine_text_fields(df)
    assert "Head" in combined.iloc[0]
    assert "Egyptian Art" in combined.iloc[0]


def test_tfidf_fit_then_transform() -> None:
    extractor = TextFeatureExtractor(min_df=1, max_df=1.0, ngram_range=(1, 1))
    extractor.stop_words = set()

    texts = pd.Series(["ancient egypt", "portrait oil", "egypt stone"])
    fit_features = extractor.extract_tfidf_features(texts, fit=True)
    trans_features = extractor.extract_tfidf_features(texts, fit=False)
    assert fit_features.shape[0] == 3
    assert trans_features.shape == fit_features.shape


def test_tfidf_transform_without_fit_raises() -> None:
    extractor = TextFeatureExtractor(min_df=1, max_df=1.0)
    with pytest.raises(ValueError, match="Vectorizer not fitted"):
        extractor.extract_tfidf_features(pd.Series(["x"]), fit=False)


def test_topic_features_and_top_words() -> None:
    extractor = TextFeatureExtractor(min_df=1, max_df=1.0, ngram_range=(1, 1), max_features=50)
    extractor.stop_words = set()
    texts = pd.Series(["ancient egypt stone", "portrait oil canvas", "egypt sculpture"])
    tfidf = extractor.extract_tfidf_features(texts, fit=True)
    topics = extractor.extract_topic_features(tfidf, n_topics=2, fit=True)
    assert topics.shape == (3, 2)
    words = extractor.get_top_words_per_topic(n_words=3)
    assert set(words.keys()) == {0, 1}


def test_extract_metadata_features() -> None:
    df = pd.DataFrame(
        {
            "department": ["Egyptian Art", "Paintings"],
            "objectDate": ["100 BCE", None],
            "artist": ["Unknown", None],
        }
    )
    extractor = TextFeatureExtractor(min_df=1, max_df=1.0)
    out = extractor.extract_metadata_features(df)
    assert "has_date" in out.columns
    assert "has_artist" in out.columns


def test_extract_all_text_features_and_save(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "title": ["Head", "Portrait", "Vase"],
            "artist": ["Unknown", "A", "B"],
            "medium": ["Stone", "Oil", "Clay"],
            "department": ["Egyptian Art", "Paintings", "Greek Art"],
            "objectDate": ["100", "1800", "200"],
        }
    )
    out_path = tmp_path / "text_features.pkl"
    features = extract_all_text_features(df, n_topics=2, save_path=str(out_path))
    assert features["tfidf"].shape[0] == 3
    assert features["topics"].shape == (3, 2)
    assert isinstance(features["metadata"], np.ndarray)
    assert out_path.exists()
