import numpy as np
import pandas as pd

from src.models import ExhibitionRecommender


def make_recommender() -> ExhibitionRecommender:
    meta = pd.DataFrame(
        {
            "objectID": [1, 2, 3, 4],
            "title": ["Egypt Head", "Egypt Vase", "Portrait A", "Portrait B"],
            "artist": ["x", "y", "z", "w"],
            "department": ["Egyptian Art", "Egyptian Art", "Paintings", "Paintings"],
            "objectDate": ["100", "120", "1800", "1810"],
            "medium": ["stone", "clay", "oil", "oil"],
            "image_path": ["images/1.jpg", "images/2.jpg", "images/3.jpg", "images/4.jpg"],
        }
    )

    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = ["egypt head", "egypt vase", "portrait painting", "portrait drawing"]
    vec = TfidfVectorizer().fit(docs)
    embeddings = vec.transform(docs).toarray().astype(np.float32)
    return ExhibitionRecommender(embeddings, meta, vec)


def test_recommend_for_theme_returns_scores():
    rec = make_recommender()
    out = rec.recommend_for_theme("egypt", n_recommendations=2)
    assert len(out) == 2
    assert "score" in out.columns


def test_recommend_exhibitions_splits_themes():
    rec = make_recommender()
    out = rec.recommend_exhibitions(["egypt", "portrait"], max_pieces_per_exhibition=2)
    assert set(out.keys()) == {"egypt", "portrait"}


def test_coherence_range():
    rec = make_recommender()
    score = rec.evaluate_coherence([1, 2])
    assert 0.0 <= score <= 1.0
