import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models import ExhibitionRecommender


class DummyRanker:
    pass


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

    docs = ["egypt head", "egypt vase", "portrait painting", "portrait drawing"]
    vec = TfidfVectorizer().fit(docs)
    embeddings = vec.transform(docs).toarray().astype(np.float32)
    numeric_features = np.array(
        [
            [1.0, 100.0, 50.0, 100.0, 150.0],
            [1.0, 120.0, 70.0, 90.0, 140.0],
            [1.0, 1800.0, 10.0, 20.0, 30.0],
            [1.0, 1810.0, 15.0, 25.0, 35.0],
        ],
        dtype=np.float32,
    )
    numeric_cols = [
        "meta_has_year",
        "meta_year_mean",
        "vision_avg_red",
        "vision_avg_green",
        "vision_avg_blue",
    ]
    return ExhibitionRecommender(
        embeddings,
        meta,
        vec,
        numeric_features=numeric_features,
        numeric_columns=numeric_cols,
    )


def test_recommend_for_theme_returns_scores():
    rec = make_recommender()
    out = rec.recommend_for_theme("egypt", n_recommendations=2)
    assert len(out) == 2
    assert "score" in out.columns
    assert out["score"].between(0.0, 1.0).all()


def test_recommend_exhibitions_splits_themes():
    rec = make_recommender()
    out = rec.recommend_exhibitions(
        ["egypt", "portrait", "vase"], max_pieces_per_exhibition=2, min_similarity=0.0
    )
    assert set(out.keys()) == {"egypt", "portrait", "vase"}
    assert len(out["egypt"]) > 0
    assert len(out["portrait"]) > 0


def test_coherence_range():
    rec = make_recommender()
    score = rec.evaluate_coherence([1, 2])
    assert 0.0 <= score <= 1.0


def test_recommend_for_theme_respects_exclusions():
    rec = make_recommender()
    out = rec.recommend_for_theme("egypt", n_recommendations=3, exclude_ids=[1, 2])
    assert not set(out["object_id"]).intersection({1, 2})


def test_score_by_tokens_empty_returns_zeros():
    rec = make_recommender()
    out = rec.score_by_tokens([])
    assert np.allclose(out, 0.0)
    assert out.shape == (4,)


def test_cosine_similarity_matching_prefers_theme_aligned_items():
    rec = make_recommender()
    scores = rec.score_by_tokens(["egypt"])
    top_two = np.argsort(scores)[::-1][:2]
    top_ids = {int(rec.metadata.iloc[idx]["objectID"]) for idx in top_two}
    assert top_ids == {1, 2}


def test_query_numeric_features_maps_year_and_color():
    rec = make_recommender()
    vec = rec._query_numeric_features("gold artifacts from 1900 1910")
    col_to_idx = {name: idx for idx, name in enumerate(rec.numeric_columns)}

    assert vec[col_to_idx["meta_has_year"]] == 1.0
    assert vec[col_to_idx["meta_year_mean"]] == 1905.0
    assert vec[col_to_idx["vision_avg_red"]] > 0.0


def test_rerank_falls_back_to_base_scores_on_ranker_error():
    class BrokenRanker:
        def predict(self, _):
            raise RuntimeError("boom")

    rec = make_recommender()
    rec.ranker = BrokenRanker()  # type: ignore[assignment]
    qarr = rec._query_vector(["egypt"])
    base_scores = rec.score_by_tokens(["egypt"])
    candidate_indices = np.array([0, 1, 2], dtype=int)

    reranked = rec._rerank_scores("egypt", qarr, base_scores, candidate_indices)
    assert np.allclose(reranked, base_scores[candidate_indices].astype(np.float32))


def test_recommend_for_theme_applies_diversity_constraints():
    meta = pd.DataFrame(
        {
            "objectID": [10, 11, 12, 13],
            "title": ["Egypt A", "Egypt B", "Egypt C", "Egypt D"],
            "artist": ["same_artist", "same_artist", "other_artist", "third_artist"],
            "department": ["Egyptian Art", "Egyptian Art", "Egyptian Art", "Sculpture"],
            "objectDate": ["100", "110", "120", "130"],
            "medium": ["stone", "stone", "stone", "stone"],
            "image_path": ["images/10.jpg", "images/11.jpg", "images/12.jpg", "images/13.jpg"],
        }
    )
    docs = ["egypt statue", "egypt relief", "egypt artifact", "egypt sculpture"]
    vec = TfidfVectorizer().fit(docs)
    embeddings = vec.transform(docs).toarray().astype(np.float32)
    rec = ExhibitionRecommender(embeddings, meta, vec)

    out = rec.recommend_for_theme(
        "egypt",
        n_recommendations=3,
        min_score=0.0,
        max_per_artist=1,
        max_per_department=2,
        diversity_lambda=0.7,
    )

    returned_artists = out["artist"].astype(str).str.lower().tolist()
    assert returned_artists.count("same_artist") <= 1
    returned_departments = out["department"].astype(str).str.lower().tolist()
    assert returned_departments.count("egyptian art") <= 2


def test_from_artifacts_loads_numeric_and_ranker(tmp_path):
    meta = pd.DataFrame(
        {
            "objectID": [10, 11],
            "title": ["A", "B"],
            "artist": ["x", "y"],
            "department": ["d1", "d2"],
            "objectDate": ["1900", "1901"],
            "medium": ["m1", "m2"],
            "image_path": ["images/10.jpg", "images/11.jpg"],
        }
    )
    docs = ["ancient art", "modern art"]
    vec = TfidfVectorizer().fit(docs)
    emb = vec.transform(docs).toarray().astype(np.float32)

    np.savez_compressed(tmp_path / "embeddings.npz", embeddings=emb)
    meta.to_csv(tmp_path / "meta.csv", index=False)
    joblib.dump(vec, tmp_path / "text_vectorizer.joblib")
    pd.DataFrame(
        {
            "objectID": [10, 11],
            "meta_has_year": [1.0, 1.0],
            "meta_year_mean": [1900.0, 1901.0],
        }
    ).to_csv(tmp_path / "numeric_features.csv", index=False)

    joblib.dump(DummyRanker(), tmp_path / "lightgbm_ranker.joblib")

    rec = ExhibitionRecommender.from_artifacts(str(tmp_path))
    assert rec.numeric_features.shape == (2, 2)
    assert rec.ranker is not None
