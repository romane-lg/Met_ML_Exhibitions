from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from src.api.main import create_app
from src.models import ExhibitionRecommender


def make_recommender() -> ExhibitionRecommender:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    meta = pd.DataFrame(
        {
            "objectID": [1, 2],
            "title": ["Egypt", "Portrait"],
            "artist": ["a", "b"],
            "department": ["Egyptian Art", "Paintings"],
            "objectDate": ["100", "1800"],
            "medium": ["stone", "oil"],
            "image_path": ["images/1.jpg", "images/2.jpg"],
        }
    )
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer().fit(["egypt", "portrait"])
    return ExhibitionRecommender(embeddings, meta, vec)


def test_health_endpoint():
    app = create_app()
    app.state.recommender = make_recommender()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_theme_endpoint():
    app = create_app()
    app.state.recommender = make_recommender()
    client = TestClient(app)
    r = client.post("/recommendations/theme", json={"theme": "egypt", "k": 1, "min_similarity": 0.0})
    assert r.status_code == 200
    assert "results" in r.json()
