from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import scripts.compare_clip_modes as compare_clip_modes
import scripts.evaluate_backends as evaluate_backends
import scripts.tune_clip_retrieval as tune_clip_retrieval


class FakeClipRecommender:
    def __init__(self, backend: str = "clip") -> None:
        self.embedding_backend = backend
        self.clip_similarity_weight = 0.8
        self.clip_lexical_weight = 0.2
        self.clip_prompt_ensemble = True
        self.metadata = pd.DataFrame(
            {
                "objectID": [1, 2, 3, 4],
                "title": ["Portrait of a Woman", "Saber", "Saint Study", "Egyptian Relief"],
                "artist": ["A", "B", "C", "D"],
                "department": ["A", "A", "B", "B"],
                "medium": ["Oil", "Steel", "Ink", "Stone"],
            }
        )

    def recommend_for_theme(self, theme: str, n_recommendations: int = 8, **_kwargs) -> pd.DataFrame:
        lower = theme.lower()
        if lower in {"a", "portrait", "portraits"}:
            rows = [
                {"object_id": 1, "title": "Portrait of a Woman", "artist": "A", "department": "A", "medium": "Oil", "score": 0.9},
                {"object_id": 2, "title": "Saber", "artist": "B", "department": "A", "medium": "Steel", "score": 0.6},
            ]
        elif "christian" in lower or lower in {"b", "religious art"}:
            rows = [
                {"object_id": 3, "title": "Saint Study", "artist": "C", "department": "B", "medium": "Ink", "score": 0.85},
                {"object_id": 4, "title": "Egyptian Relief", "artist": "D", "department": "B", "medium": "Stone", "score": 0.55},
            ]
        else:
            rows = [
                {"object_id": 4, "title": "Egyptian Relief", "artist": "D", "department": "B", "medium": "Stone", "score": 0.8},
                {"object_id": 1, "title": "Portrait of a Woman", "artist": "A", "department": "A", "medium": "Oil", "score": 0.5},
            ]
        return pd.DataFrame(rows[:n_recommendations])


def test_compare_clip_modes_main_prints_report(monkeypatch, capsys, tmp_path):
    del tmp_path
    monkeypatch.setattr(
        compare_clip_modes.ExhibitionRecommender,
        "from_artifacts",
        staticmethod(lambda _p: FakeClipRecommender("clip")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_clip_modes.py",
            "--artifacts",
            "dummy",
            "--themes",
            "portrait,christian",
            "--top-k",
            "2",
        ],
    )
    compare_clip_modes.main()
    out = capsys.readouterr().out
    assert "hit_rate_base" in out
    assert "Mean keyword hit rate" in out


def test_tune_clip_retrieval_main_prints_table(monkeypatch, capsys):
    monkeypatch.setattr(
        tune_clip_retrieval.ExhibitionRecommender,
        "from_artifacts",
        staticmethod(lambda _p: FakeClipRecommender("clip")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_clip_retrieval.py",
            "--artifacts",
            "dummy",
            "--themes",
            "portrait,christian",
            "--clip-weights",
            "0.5,0.9",
            "--top-k",
            "2",
            "--include-no-prompt-ensemble",
        ],
    )
    tune_clip_retrieval.main()
    out = capsys.readouterr().out
    assert "clip_weight" in out
    assert "keyword_hit_rate" in out


def test_evaluate_backends_main_writes_json(monkeypatch, tmp_path: Path):
    def fake_from_artifacts(_p: str) -> FakeClipRecommender:
        return FakeClipRecommender("clip")

    monkeypatch.setattr(
        evaluate_backends.ExhibitionRecommender,
        "from_artifacts",
        staticmethod(fake_from_artifacts),
    )
    json_out = tmp_path / "eval.json"
    # backend metadata file is optional; create one path-backed directory for completeness.
    artifacts = tmp_path / "artifacts_eval"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "embedding_backend.json").write_text(
        json.dumps({"backend": "clip"}, ensure_ascii=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_backends.py",
            "--artifacts",
            str(artifacts),
            "--k",
            "2",
            "--json-out",
            str(json_out),
        ],
    )
    evaluate_backends.main()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload[0]["backend"] in {"clip", "tfidf"}
