from __future__ import annotations

import sys
from typing import cast

import numpy as np
import pandas as pd

import scripts.evaluate_model_comprehensive as eval_comp
from src.models.recommender import ExhibitionRecommender


class FakeComprehensiveRecommender:
    def __init__(self, backend: str = "clip") -> None:
        self.embedding_backend = backend
        self.metadata = pd.DataFrame(
            {
                "objectID": [1, 2, 3, 4],
                "artist": ["A", "B", "C", "D"],
                "department": ["DeptA", "DeptA", "DeptB", "DeptB"],
                "title": ["Portrait", "Armor", "Saint", "Relief"],
                "medium": ["Oil", "Steel", "Ink", "Stone"],
            }
        )
        self.embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        self.id_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}

    def recommend_for_theme(self, theme: str, n_recommendations: int = 2, **_kwargs) -> pd.DataFrame:
        lower = theme.lower()
        if "depta" in lower or "portrait" in lower:
            rows = [
                {"object_id": 1, "artist": "A", "department": "DeptA", "score": 0.9},
                {"object_id": 2, "artist": "B", "department": "DeptA", "score": 0.7},
            ]
        elif "deptb" in lower or "christian" in lower or "religious" in lower:
            rows = [
                {"object_id": 3, "artist": "C", "department": "DeptB", "score": 0.85},
                {"object_id": 4, "artist": "D", "department": "DeptB", "score": 0.65},
            ]
        else:
            rows = [
                {"object_id": 1, "artist": "A", "department": "DeptA", "score": 0.8},
                {"object_id": 3, "artist": "C", "department": "DeptB", "score": 0.6},
            ]
        return pd.DataFrame(rows[:n_recommendations])

    def recommend_exhibitions(
        self,
        themes: list[str],
        max_pieces_per_exhibition: int = 2,
        **_kwargs,
    ) -> dict[str, pd.DataFrame]:
        used: set[int] = set()
        out: dict[str, pd.DataFrame] = {}
        for theme in themes:
            frame = self.recommend_for_theme(theme, n_recommendations=max_pieces_per_exhibition)
            if not frame.empty:
                frame = frame[~frame["object_id"].isin(used)].copy()
                used.update(int(v) for v in frame["object_id"].tolist())
            out[theme] = frame
        return out


def test_exhibition_coverage_metrics_has_zero_overlap():
    rec = FakeComprehensiveRecommender()
    coverage, overlap = eval_comp._exhibition_coverage_metrics(
        cast(ExhibitionRecommender, rec),
        ["DeptA", "DeptB"],
        top_k=2,
    )
    assert coverage > 0.0
    assert overlap == 0.0


def test_evaluate_returns_new_coverage_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(
        eval_comp.ExhibitionRecommender,
        "from_artifacts",
        staticmethod(lambda _p: FakeComprehensiveRecommender("clip")),
    )
    rep = eval_comp.evaluate(
        artifacts_dir=str(tmp_path),
        top_k=2,
        min_department_size=2,
        latency_runs=4,
    )
    assert rep.backend in {"clip", "tfidf"}
    assert 0.0 <= rep.independent_unique_id_coverage <= 1.0
    assert 0.0 <= rep.exhibition_unique_id_coverage <= 1.0
    assert 0.0 <= rep.exhibition_overlap_rate <= 1.0
    payload = eval_comp._to_dict(rep)
    assert "independent_unique_id_coverage" in payload
    assert "exhibition_unique_id_coverage" in payload
    assert "exhibition_overlap_rate" in payload


def test_main_prints_comprehensive_columns(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        eval_comp.ExhibitionRecommender,
        "from_artifacts",
        staticmethod(lambda _p: FakeComprehensiveRecommender("clip")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_model_comprehensive.py",
            "--artifacts",
            str(tmp_path),
            "--top-k",
            "2",
            "--latency-runs",
            "3",
        ],
    )
    eval_comp.main()
    out = capsys.readouterr().out
    assert "id_cov_ind" in out
    assert "id_cov_exh" in out
    assert "exh_overlap" in out
