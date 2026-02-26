from __future__ import annotations

import json
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.bootstrap import ensure_artifacts
from src.config import Settings


class StubSettings:
    def __init__(
        self,
        artifacts_dir: str,
        *,
        auto_build_on_startup: bool,
        embedding_backend: str,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.auto_build_on_startup = auto_build_on_startup
        self.embedding_backend = embedding_backend
        self.clip_model_name = "ViT-B-32"
        self.clip_pretrained = "laion2b_s34b_b79k"
        self.clip_device = "cpu"
        self.clip_batch_size = 32
        self.clip_text_weight = 0.5
        self.clip_image_weight = 0.5
        self.clip_retrieval_weight = 0.8
        self.clip_lexical_weight = 0.2
        self.clip_prompt_ensemble = True


def test_bootstrap_inference_only_default_does_not_autobuild(tmp_path, monkeypatch):
    called = {"value": False}

    def fake_run_build(**_kwargs):
        called["value"] = True

    monkeypatch.setattr("src.bootstrap.run_build", fake_run_build)

    settings = StubSettings(
        artifacts_dir=str(tmp_path),
        auto_build_on_startup=False,
        embedding_backend="tfidf",
    )
    status = ensure_artifacts(cast(Settings, settings))
    assert status.ready is False
    assert called["value"] is False
    assert status.error is not None
    assert "make build-features" in status.error


def test_bootstrap_autobuild_uses_selected_backend(tmp_path, monkeypatch):
    def fake_run_build(**kwargs):
        assert kwargs["embedding_backend"] == "clip"
        artifacts = tmp_path
        np.savez_compressed(artifacts / "embeddings.npz", embeddings=np.zeros((1, 2), dtype=np.float32))
        pd.DataFrame({"objectID": [1]}).to_csv(artifacts / "meta.csv", index=False)
        (artifacts / "tokens.json").write_text(json.dumps({"1": {"text": [], "image": []}}), encoding="utf-8")
        (artifacts / "embedding_backend.json").write_text(
            json.dumps({"backend": "clip"}, ensure_ascii=True),
            encoding="utf-8",
        )
        joblib.dump(
            {
                "model_name": "ViT-B-32",
                "pretrained": "laion2b_s34b_b79k",
                "device": "cpu",
                "batch_size": 32,
            },
            artifacts / "clip_metadata.joblib",
        )

    monkeypatch.setattr("src.bootstrap.run_build", fake_run_build)

    settings = StubSettings(
        artifacts_dir=str(tmp_path),
        auto_build_on_startup=True,
        embedding_backend="clip",
    )
    status = ensure_artifacts(cast(Settings, settings))
    assert status.ready is True
    assert status.built is True
    assert status.warning is not None
    assert "similarity-only" in status.warning


def test_bootstrap_backward_compatible_without_backend_metadata(tmp_path):
    np.savez_compressed(tmp_path / "embeddings.npz", embeddings=np.zeros((1, 2), dtype=np.float32))
    pd.DataFrame({"objectID": [1]}).to_csv(tmp_path / "meta.csv", index=False)
    (tmp_path / "tokens.json").write_text(json.dumps({"1": {"text": [], "image": []}}), encoding="utf-8")
    vec = TfidfVectorizer().fit(["art"])
    joblib.dump(vec, tmp_path / "text_vectorizer.joblib")

    settings = StubSettings(
        artifacts_dir=str(tmp_path),
        auto_build_on_startup=False,
        embedding_backend="tfidf",
    )
    status = ensure_artifacts(cast(Settings, settings))
    assert status.ready is True
    assert status.warning is not None
    assert "xgboost_ranker.json" in status.warning


def test_bootstrap_ready_without_ranker_emits_warning(tmp_path):
    np.savez_compressed(tmp_path / "embeddings.npz", embeddings=np.zeros((1, 2), dtype=np.float32))
    pd.DataFrame({"objectID": [1]}).to_csv(tmp_path / "meta.csv", index=False)
    (tmp_path / "tokens.json").write_text(json.dumps({"1": {"text": [], "image": []}}), encoding="utf-8")
    (tmp_path / "embedding_backend.json").write_text(json.dumps({"backend": "clip"}), encoding="utf-8")
    joblib.dump(
        {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "device": "cpu",
            "batch_size": 32,
        },
        tmp_path / "clip_metadata.joblib",
    )

    settings = StubSettings(
        artifacts_dir=str(tmp_path),
        auto_build_on_startup=False,
        embedding_backend="clip",
    )
    status = ensure_artifacts(cast(Settings, settings))
    assert status.ready is True
    assert status.error is None
    assert status.warning is not None
    assert "make train-ranker" in status.warning
