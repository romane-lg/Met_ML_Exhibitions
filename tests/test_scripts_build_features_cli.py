from __future__ import annotations

import sys

import scripts.build_features as build_features_cli


def test_build_features_main_passes_expected_kwargs(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_build(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(build_features_cli, "run_build", fake_run_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_features.py",
            "--limit",
            "12",
            "--force",
            "--offline",
            "--embedding-backend",
            "clip",
            "--clip-model-name",
            "ViT-B-32",
            "--clip-pretrained",
            "laion2b_s34b_b79k",
            "--clip-device",
            "cpu",
            "--clip-batch-size",
            "16",
            "--clip-text-weight",
            "0.7",
            "--clip-image-weight",
            "0.3",
            "--clip-retrieval-weight",
            "0.9",
            "--clip-lexical-weight",
            "0.1",
            "--no-clip-prompt-ensemble",
        ],
    )

    build_features_cli.main()

    assert captured["limit"] == 12
    assert captured["force"] is True
    assert captured["offline"] is True
    assert captured["embedding_backend"] == "clip"
    assert captured["clip_batch_size"] == 16
    assert captured["clip_text_weight"] == 0.7
    assert captured["clip_image_weight"] == 0.3
    assert captured["clip_retrieval_weight"] == 0.9
    assert captured["clip_lexical_weight"] == 0.1
    assert captured["clip_prompt_ensemble"] is False
