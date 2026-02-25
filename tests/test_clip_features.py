from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.clip_features import CLIPEncoder, l2_normalize


def test_l2_normalize_rows():
    arr = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(arr)
    assert np.allclose(out[0], np.array([0.6, 0.8], dtype=np.float32))
    assert np.allclose(out[1], np.array([0.0, 0.0], dtype=np.float32))


def test_clip_encoder_missing_dependency_raises(monkeypatch):
    encoder = CLIPEncoder()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in {"open_clip", "torch"}:
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    try:
        try:
            encoder._ensure_loaded()
        except RuntimeError as exc:
            assert "OpenCLIP dependencies are missing" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("Expected RuntimeError for missing OpenCLIP dependencies")
    finally:
        monkeypatch.setattr("builtins.__import__", real_import)


def test_encode_images_missing_files_returns_zero_vectors(monkeypatch, tmp_path):
    encoder = CLIPEncoder()

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self.arr = arr

        def to(self, _device: str):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.arr

    class FakeTorch:
        @staticmethod
        def no_grad():
            return FakeNoGrad()

        @staticmethod
        def stack(items):
            data = np.stack(items, axis=0).astype(np.float32)
            return FakeTensor(data)

    class FakeModel:
        @staticmethod
        def encode_image(batch):
            values = batch.numpy()
            return FakeTensor(values)

    def fake_ensure_loaded():
        encoder._torch = FakeTorch()
        encoder._model = FakeModel()
        encoder._preprocess = lambda _img: np.array([1.0, 0.0, 0.0], dtype=np.float32)
        encoder._tokenizer = lambda texts: texts
        encoder._embed_dim = 3

    monkeypatch.setattr(encoder, "_ensure_loaded", fake_ensure_loaded)

    missing_path = Path(tmp_path / "missing.jpg")
    out, errors = encoder.encode_images([missing_path])
    assert out.shape == (1, 3)
    assert np.allclose(out, 0.0)
    assert errors
