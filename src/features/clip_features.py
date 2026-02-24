from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    values = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (values / norms).astype(np.float32, copy=False)


class CLIPEncoder:
    """Lazy OpenCLIP wrapper for batched text/image embedding."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.batch_size = max(1, int(batch_size))

        self._torch: Any | None = None
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None
        self._embed_dim: int | None = None

    @property
    def embedding_dimension(self) -> int:
        self._ensure_loaded()
        assert self._embed_dim is not None
        return self._embed_dim

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            assert self._embed_dim is not None
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        assert self._torch is not None
        assert self._model is not None
        assert self._tokenizer is not None
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            with self._torch.no_grad():
                tokens = self._tokenizer(batch).to(self.device)
                feats = self._model.encode_text(tokens)
            vectors.append(feats.detach().cpu().numpy().astype(np.float32, copy=False))
        return l2_normalize(np.vstack(vectors))

    def encode_images(self, image_paths: list[Path | None]) -> tuple[np.ndarray, list[str]]:
        self._ensure_loaded()
        if not image_paths:
            assert self._embed_dim is not None
            return np.zeros((0, self._embed_dim), dtype=np.float32), []

        assert self._torch is not None
        assert self._model is not None
        assert self._preprocess is not None
        assert self._embed_dim is not None

        outputs = np.zeros((len(image_paths), self._embed_dim), dtype=np.float32)
        errors: list[str] = []

        valid_indices: list[int] = []
        tensors: list[Any] = []
        for idx, path in enumerate(image_paths):
            if path is None:
                errors.append(f"{idx}:missing_image_file")
                continue
            if not path.exists() or not path.is_file():
                errors.append(f"{idx}:missing_image_file")
                continue
            try:
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                tensors.append(self._preprocess(rgb))
                valid_indices.append(idx)
            except (UnidentifiedImageError, OSError, ValueError):
                errors.append(f"{idx}:unsupported_or_corrupt_image")

        for start in range(0, len(valid_indices), self.batch_size):
            batch_indices = valid_indices[start : start + self.batch_size]
            batch_tensors = tensors[start : start + self.batch_size]
            if not batch_tensors:
                continue
            stacked = self._torch.stack(batch_tensors).to(self.device)
            with self._torch.no_grad():
                feats = self._model.encode_image(stacked)
            batch_vectors = feats.detach().cpu().numpy().astype(np.float32, copy=False)
            batch_vectors = l2_normalize(batch_vectors)
            for offset, row_idx in enumerate(batch_indices):
                outputs[row_idx] = batch_vectors[offset]
        return outputs, errors

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip  # type: ignore[import-not-found]
            import torch
        except Exception as exc:  # pragma: no cover - dependency/runtime specific
            raise RuntimeError(
                "OpenCLIP dependencies are missing. Install `open_clip_torch` and `torch`."
            ) from exc

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)
        model.eval()

        self._torch = torch
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._embed_dim = int(getattr(model, "text_projection").shape[1])
