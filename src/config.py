from __future__ import annotations

import functools
from pathlib import Path
from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_clip_device() -> str:
    """Return 'mps' on Apple Silicon when PyTorch MPS is available, else 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Settings(BaseSettings):
    data_csv: str = Field(default="data/raw/met_data.csv", validation_alias="MET_DATA_CSV")
    images_dir: str = Field(default="data/raw/images", validation_alias="MET_IMAGES_DIR")
    artifacts_dir: str = Field(default="artifacts", validation_alias="MET_ARTIFACTS_DIR")
    auto_build_on_startup: bool = Field(
        default=False,
        validation_alias="MET_AUTO_BUILD_ON_STARTUP",
    )
    embedding_backend: str = Field(default="clip", validation_alias="MET_EMBEDDING_BACKEND")
    clip_model_name: str = Field(default="ViT-B-32", validation_alias="MET_CLIP_MODEL_NAME")
    clip_pretrained: str = Field(
        default="laion2b_s34b_b79k",
        validation_alias="MET_CLIP_PRETRAINED",
    )
    clip_device: str = Field(default_factory=_default_clip_device, validation_alias="MET_CLIP_DEVICE")
    clip_batch_size: int = Field(default=32, validation_alias="MET_CLIP_BATCH_SIZE")
    clip_text_weight: float = Field(default=0.5, validation_alias="MET_CLIP_TEXT_WEIGHT")
    clip_image_weight: float = Field(default=0.5, validation_alias="MET_CLIP_IMAGE_WEIGHT")
    clip_retrieval_weight: float = Field(default=0.8, validation_alias="MET_CLIP_RETRIEVAL_WEIGHT")
    clip_lexical_weight: float = Field(default=0.2, validation_alias="MET_CLIP_LEXICAL_WEIGHT")
    clip_prompt_ensemble: bool = Field(default=True, validation_alias="MET_CLIP_PROMPT_ENSEMBLE")
    enable_vision: bool = Field(default=False, validation_alias="MET_ENABLE_VISION")
    vision_max_labels: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "Settings":
        repo_root = Path(__file__).resolve().parents[1]

        def _abs(path_value: str | None) -> str | None:
            if not path_value:
                return path_value
            path = Path(path_value)
            if path.is_absolute():
                return str(path)
            return str((repo_root / path).resolve())

        self.data_csv = _abs(self.data_csv) or self.data_csv
        self.images_dir = _abs(self.images_dir) or self.images_dir
        self.artifacts_dir = _abs(self.artifacts_dir) or self.artifacts_dir
        return self


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
