from __future__ import annotations

import functools
from pathlib import Path
from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_csv: str = Field(default="data/raw/met_data.csv", validation_alias="MET_DATA_CSV")
    images_dir: str = Field(default="data/raw/images", validation_alias="MET_IMAGES_DIR")
    artifacts_dir: str = Field(default="artifacts", validation_alias="MET_ARTIFACTS_DIR")
    google_project: str | None = Field(default=None, validation_alias="GOOGLE_CLOUD_PROJECT")
    google_credentials: str | None = Field(
        default=None, validation_alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    enable_vision: bool = Field(default=True, validation_alias="MET_ENABLE_VISION")
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
        self.google_credentials = _abs(self.google_credentials)
        return self


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
