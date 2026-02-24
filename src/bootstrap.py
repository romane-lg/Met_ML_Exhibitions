from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scripts.build_features import run_build
from src.config import Settings


@dataclass
class BootstrapStatus:
    ready: bool
    built: bool
    error: str | None = None
    warning: str | None = None


def _detect_backend(base: Path, default_backend: str = "tfidf") -> str:
    backend_path = base / "embedding_backend.json"
    if backend_path.exists():
        try:
            payload = json.loads(backend_path.read_text(encoding="utf-8"))
            value = str(payload.get("backend", "tfidf")).strip().lower()
            if value in {"tfidf", "clip"}:
                return value
        except Exception:
            return default_backend
    return default_backend


def _required_artifacts(settings: Settings) -> list[Path]:
    base = Path(settings.artifacts_dir)
    backend = _detect_backend(base, default_backend=settings.embedding_backend)
    required = [
        base / "embeddings.npz",
        base / "meta.csv",
        base / "tokens.json",
    ]
    # Backward compatible: older TF-IDF artifacts may not include this metadata file.
    if (base / "embedding_backend.json").exists():
        required.append(base / "embedding_backend.json")
    if backend == "clip":
        required.append(base / "clip_metadata.joblib")
    else:
        required.append(base / "text_vectorizer.joblib")
    return required


def _has_vision_tokens(tokens_path: Path) -> bool:
    if not tokens_path.exists():
        return False
    data = json.loads(tokens_path.read_text(encoding="utf-8"))
    for item in data.values():
        image_tokens = item.get("image", [])
        if isinstance(image_tokens, list) and any(str(token).strip() for token in image_tokens):
            return True
    return False


def _vision_key_message(settings: Settings) -> str:
    creds = settings.google_credentials or "config/service_account.json"
    return (
        "Google Vision output is missing. Add your service-account key at "
        "`config/service_account.json` and set "
        f"`GOOGLE_APPLICATION_CREDENTIALS={creds}` in `.env`, then run `make build-features`."
    )


def _missing_artifact_message(missing: list[Path]) -> str:
    joined = ", ".join(str(path) for path in missing)
    return (
        "Missing prebuilt artifacts. Run `make build-features` (or enable startup rebuild with "
        "`MET_AUTO_BUILD_ON_STARTUP=true`) before launching the app. Missing: "
        f"{joined}"
    )


def ensure_artifacts(settings: Settings) -> BootstrapStatus:
    required = _required_artifacts(settings)
    missing = [path for path in required if not path.exists()]
    built = False

    if missing:
        if not settings.auto_build_on_startup:
            return BootstrapStatus(ready=False, built=False, error=_missing_artifact_message(missing))
        try:
            run_build(
                force=False,
                offline=not settings.enable_vision,
                embedding_backend=settings.embedding_backend,
                clip_model_name=settings.clip_model_name,
                clip_pretrained=settings.clip_pretrained,
                clip_device=settings.clip_device,
                clip_batch_size=settings.clip_batch_size,
                clip_text_weight=settings.clip_text_weight,
                clip_image_weight=settings.clip_image_weight,
            )
            built = True
        except Exception as exc:  # pragma: no cover - surfaced to API/UI state
            return BootstrapStatus(ready=False, built=built, error=str(exc))

    missing_after = [path for path in required if not path.exists()]
    if missing_after:
        return BootstrapStatus(ready=False, built=built, error=_missing_artifact_message(missing_after))

    tokens_path = Path(settings.artifacts_dir) / "tokens.json"
    has_vision = _has_vision_tokens(tokens_path)
    if settings.enable_vision and not has_vision:
        return BootstrapStatus(
            ready=True,
            built=built,
            warning=_vision_key_message(settings),
        )

    return BootstrapStatus(ready=True, built=built)
