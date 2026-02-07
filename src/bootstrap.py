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


def _required_artifacts(settings: Settings) -> list[Path]:
    base = Path(settings.artifacts_dir)
    return [
        base / "embeddings.npz",
        base / "meta.csv",
        base / "tokens.json",
        base / "text_vectorizer.joblib",
    ]


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


def ensure_artifacts(settings: Settings) -> BootstrapStatus:
    required = _required_artifacts(settings)
    missing = [path for path in required if not path.exists()]
    built = False

    if missing:
        try:
            run_build(force=False, offline=not settings.enable_vision)
            built = True
        except Exception as exc:  # pragma: no cover - surfaced to API/UI state
            return BootstrapStatus(ready=False, built=built, error=str(exc))

    missing_after = [path for path in required if not path.exists()]
    if missing_after:
        joined = ", ".join(str(path) for path in missing_after)
        return BootstrapStatus(ready=False, built=built, error=f"Missing artifacts: {joined}")

    tokens_path = Path(settings.artifacts_dir) / "tokens.json"
    has_vision = _has_vision_tokens(tokens_path)
    if not has_vision:
        return BootstrapStatus(
            ready=True,
            built=built,
            warning=_vision_key_message(settings),
        )

    return BootstrapStatus(ready=True, built=built)
