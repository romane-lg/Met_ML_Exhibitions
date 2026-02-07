from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import get_settings
from src.features.image_features import (
    clean_vision_response,
    extract_numeric_features,
    vision_tokens_from_features,
)
from src.loaders import VisionAPILoader

logger = logging.getLogger(__name__)


def build_text(row: pd.Series) -> str:
    parts = [
        str(row.get("title") or ""),
        str(row.get("artist") or ""),
        str(row.get("department") or ""),
        str(row.get("objectDate") or ""),
        str(row.get("medium") or ""),
        str(row.get("description") or ""),
    ]
    return " ".join(p for p in parts if p).strip()


def tokenize_local(text: str) -> list[str]:
    return [t.lower() for t in text.replace("/", " ").replace(";", " ").split() if t]


def resolve_image_path(raw_image_path: str, images_dir: str) -> Path:
    path = Path(raw_image_path)
    if path.is_absolute():
        return path

    images_base = Path(images_dir)
    if path.parts and path.parts[0].lower() == "images":
        return images_base.parent / path
    return images_base / path


def is_supported_image_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return False
    signatures = [
        b"\xff\xd8\xff",  # JPEG
        b"\x89PNG\r\n\x1a\n",  # PNG
        b"GIF87a",  # GIF
        b"GIF89a",  # GIF
        b"RIFF",  # WEBP (container check)
        b"BM",  # BMP
        b"II*\x00",  # TIFF little-endian
        b"MM\x00*",  # TIFF big-endian
    ]
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return True
    return any(header.startswith(sig) for sig in signatures if sig != b"RIFF")


def extract_metadata_numeric_features(row: pd.Series) -> dict[str, float]:
    date_text = str(row.get("objectDate") or "")
    years = [float(match) for match in re.findall(r"\d{3,4}", date_text)]
    return {
        "meta_has_year": 1.0 if years else 0.0,
        "meta_year_mean": float(sum(years) / len(years)) if years else 0.0,
    }


def run_build(limit: int | None = None, force: bool = False, offline: bool = False) -> None:
    settings = get_settings()
    data_csv = Path(settings.data_csv)
    artifacts = Path(settings.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    if not data_csv.exists():
        raise RuntimeError(f"Metadata file not found at {data_csv}")

    emb_path = artifacts / "embeddings.npz"
    meta_path = artifacts / "meta.csv"
    tok_path = artifacts / "tokens.json"
    vec_path = artifacts / "text_vectorizer.joblib"
    if emb_path.exists() and meta_path.exists() and tok_path.exists() and vec_path.exists() and not force:
        print("Artifacts already exist. Use --force to rebuild.")
        return

    df = pd.read_csv(data_csv)
    if limit:
        df = df.head(limit)
    total_rows = len(df)
    logger.info("Starting feature build for %d records (force=%s, offline=%s)", total_rows, force, offline)

    use_vision = not offline and settings.enable_vision
    if use_vision:
        creds_path = settings.google_credentials
        if not creds_path:
            raise RuntimeError(
                "Vision output is missing and credentials are not set. "
                "Add your key at config/service_account.json and set "
                "GOOGLE_APPLICATION_CREDENTIALS=config/service_account.json in .env."
            )
        if not Path(creds_path).exists():
            raise RuntimeError(
                "Vision output is missing and credentials file was not found: "
                f"{creds_path}. Add your key at config/service_account.json."
            )
    loader = VisionAPILoader(credentials_path=settings.google_credentials) if use_vision else None
    cache: dict[str, dict[str, list[str]]] = {}
    if tok_path.exists() and not force:
        cache = json.loads(tok_path.read_text(encoding="utf-8"))

    docs = []
    descriptions = []
    numeric_rows: list[dict[str, float | int]] = []
    vision_errors: list[dict[str, str]] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        oid = str(int(getattr(row, "objectID")))
        logger.info("Processing %d/%d - objectID=%s", idx, total_rows, oid)
        row_series = pd.Series(row._asdict())
        numeric_features = extract_metadata_numeric_features(row_series)
        if oid in cache and not force:
            text_tokens = cache[oid].get("text", [])
            image_tokens = cache[oid].get("image", [])
            numeric_features.update(extract_numeric_features({}))
            logger.info("Using cached tokens for objectID=%s", oid)
        else:
            text = build_text(row_series)
            text_tokens = tokenize_local(text)
            image_tokens: list[str] = []
            if loader is not None:
                image_path = resolve_image_path(str(getattr(row, "image_path", "") or ""), settings.images_dir)
                if not image_path.exists():
                    vision_errors.append(
                        {"objectID": oid, "image_path": str(image_path), "error": "missing_image_file"}
                    )
                    logger.warning("Missing image for objectID=%s path=%s", oid, image_path)
                elif not is_supported_image_file(image_path):
                    vision_errors.append(
                        {"objectID": oid, "image_path": str(image_path), "error": "unsupported_or_corrupt_image"}
                    )
                    logger.warning("Unsupported/corrupt image for objectID=%s path=%s", oid, image_path)
                else:
                    try:
                        raw_features = loader.load_image_features(str(image_path))
                        numeric_features.update(extract_numeric_features(raw_features))
                        features = clean_vision_response(raw_features)
                        image_tokens = vision_tokens_from_features(features)
                        if not image_tokens:
                            vision_errors.append(
                                {
                                    "objectID": oid,
                                    "image_path": str(image_path),
                                    "error": "empty_vision_response",
                                }
                            )
                            logger.warning("Empty Vision response for objectID=%s", oid)
                    except Exception as exc:
                        vision_errors.append(
                            {
                                "objectID": oid,
                                "image_path": str(image_path),
                                "error": f"vision_exception:{type(exc).__name__}",
                            }
                        )
                        logger.exception("Vision exception for objectID=%s", oid)
            else:
                numeric_features.update(extract_numeric_features({}))
            cache[oid] = {"text": text_tokens, "image": image_tokens}
        numeric_rows.append({"objectID": int(getattr(row, "objectID")), **numeric_features})

        merged = text_tokens + image_tokens
        docs.append(" ".join(merged))
        descriptions.append(
            {
                "objectID": int(getattr(row, "objectID")),
                "description": " ".join(
                    [build_text(pd.Series(row._asdict())), "vision_tokens:", " ".join(image_tokens), "text_tokens:", " ".join(text_tokens)]
                ).strip(),
            }
        )

    if not any(d.strip() for d in docs):
        docs = ["_empty_"] * len(docs)

    vectorizer = TfidfVectorizer(min_df=1, max_features=10000, ngram_range=(1, 2))
    mat = vectorizer.fit_transform(docs)
    emb = normalize(mat, norm="l2", axis=1).toarray().astype(np.float32)

    np.savez_compressed(emb_path, embeddings=emb)
    df.to_csv(meta_path, index=False)
    pd.DataFrame(descriptions).to_csv(artifacts / "descriptions.csv", index=False)
    if numeric_rows:
        pd.DataFrame(numeric_rows).to_csv(artifacts / "numeric_features.csv", index=False)
    joblib.dump(vectorizer, vec_path)
    tok_path.write_text(json.dumps(cache, ensure_ascii=True, indent=2), encoding="utf-8")
    if vision_errors:
        pd.DataFrame(vision_errors).to_csv(artifacts / "vision_errors.csv", index=False)
        logger.warning("Vision extraction completed with %d image issues", len(vision_errors))
    logger.info("Feature build completed. Artifacts written to %s", artifacts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run_build(limit=args.limit, force=args.force, offline=args.offline)


if __name__ == "__main__":
    main()
