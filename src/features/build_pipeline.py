from __future__ import annotations

import json
import logging
import pickle
import re
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

from src.config import get_settings
from src.features.clip_features import CLIPEncoder, l2_normalize
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


def _validate_combined_params(
    pca_variance: float,
    pca_max_components: int,
    text_weight: float,
    vision_weight: float,
    numeric_weight: float,
) -> None:
    if not (0.0 < pca_variance <= 1.0):
        raise ValueError("pca_variance must be in the range (0, 1].")
    if pca_max_components < 1:
        raise ValueError("pca_max_components must be >= 1.")
    for name, value in (
        ("text_weight", text_weight),
        ("vision_weight", vision_weight),
        ("numeric_weight", numeric_weight),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0.")


def _validate_embedding_backend(embedding_backend: str) -> Literal["tfidf", "clip"]:
    backend = embedding_backend.strip().lower()
    if backend not in {"tfidf", "clip"}:
        raise ValueError("embedding_backend must be either 'tfidf' or 'clip'.")
    return cast(Literal["tfidf", "clip"], backend)


def build_numeric_feature_matrix(
    object_ids: np.ndarray,
    numeric_rows: list[dict[str, float | int]],
) -> tuple[np.ndarray, list[str]]:
    if len(object_ids) == 0:
        return np.zeros((0, 0), dtype=np.float32), []
    if not numeric_rows:
        return np.zeros((len(object_ids), 0), dtype=np.float32), []

    numeric_frame = pd.DataFrame(numeric_rows)
    aligned = pd.DataFrame({"objectID": object_ids.astype(int)}).merge(
        numeric_frame,
        on="objectID",
        how="left",
    )
    aligned = aligned.fillna(0.0)
    feature_columns = [col for col in aligned.columns if col != "objectID"]
    if not feature_columns:
        return np.zeros((len(object_ids), 0), dtype=np.float32), []
    return aligned[feature_columns].to_numpy(dtype=np.float32), feature_columns


def _vectorize_docs(
    docs: list[str],
    max_features: int,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[np.ndarray, TfidfVectorizer | None]:
    n_samples = len(docs)
    if n_samples == 0:
        return np.zeros((0, 0), dtype=np.float32), None
    if not any(doc.strip() for doc in docs):
        return np.zeros((n_samples, 0), dtype=np.float32), None

    vectorizer = TfidfVectorizer(min_df=1, max_features=max_features, ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(docs).toarray().astype(np.float32)
    return matrix, vectorizer


def _scale_numeric_features(
    numeric_features: np.ndarray,
) -> tuple[np.ndarray, StandardScaler | None]:
    if numeric_features.shape[1] == 0:
        return numeric_features, None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_features).astype(np.float32)
    return scaled, scaler


def _reduce_with_pca(
    features: np.ndarray,
    pca_variance: float,
    pca_max_components: int,
) -> tuple[np.ndarray, PCA | None, dict[str, float | int | bool]]:
    n_samples, n_features = features.shape
    if n_samples == 0:
        return features, None, {
            "explained_variance_ratio_sum": 0.0,
            "selected_components": 0,
            "pca_applied": False,
        }

    if n_features == 0:
        return np.zeros((n_samples, 1), dtype=np.float32), None, {
            "explained_variance_ratio_sum": 0.0,
            "selected_components": 1,
            "pca_applied": False,
        }

    max_allowed = min(pca_max_components, n_samples, n_features)
    max_allowed = max(1, int(max_allowed))
    if max_allowed <= 1 or n_samples <= 1:
        reduced = features[:, :max_allowed].astype(np.float32)
        return reduced, None, {
            "explained_variance_ratio_sum": 1.0 if reduced.shape[1] > 0 else 0.0,
            "selected_components": int(reduced.shape[1]),
            "pca_applied": False,
        }

    pca = PCA(n_components=max_allowed, svd_solver="full", random_state=42)
    projected = pca.fit_transform(features).astype(np.float32)
    explained = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0)
    cumulative = np.cumsum(explained)
    selected = int(np.searchsorted(cumulative, pca_variance, side="left") + 1)
    selected = min(selected, max_allowed)
    selected = max(1, selected)
    reduced = projected[:, :selected].astype(np.float32)
    explained_sum = float(cumulative[selected - 1]) if cumulative.size else 0.0
    return reduced, pca, {
        "explained_variance_ratio_sum": explained_sum,
        "selected_components": selected,
        "pca_applied": True,
    }

def build_combined_embeddings_payload(
    object_ids: np.ndarray,
    text_docs: list[str],
    vision_docs: list[str],
    numeric_rows: list[dict[str, float | int]],
    pca_variance: float = 0.95,
    pca_max_components: int = 256,
    text_weight: float = 1.0,
    vision_weight: float = 1.0,
    numeric_weight: float = 1.0,
) -> dict[str, Any]:
    _validate_combined_params(
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
        text_weight=text_weight,
        vision_weight=vision_weight,
        numeric_weight=numeric_weight,
    )
    if not (len(object_ids) == len(text_docs) == len(vision_docs)):
        raise ValueError("object_ids, text_docs, and vision_docs must have identical lengths.")

    text_features, text_vectorizer = _vectorize_docs(
        text_docs,
        max_features=10000,
        ngram_range=(1, 2),
    )
    vision_features, vision_vectorizer = _vectorize_docs(
        vision_docs,
        max_features=8000,
        ngram_range=(1, 2),
    )
    numeric_features, numeric_columns = build_numeric_feature_matrix(object_ids, numeric_rows)
    scaled_numeric, numeric_scaler = _scale_numeric_features(numeric_features)

    weighted_parts: list[np.ndarray] = []
    if text_features.shape[1] > 0:
        weighted_parts.append(text_features * np.float32(text_weight))
    if vision_features.shape[1] > 0:
        weighted_parts.append(vision_features * np.float32(vision_weight))
    if scaled_numeric.shape[1] > 0:
        weighted_parts.append(scaled_numeric * np.float32(numeric_weight))
    if weighted_parts:
        combined_raw = np.hstack(weighted_parts).astype(np.float32)
    else:
        combined_raw = np.zeros((len(object_ids), 1), dtype=np.float32)

    reduced, pca_model, pca_metrics = _reduce_with_pca(
        combined_raw,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
    )
    embeddings = normalize(reduced, norm="l2", axis=1).astype(np.float32)

    config = {
        "pca_variance": pca_variance,
        "pca_max_components": pca_max_components,
        "text_weight": text_weight,
        "vision_weight": vision_weight,
        "numeric_weight": numeric_weight,
        "row_count": int(len(object_ids)),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    metrics = {
        **pca_metrics,
        "combined_feature_dim_before_pca": int(combined_raw.shape[1]),
        "text_feature_dim": int(text_features.shape[1]),
        "vision_feature_dim": int(vision_features.shape[1]),
        "numeric_feature_dim": int(scaled_numeric.shape[1]),
    }
    return {
        "object_ids": object_ids.astype(np.int64),
        "embeddings": embeddings,
        "pca_model": pca_model,
        "numeric_scaler": numeric_scaler,
        "text_vectorizer": text_vectorizer,
        "vision_vectorizer": vision_vectorizer,
        "numeric_feature_columns": numeric_columns,
        "config": config,
        "metrics": metrics,
    }


def atomic_pickle_dump(payload: dict[str, object], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            dir=str(target_path.parent),
            delete=False,
        ) as tmp_file:
            tmp_file_path = Path(tmp_file.name)
            pickle.dump(payload, tmp_file)
        if tmp_file_path is None:
            raise RuntimeError("Failed to create temporary file for atomic pickle write.")
        tmp_file_path.replace(target_path)
    finally:
        if tmp_file_path is not None and tmp_file_path.exists():
            tmp_file_path.unlink(missing_ok=True)


def run_build(  # noqa: PLR0912, PLR0915
    limit: int | None = None,
    force: bool = False,
    offline: bool = False,
    pca_variance: float = 0.95,
    pca_max_components: int = 256,
    text_weight: float = 1.0,
    vision_weight: float = 1.0,
    numeric_weight: float = 1.0,
    embedding_backend: str = "tfidf",
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
    clip_device: str = "cpu",
    clip_batch_size: int = 32,
    clip_text_weight: float = 0.5,
    clip_image_weight: float = 0.5,
) -> None:
    backend = _validate_embedding_backend(embedding_backend)
    _validate_combined_params(
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
        text_weight=text_weight,
        vision_weight=vision_weight,
        numeric_weight=numeric_weight,
    )
    if clip_text_weight < 0.0 or clip_image_weight < 0.0:
        raise ValueError("clip_text_weight and clip_image_weight must be >= 0.")
    if backend == "clip" and clip_text_weight == 0.0 and clip_image_weight == 0.0:
        raise ValueError("At least one of clip_text_weight or clip_image_weight must be > 0.")
    settings = get_settings()
    data_csv = Path(settings.data_csv)
    artifacts = Path(settings.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    if not data_csv.exists():
        raise RuntimeError(f"Metadata file not found at {data_csv}")

    emb_path = artifacts / "embeddings.npz"
    combined_path = artifacts / "combined_embeddings.pkl"
    meta_path = artifacts / "meta.csv"
    tok_path = artifacts / "tokens.json"
    vec_path = artifacts / "text_vectorizer.joblib"
    clip_meta_path = artifacts / "clip_metadata.joblib"
    backend_path = artifacts / "embedding_backend.json"
    if (
        emb_path.exists()
        and combined_path.exists()
        and meta_path.exists()
        and tok_path.exists()
        and (
            (backend == "tfidf" and vec_path.exists())
            or (backend == "clip" and clip_meta_path.exists())
        )
        and not force
    ):
        print("Artifacts already exist. Use --force to rebuild.")
        return

    df = pd.read_csv(data_csv)
    if limit:
        df = df.head(limit)
    total_rows = len(df)
    logger.info(
        "Starting feature build for %d records (force=%s, offline=%s)",
        total_rows,
        force,
        offline,
    )

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
    prior_numeric_map: dict[str, dict[str, float]] = {}
    prior_numeric_path = artifacts / "numeric_features.csv"
    if prior_numeric_path.exists() and not force:
        prior_numeric = pd.read_csv(prior_numeric_path)
        for row in prior_numeric.to_dict(orient="records"):
            oid_val = row.get("objectID")
            if pd.isna(oid_val):
                continue
            oid = str(int(cast(float | int | str, oid_val)))
            prior_numeric_map[oid] = {
                str(key): float(value)
                for key, value in row.items()
                if key.startswith("vision_") and pd.notna(value)
            }

    docs = []
    text_docs = []
    vision_docs = []
    clip_text_inputs: list[str] = []
    clip_image_paths: list[Path | None] = []
    descriptions = []
    numeric_rows: list[dict[str, float | int]] = []
    vision_errors: list[dict[str, str]] = []
    records = df.to_dict(orient="records")
    for idx, row in enumerate(records, start=1):
        object_id = int(cast(float | int | str, row.get("objectID", 0)))
        oid = str(object_id)
        logger.info("Processing %d/%d - objectID=%s", idx, total_rows, oid)
        row_series = pd.Series(row)
        clip_text_inputs.append(build_text(row_series))
        numeric_features = extract_metadata_numeric_features(row_series)
        image_path_obj: Path | None = resolve_image_path(
            str(row.get("image_path", "") or ""),
            settings.images_dir,
        )
        clip_image_paths.append(image_path_obj)
        if oid in cache and not force:
            text_tokens = cache[oid].get("text", [])
            image_tokens = cache[oid].get("image", [])
            numeric_features.update(prior_numeric_map.get(oid, extract_numeric_features({})))
            logger.info("Using cached tokens for objectID=%s", oid)
        else:
            text = build_text(row_series)
            text_tokens = tokenize_local(text)
            image_tokens: list[str] = []
            if loader is not None:
                image_path = image_path_obj if image_path_obj is not None else Path("")
                if not image_path.exists():
                    vision_errors.append(
                        {
                            "objectID": oid,
                            "image_path": str(image_path),
                            "error": "missing_image_file",
                        }
                    )
                    logger.warning("Missing image for objectID=%s path=%s", oid, image_path)
                elif not is_supported_image_file(image_path):
                    vision_errors.append(
                        {
                            "objectID": oid,
                            "image_path": str(image_path),
                            "error": "unsupported_or_corrupt_image",
                        }
                    )
                    logger.warning(
                        "Unsupported/corrupt image for objectID=%s path=%s",
                        oid,
                        image_path,
                    )
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
        numeric_rows.append({"objectID": object_id, **numeric_features})

        merged = text_tokens + image_tokens
        text_docs.append(" ".join(text_tokens))
        vision_docs.append(" ".join(image_tokens))
        docs.append(" ".join(merged))
        descriptions.append(
            {
                "objectID": object_id,
                "description": " ".join(
                    [
                        build_text(pd.Series(row)),
                        "vision_tokens:",
                        " ".join(image_tokens),
                        "text_tokens:",
                        " ".join(text_tokens),
                    ]
                ).strip(),
            }
        )

    if not any(d.strip() for d in docs):
        docs = ["_empty_"] * len(docs)

    retrieval_vectorizer: TfidfVectorizer | None = None
    clip_metadata: dict[str, Any] | None = None
    clip_text_embeddings: np.ndarray | None = None
    clip_image_embeddings: np.ndarray | None = None

    if backend == "tfidf":
        retrieval_vectorizer = TfidfVectorizer(min_df=1, max_features=10000, ngram_range=(1, 2))
        mat = retrieval_vectorizer.fit_transform(docs)
        emb = normalize(mat, norm="l2", axis=1).toarray().astype(np.float32)
    else:
        encoder = CLIPEncoder(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=clip_device,
            batch_size=clip_batch_size,
        )
        clip_text_embeddings = encoder.encode_texts(clip_text_inputs)
        clip_image_embeddings, clip_image_errors = encoder.encode_images(clip_image_paths)
        if clip_image_errors:
            for item in clip_image_errors:
                idx_txt, reason = item.split(":", maxsplit=1)
                row_idx = int(idx_txt)
                row_data = records[row_idx]
                vision_errors.append(
                    {
                        "objectID": str(int(cast(float | int | str, row_data.get("objectID", 0)))),
                        "image_path": str(clip_image_paths[row_idx]),
                        "error": f"clip_{reason}",
                    }
                )
        weighted = np.zeros_like(clip_text_embeddings, dtype=np.float32)
        if clip_text_weight > 0.0:
            weighted += clip_text_embeddings * np.float32(clip_text_weight)
        if clip_image_weight > 0.0:
            weighted += clip_image_embeddings * np.float32(clip_image_weight)
        emb = l2_normalize(weighted)
        clip_metadata = {
            "model_name": clip_model_name,
            "pretrained": clip_pretrained,
            "device": clip_device,
            "batch_size": int(clip_batch_size),
            "text_weight": float(clip_text_weight),
            "image_weight": float(clip_image_weight),
            "embedding_dimension": int(emb.shape[1]) if emb.ndim == 2 else 0,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

    object_ids = df["objectID"].to_numpy(dtype=np.int64, copy=True)
    combined_payload = build_combined_embeddings_payload(
        object_ids=object_ids,
        text_docs=text_docs,
        vision_docs=vision_docs,
        numeric_rows=numeric_rows,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
        text_weight=text_weight,
        vision_weight=vision_weight,
        numeric_weight=numeric_weight,
    )
    if backend == "clip" and clip_text_embeddings is not None and clip_image_embeddings is not None:
        combined_payload["clip_text_embeddings"] = clip_text_embeddings
        combined_payload["clip_image_embeddings"] = clip_image_embeddings

    np.savez_compressed(emb_path, embeddings=emb)
    atomic_pickle_dump(combined_payload, combined_path)
    df.to_csv(meta_path, index=False)
    pd.DataFrame(descriptions).to_csv(artifacts / "descriptions.csv", index=False)
    if numeric_rows:
        pd.DataFrame(numeric_rows).to_csv(artifacts / "numeric_features.csv", index=False)
    if backend == "tfidf" and retrieval_vectorizer is not None:
        joblib.dump(retrieval_vectorizer, vec_path)
    if backend == "clip" and clip_metadata is not None:
        joblib.dump(clip_metadata, clip_meta_path)
    backend_payload: dict[str, Any] = {
        "backend": backend,
        "dimension": int(emb.shape[1]) if emb.ndim == 2 else 0,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    if clip_metadata is not None:
        backend_payload.update(
            {
                "model_name": clip_metadata["model_name"],
                "pretrained": clip_metadata["pretrained"],
                "device": clip_metadata["device"],
            }
        )
    backend_path.write_text(json.dumps(backend_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tok_path.write_text(json.dumps(cache, ensure_ascii=True, indent=2), encoding="utf-8")
    if vision_errors:
        pd.DataFrame(vision_errors).to_csv(artifacts / "vision_errors.csv", index=False)
        logger.warning("Vision extraction completed with %d image issues", len(vision_errors))
    logger.info("Feature build completed with backend=%s. Artifacts written to %s", backend, artifacts)
