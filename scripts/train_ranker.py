from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import normalize

from src.config import get_settings


def _load_numeric_features(artifacts: Path, meta: pd.DataFrame) -> np.ndarray:
    numeric_path = artifacts / "numeric_features.csv"
    if not numeric_path.exists():
        return np.zeros((len(meta), 0), dtype=np.float32)
    numeric = pd.read_csv(numeric_path)
    merged = meta[["objectID"]].merge(numeric, on="objectID", how="left").fillna(0.0)
    cols = [col for col in merged.columns if col != "objectID"]
    if not cols:
        return np.zeros((len(meta), 0), dtype=np.float32)
    return merged[cols].to_numpy(dtype=np.float32)


def _build_pair_features(
    embeddings: np.ndarray,
    numeric: np.ndarray,
    idx_a: int,
    idx_b: int,
) -> np.ndarray:
    emb_diff = np.abs(embeddings[idx_a] - embeddings[idx_b])
    cosine = np.array([float(np.dot(embeddings[idx_a], embeddings[idx_b]))], dtype=np.float32)
    num_diff = np.abs(numeric[idx_a] - numeric[idx_b]) if numeric.shape[1] > 0 else np.zeros((0,), dtype=np.float32)
    return np.concatenate([emb_diff, cosine, num_diff]).astype(np.float32)


def _sample_pairs(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    numeric: np.ndarray,
    hard_negatives_per_anchor: int = 2,
    random_negatives_per_anchor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    groups = meta.groupby("department", dropna=False).indices
    n = len(meta)
    rng = np.random.default_rng(42)
    sims = embeddings @ embeddings.T

    feats: list[np.ndarray] = []
    labels: list[int] = []

    for idx, row in meta.iterrows():
        idx_i = int(cast(Any, idx))
        dept = row.get("department")
        positives = [int(i) for i in groups.get(dept, []) if int(i) != idx_i]
        if positives:
            # hard positive: most semantically similar same-department item.
            pos_scores = [(int(p), float(sims[idx_i, int(p)])) for p in positives]
            pos_scores.sort(key=lambda x: x[1], reverse=True)
            p = int(pos_scores[0][0])
            feats.append(_build_pair_features(embeddings, numeric, idx_i, p))
            labels.append(1)

        not_same_dept = [int(j) for j in range(n) if int(j) not in groups.get(dept, []) and int(j) != idx_i]
        if not_same_dept:
            # hard negatives: nearest neighbors outside department.
            neg_scores = [(j, float(sims[idx_i, j])) for j in not_same_dept]
            neg_scores.sort(key=lambda x: x[1], reverse=True)
            for neg_idx, _ in neg_scores[:hard_negatives_per_anchor]:
                feats.append(_build_pair_features(embeddings, numeric, idx_i, int(neg_idx)))
                labels.append(0)

            # random negatives for robustness.
            if random_negatives_per_anchor > 0:
                sampled = rng.choice(not_same_dept, size=min(random_negatives_per_anchor, len(not_same_dept)), replace=False)
                for neg in sampled.tolist():
                    feats.append(_build_pair_features(embeddings, numeric, idx_i, int(neg)))
                    labels.append(0)

    if not feats:
        raise RuntimeError("No training pairs found.")
    return np.vstack(feats), np.asarray(labels, dtype=np.int32)


def _train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(123)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = max(1, int(len(idx) * (1.0 - val_ratio)))
    split = min(split, len(idx) - 1) if len(idx) > 1 else len(idx)
    train_idx = idx[:split]
    val_idx = idx[split:] if split < len(idx) else idx[:0]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def main() -> None:
    settings = get_settings()
    artifacts = Path(settings.artifacts_dir)
    embeddings = normalize(np.load(artifacts / "embeddings.npz")["embeddings"], norm="l2", axis=1)
    meta = pd.read_csv(artifacts / "meta.csv")
    numeric = _load_numeric_features(artifacts, meta)

    X, y = _sample_pairs(embeddings, meta, numeric)
    X_train, y_train, X_val, y_val = _train_val_split(X, y)
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training labels are degenerate. Need both positive and negative samples.")

    model_cls = getattr(lgb, "LGBMClassifier")
    model = model_cls(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
    )
    fit_kwargs: dict[str, Any] = {}
    if len(y_val) > 0 and len(np.unique(y_val)) > 1:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_metric"] = "auc"
        fit_kwargs["callbacks"] = [lgb.early_stopping(40, verbose=False)]
    model.fit(X_train, y_train, **fit_kwargs)

    joblib.dump(model, artifacts / "lightgbm_ranker.joblib")
    metrics: dict[str, Any] = {
        "n_samples_total": int(len(y)),
        "n_samples_train": int(len(y_train)),
        "n_samples_val": int(len(y_val)),
        "positive_rate_train": float(np.mean(y_train)),
        "best_iteration": int(getattr(model, "best_iteration_", -1) or -1),
    }
    if len(y_val) > 0 and len(np.unique(y_val)) > 1:
        probs = model.predict_proba(X_val)[:, 1]
        metrics["val_auc"] = float(roc_auc_score(y_val, probs))
        metrics["val_average_precision"] = float(average_precision_score(y_val, probs))
    (artifacts / "ranker_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
