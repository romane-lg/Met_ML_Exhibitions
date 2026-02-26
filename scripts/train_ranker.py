from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ndcg_score
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = meta.groupby("department", dropna=False).indices
    n = len(meta)
    rng = np.random.default_rng(42)
    sims = embeddings @ embeddings.T

    feats: list[np.ndarray] = []
    labels: list[int] = []
    qids: list[int] = []

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
            qids.append(idx_i)

        not_same_dept = [int(j) for j in range(n) if int(j) not in groups.get(dept, []) and int(j) != idx_i]
        if not_same_dept:
            # hard negatives: nearest neighbors outside department.
            neg_scores = [(j, float(sims[idx_i, j])) for j in not_same_dept]
            neg_scores.sort(key=lambda x: x[1], reverse=True)
            for neg_idx, _ in neg_scores[:hard_negatives_per_anchor]:
                feats.append(_build_pair_features(embeddings, numeric, idx_i, int(neg_idx)))
                labels.append(0)
                qids.append(idx_i)

            # random negatives for robustness.
            if random_negatives_per_anchor > 0:
                sampled = rng.choice(not_same_dept, size=min(random_negatives_per_anchor, len(not_same_dept)), replace=False)
                for neg in sampled.tolist():
                    feats.append(_build_pair_features(embeddings, numeric, idx_i, int(neg)))
                    labels.append(0)
                    qids.append(idx_i)

    if not feats:
        raise RuntimeError("No training pairs found.")
    return (
        np.vstack(feats),
        np.asarray(labels, dtype=np.int32),
        np.asarray(qids, dtype=np.int32),
    )


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    qids: np.ndarray,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, ...]:
    unique_qids = np.unique(qids)
    if unique_qids.size < 2:
        return X, y, qids, X[:0], y[:0], qids[:0]

    rng = np.random.default_rng(123)
    shuffled_qids = unique_qids.copy()
    rng.shuffle(shuffled_qids)
    split = max(1, int(len(shuffled_qids) * (1.0 - val_ratio)))
    split = min(split, len(shuffled_qids) - 1)
    train_qids = set(int(v) for v in shuffled_qids[:split].tolist())
    val_qids = set(int(v) for v in shuffled_qids[split:].tolist())

    train_mask = np.array([int(qid) in train_qids for qid in qids], dtype=bool)
    val_mask = np.array([int(qid) in val_qids for qid in qids], dtype=bool)
    return X[train_mask], y[train_mask], qids[train_mask], X[val_mask], y[val_mask], qids[val_mask]


def _group_sizes(qids: np.ndarray) -> list[int]:
    if qids.size == 0:
        return []
    sizes: list[int] = []
    current = int(qids[0])
    count = 1
    for qid in qids[1:]:
        qid_i = int(qid)
        if qid_i == current:
            count += 1
        else:
            sizes.append(count)
            current = qid_i
            count = 1
    sizes.append(count)
    return sizes


def _mean_group_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, qids: np.ndarray, k: int = 10) -> float:
    if y_true.size == 0 or y_pred.size == 0 or qids.size == 0:
        return 0.0
    scores: list[float] = []
    for qid in np.unique(qids):
        mask = qids == qid
        rel = y_true[mask]
        pred = y_pred[mask]
        if rel.size < 2:
            continue
        scores.append(float(ndcg_score(rel.reshape(1, -1), pred.reshape(1, -1), k=k)))
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    settings = get_settings()
    artifacts = Path(settings.artifacts_dir)
    embeddings = normalize(np.load(artifacts / "embeddings.npz")["embeddings"], norm="l2", axis=1)
    meta = pd.read_csv(artifacts / "meta.csv")
    numeric = _load_numeric_features(artifacts, meta)

    X, y, qids = _sample_pairs(embeddings, meta, numeric)
    X_train, y_train, qid_train, X_val, y_val, qid_val = _train_val_split(X, y, qids)
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training labels are degenerate. Need both positive and negative samples.")

    train_groups = _group_sizes(qid_train)
    if not train_groups:
        raise RuntimeError("No query groups available for XGBoost LTR training.")
    val_groups = _group_sizes(qid_val)

    model_cls = getattr(xgb, "XGBRanker", None)
    if model_cls is None:
        raise RuntimeError("xgboost.XGBRanker is unavailable in this environment.")
    model = model_cls(
        objective="rank:ndcg",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1.0,
        random_state=42,
        tree_method="hist",
    )
    if len(y_val) > 0 and val_groups:
        model.fit(
            X_train,
            y_train,
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, group=train_groups, verbose=False)

    model_path = artifacts / "xgboost_ranker.json"
    model.save_model(str(model_path))
    metrics: dict[str, Any] = {
        "model_family": "xgboost",
        "objective": "rank:ndcg",
        "n_samples_total": int(len(y)),
        "n_samples_train": int(len(y_train)),
        "n_samples_val": int(len(y_val)),
        "n_queries_total": int(np.unique(qids).size),
        "n_queries_train": int(np.unique(qid_train).size),
        "n_queries_val": int(np.unique(qid_val).size),
        "positive_rate_train": float(np.mean(y_train)) if len(y_train) else 0.0,
        "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
        "feature_dim": int(X.shape[1]),
    }
    if len(y_val) > 0 and len(np.unique(y_val)) > 1 and len(np.unique(qid_val)) > 0:
        val_scores = model.predict(X_val).astype(np.float32)
        metrics["val_ndcg_at_10"] = _mean_group_ndcg_at_k(y_val.astype(np.float32), val_scores, qid_val, k=10)
    (artifacts / "ranker_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
