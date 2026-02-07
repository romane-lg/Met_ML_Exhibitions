from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
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


def main() -> None:
    settings = get_settings()
    artifacts = Path(settings.artifacts_dir)
    embeddings = normalize(np.load(artifacts / "embeddings.npz")["embeddings"], norm="l2", axis=1)
    meta = pd.read_csv(artifacts / "meta.csv")
    numeric = _load_numeric_features(artifacts, meta)

    groups = meta.groupby("department", dropna=False).indices
    rng = np.random.default_rng(42)

    feats = []
    labels = []
    n = len(meta)
    for idx, row in meta.iterrows():
        dept = row.get("department")
        positives = [i for i in groups.get(dept, []) if i != idx]
        if positives:
            p = int(rng.choice(positives))
            emb_diff = np.abs(embeddings[idx] - embeddings[p])
            cosine = np.array([float(np.dot(embeddings[idx], embeddings[p]))], dtype=np.float32)
            num_diff = np.abs(numeric[idx] - numeric[p]) if numeric.shape[1] > 0 else np.zeros((0,), dtype=np.float32)
            feats.append(np.concatenate([emb_diff, cosine, num_diff]))
            labels.append(1)

        for _ in range(2):
            neg = int(rng.integers(0, n))
            if neg == idx:
                continue
            emb_diff = np.abs(embeddings[idx] - embeddings[neg])
            cosine = np.array([float(np.dot(embeddings[idx], embeddings[neg]))], dtype=np.float32)
            num_diff = np.abs(numeric[idx] - numeric[neg]) if numeric.shape[1] > 0 else np.zeros((0,), dtype=np.float32)
            feats.append(np.concatenate([emb_diff, cosine, num_diff]))
            labels.append(0)

    if not feats:
        raise RuntimeError("No training pairs found")

    X = np.vstack(feats)
    y = np.asarray(labels)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)

    joblib.dump(model, artifacts / "lightgbm_ranker.joblib")


if __name__ == "__main__":
    main()
