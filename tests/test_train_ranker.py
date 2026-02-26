from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_ranker import _build_pair_features, _sample_pairs, _train_val_split


def test_build_pair_features_shape_and_cosine_slot():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    num = np.array([[1.0, 2.0], [2.0, 5.0]], dtype=np.float32)
    feat = _build_pair_features(emb, num, 0, 1)
    # emb diff (2) + cosine (1) + num diff (2)
    assert feat.shape == (5,)
    assert np.isclose(feat[2], 0.0)  # cosine between orthogonal vectors


def test_train_val_split_non_empty():
    X = np.random.RandomState(0).rand(12, 4).astype(np.float32)
    y = np.array([0, 1] * 6, dtype=np.int32)
    qids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int32)
    Xtr, ytr, qtr, Xv, yv, qv = _train_val_split(X, y, qids, val_ratio=0.25)
    assert len(ytr) > 0
    assert len(yv) > 0
    assert len(ytr) + len(yv) == len(y)
    assert len(qtr) == len(ytr)
    assert len(qv) == len(yv)
    assert Xtr.shape[1] == X.shape[1] == Xv.shape[1]


def test_sample_pairs_returns_positive_and_negative_labels():
    meta = pd.DataFrame(
        {
            "objectID": [1, 2, 3, 4],
            "department": ["A", "A", "B", "B"],
        }
    )
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    numeric = np.zeros((4, 0), dtype=np.float32)
    X, y, qids = _sample_pairs(
        emb,
        meta,
        numeric,
        hard_negatives_per_anchor=1,
        random_negatives_per_anchor=0,
    )
    assert X.shape[0] == y.shape[0]
    assert qids.shape[0] == y.shape[0]
    assert set(np.unique(y).tolist()) == {0, 1}
