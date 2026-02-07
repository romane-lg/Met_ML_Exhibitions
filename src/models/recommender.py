from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


@dataclass
class Recommendation:
    object_id: int
    score: float
    title: str | None
    artist: str | None
    department: str | None
    object_date: str | None
    medium: str | None
    image_path: str | None


class ExhibitionRecommender:
    """Content-based recommender backed by dense combined embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        text_vectorizer: TfidfVectorizer,
        ranker: object | None = None,
        numeric_features: np.ndarray | None = None,
        numeric_columns: list[str] | None = None,
    ) -> None:
        self.embeddings = normalize(np.asarray(embeddings), norm="l2", axis=1)
        self.metadata = metadata.reset_index(drop=True)
        self.text_vectorizer = text_vectorizer
        self.ranker = ranker
        self.numeric_features = (
            np.asarray(numeric_features, dtype=np.float32)
            if numeric_features is not None
            else np.zeros((len(self.metadata), 0), dtype=np.float32)
        )
        self.numeric_columns = numeric_columns or []
        self.nn = NearestNeighbors(metric="cosine")
        self.nn.fit(self.embeddings)
        self.id_to_idx = {
            int(row.objectID): idx for idx, row in self.metadata.iterrows() if pd.notna(row.objectID)
        }

    @classmethod
    def from_artifacts(cls, artifacts_dir: str) -> "ExhibitionRecommender":
        base = Path(artifacts_dir)
        embeddings = np.load(base / "embeddings.npz")["embeddings"]
        metadata = pd.read_csv(base / "meta.csv")
        text_vectorizer = joblib.load(base / "text_vectorizer.joblib")
        ranker_path = base / "lightgbm_ranker.joblib"
        ranker = joblib.load(ranker_path) if ranker_path.exists() else None
        numeric_path = base / "numeric_features.csv"
        if numeric_path.exists():
            numeric = pd.read_csv(numeric_path)
            merged = metadata[["objectID"]].merge(numeric, on="objectID", how="left").fillna(0.0)
            numeric_columns = [col for col in merged.columns if col != "objectID"]
            numeric_features = merged[numeric_columns].to_numpy(dtype=np.float32) if numeric_columns else None
        else:
            numeric_columns = []
            numeric_features = None
        return cls(embeddings, metadata, text_vectorizer, ranker, numeric_features, numeric_columns)

    def recommend_for_theme(
        self,
        theme_query: str,
        n_recommendations: int = 10,
        exclude_ids: Iterable[int] | None = None,
        min_score: float = 0.0,
    ) -> pd.DataFrame:
        tokens = self._simple_tokenize(theme_query)
        qarr = self._query_vector(tokens)
        scores = (self.embeddings @ qarr.T).ravel()

        if exclude_ids:
            excluded = set(exclude_ids)
            mask = self.metadata["objectID"].astype("Int64").isin(excluded)
            scores[mask.to_numpy()] = -1.0

        pool_size = min(len(self.metadata), max(n_recommendations * 8, 80))
        ranked = np.argsort(scores)[::-1][:pool_size]
        reranked_scores = self._rerank_scores(theme_query, qarr, scores, ranked)
        calibrated = self._calibrate_scores(reranked_scores)
        order = np.argsort(calibrated)[::-1]
        ranked = ranked[order]
        ranked_scores = calibrated[order]
        rows = []
        for pos, idx in enumerate(ranked):
            if len(rows) >= n_recommendations:
                break
            score = float(ranked_scores[pos])
            if score < min_score:
                continue
            row = self._row(idx, score)
            rows.append(row)

        return pd.DataFrame([r.__dict__ for r in rows])

    def recommend_exhibitions(
        self,
        themes: list[str],
        max_pieces_per_exhibition: int = 10,
        min_pieces_per_exhibition: int = 5,
        min_similarity: float = 0.1,
    ) -> dict[str, pd.DataFrame]:
        used_ids: set[int] = set()
        results: dict[str, pd.DataFrame] = {}
        for theme in themes:
            recs = self.recommend_for_theme(
                theme,
                n_recommendations=max_pieces_per_exhibition,
                exclude_ids=used_ids,
                min_score=min_similarity,
            )
            if len(recs) < min_pieces_per_exhibition:
                # fallback without threshold for low-coverage themes
                recs = self.recommend_for_theme(
                    theme,
                    n_recommendations=max_pieces_per_exhibition,
                    exclude_ids=used_ids,
                    min_score=0.0,
                )
            results[theme] = recs
            used_ids.update(int(v) for v in recs.get("object_id", []) if pd.notna(v))
        return results

    def evaluate_coherence(self, artwork_ids: list[int]) -> float:
        indices = [self.id_to_idx[item] for item in artwork_ids if item in self.id_to_idx]
        if len(indices) < 2:
            return 0.0
        subset = self.embeddings[indices]
        similarity = cosine_similarity(subset)
        mask = ~np.eye(len(indices), dtype=bool)
        return float(similarity[mask].mean())

    def score_by_tokens(self, tokens: list[str]) -> np.ndarray:
        if not tokens:
            return np.zeros(len(self.metadata), dtype=float)
        qarr = self._query_vector(tokens)
        return (self.embeddings @ qarr.T).ravel()

    def _query_vector(self, tokens: list[str]) -> np.ndarray:
        query = " ".join(tokens)
        qvec = self.text_vectorizer.transform([query])
        qarr = normalize(qvec, norm="l2", axis=1).toarray().astype(np.float32)
        return qarr.ravel()

    def _rerank_scores(
        self,
        theme_query: str,
        qarr: np.ndarray,
        base_scores: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        base = base_scores[candidate_indices].astype(np.float32)
        if self.ranker is None:
            return base

        query_num = self._query_numeric_features(theme_query)
        feats: list[np.ndarray] = []
        for idx in candidate_indices:
            emb_diff = np.abs(self.embeddings[idx] - qarr)
            cosine = np.array([float(np.dot(self.embeddings[idx], qarr))], dtype=np.float32)
            if self.numeric_features.shape[1] > 0:
                num_diff = np.abs(self.numeric_features[idx] - query_num)
            else:
                num_diff = np.zeros((0,), dtype=np.float32)
            feats.append(np.concatenate([emb_diff, cosine, num_diff]))
        X = np.vstack(feats)
        try:
            if hasattr(self.ranker, "predict_proba"):
                rank_scores = self.ranker.predict_proba(X)[:, 1].astype(np.float32)
            else:
                rank_scores = self.ranker.predict(X).astype(np.float32)
        except Exception:
            return base
        return (0.55 * base + 0.45 * rank_scores).astype(np.float32)

    @staticmethod
    def _calibrate_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_v = float(np.min(scores))
        max_v = float(np.max(scores))
        if max_v - min_v < 1e-8:
            return np.clip(scores, 0.0, 1.0)
        return ((scores - min_v) / (max_v - min_v)).astype(np.float32)

    def _query_numeric_features(self, query: str) -> np.ndarray:
        if self.numeric_features.shape[1] == 0:
            return np.zeros((0,), dtype=np.float32)
        vector = np.zeros((len(self.numeric_columns),), dtype=np.float32)
        lower = query.lower()
        years = [float(tok) for tok in lower.replace("-", " ").split() if tok.isdigit() and 3 <= len(tok) <= 4]
        color_map = {
            "red": (255.0, 0.0, 0.0),
            "blue": (0.0, 0.0, 255.0),
            "green": (0.0, 128.0, 0.0),
            "gold": (212.0, 175.0, 55.0),
            "black": (20.0, 20.0, 20.0),
            "white": (245.0, 245.0, 245.0),
            "brown": (120.0, 80.0, 40.0),
        }
        rgb_hits = [rgb for name, rgb in color_map.items() if name in lower]
        for i, col in enumerate(self.numeric_columns):
            if col == "meta_has_year":
                vector[i] = 1.0 if years else 0.0
            elif col == "meta_year_mean":
                vector[i] = float(np.mean(years)) if years else 0.0
            elif col == "vision_ocr_number_count":
                vector[i] = float(len(years))
            elif col == "vision_ocr_number_mean":
                vector[i] = float(np.mean(years)) if years else 0.0
            elif col == "vision_avg_red" and rgb_hits:
                vector[i] = float(np.mean([rgb[0] for rgb in rgb_hits]))
            elif col == "vision_avg_green" and rgb_hits:
                vector[i] = float(np.mean([rgb[1] for rgb in rgb_hits]))
            elif col == "vision_avg_blue" and rgb_hits:
                vector[i] = float(np.mean([rgb[2] for rgb in rgb_hits]))
        return vector

    def _row(self, idx: int, score: float) -> Recommendation:
        item = self.metadata.iloc[idx]
        return Recommendation(
            object_id=int(item.get("objectID")),
            score=score,
            title=item.get("title"),
            artist=item.get("artist"),
            department=item.get("department"),
            object_date=item.get("objectDate"),
            medium=item.get("medium"),
            image_path=item.get("image_path"),
        )

    @staticmethod
    def _simple_tokenize(text: str) -> list[str]:
        return [t.lower() for t in text.replace("/", " ").replace(";", " ").split() if t]
