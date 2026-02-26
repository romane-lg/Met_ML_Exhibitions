from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.features.clip_features import CLIPEncoder
from src.features.nlp_utils import tokenize_text


@dataclass
class Recommendation:
    object_id: int
    score: float
    raw_score: float
    similarity_score: float
    title: str | None
    artist: str | None
    department: str | None
    object_date: str | None
    medium: str | None
    image_path: str | None


class Ranker(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class LDAModel(Protocol):
    def transform(self, X: object) -> np.ndarray: ...


class ExhibitionRecommender:
    """Content-based recommender backed by dense combined embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        text_vectorizer: TfidfVectorizer | None = None,
        ranker: object | None = None,
        numeric_features: np.ndarray | None = None,
        numeric_columns: list[str] | None = None,
        lda_model: LDAModel | None = None,
        embedding_backend: str = "tfidf",
        clip_model_name: str | None = None,
        clip_pretrained: str | None = None,
        clip_device: str = "cpu",
        clip_batch_size: int = 32,
        clip_similarity_weight: float = 0.8,
        clip_lexical_weight: float = 0.2,
        clip_prompt_ensemble: bool = True,
    ) -> None:
        self.embeddings = normalize(np.asarray(embeddings), norm="l2", axis=1)
        self.metadata = metadata.reset_index(drop=True)
        self.text_vectorizer = text_vectorizer
        self.ranker: Ranker | None = ranker
        self.lda_model: LDAModel | None = lda_model
        self.embedding_backend = embedding_backend
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.clip_device = clip_device
        self.clip_batch_size = clip_batch_size
        self.clip_similarity_weight = float(max(0.0, clip_similarity_weight))
        self.clip_lexical_weight = float(max(0.0, clip_lexical_weight))
        weight_sum = self.clip_similarity_weight + self.clip_lexical_weight
        if weight_sum > 0:
            self.clip_similarity_weight /= weight_sum
            self.clip_lexical_weight /= weight_sum
        else:
            self.clip_similarity_weight = 1.0
            self.clip_lexical_weight = 0.0
        self.clip_prompt_ensemble = clip_prompt_ensemble
        self._clip_encoder: CLIPEncoder | None = None
        self.numeric_features = (
            np.asarray(numeric_features, dtype=np.float32)
            if numeric_features is not None
            else np.zeros((len(self.metadata), 0), dtype=np.float32)
        )
        self.numeric_columns = numeric_columns or []
        self.nn = NearestNeighbors(metric="cosine")
        self.nn.fit(self.embeddings)
        self.id_to_idx = {
            int(row.objectID): idx
            for idx, row in self.metadata.iterrows()
            if pd.notna(row.objectID)
        }
        self._lexical_docs = self._metadata_docs(self.metadata)
        self._lexical_vectorizer = TfidfVectorizer(min_df=1, max_features=12000, ngram_range=(1, 2))
        self._lexical_matrix = self._lexical_vectorizer.fit_transform(self._lexical_docs)
        self.last_reranker_status: dict[str, object] = {
            "reranker_used": self.ranker is not None,
            "fallback_reason": None,
        }

    @classmethod
    def from_artifacts(cls, artifacts_dir: str) -> ExhibitionRecommender:
        base = Path(artifacts_dir)
        embeddings = np.load(base / "embeddings.npz")["embeddings"]
        metadata = pd.read_csv(base / "meta.csv")
        backend_meta_path = base / "embedding_backend.json"
        backend_payload: dict[str, Any] = {}
        if backend_meta_path.exists():
            backend_payload = json.loads(backend_meta_path.read_text(encoding="utf-8"))
        backend = str(backend_payload.get("backend", "tfidf")).strip().lower()

        text_vectorizer: TfidfVectorizer | None = None
        clip_model_name: str | None = None
        clip_pretrained: str | None = None
        clip_device = "cpu"
        clip_batch_size = 32
        clip_similarity_weight = 0.8
        clip_lexical_weight = 0.2
        clip_prompt_ensemble = True

        if backend == "clip":
            clip_meta_path = base / "clip_metadata.joblib"
            if clip_meta_path.exists():
                clip_meta = joblib.load(clip_meta_path)
                clip_model_name = str(clip_meta.get("model_name", "ViT-B-32"))
                clip_pretrained = str(clip_meta.get("pretrained", "laion2b_s34b_b79k"))
                clip_device = str(clip_meta.get("device", "cpu"))
                clip_batch_size = int(clip_meta.get("batch_size", 32))
                clip_similarity_weight = float(clip_meta.get("retrieval_clip_weight", 0.8))
                clip_lexical_weight = float(clip_meta.get("retrieval_lexical_weight", 0.2))
                clip_prompt_ensemble = bool(clip_meta.get("prompt_ensemble", True))
            else:
                clip_model_name = str(backend_payload.get("model_name", "ViT-B-32"))
                clip_pretrained = str(backend_payload.get("pretrained", "laion2b_s34b_b79k"))
                clip_device = str(backend_payload.get("device", "cpu"))
                clip_similarity_weight = float(backend_payload.get("retrieval_clip_weight", 0.8))
                clip_lexical_weight = float(backend_payload.get("retrieval_lexical_weight", 0.2))
                clip_prompt_ensemble = bool(backend_payload.get("prompt_ensemble", True))
        else:
            vectorizer_path = base / "text_vectorizer.joblib"
            if vectorizer_path.exists():
                text_vectorizer = joblib.load(vectorizer_path)
            else:
                raise FileNotFoundError(f"Missing TF-IDF artifact: {vectorizer_path}")
        ranker = None
        ranker_path = base / "xgboost_ranker.json"
        if ranker_path.exists():
            try:
                import xgboost as xgb

                ranker = xgb.XGBRanker()
                ranker.load_model(str(ranker_path))
            except Exception:
                ranker = None
        lda_path = base / "lda_model.joblib"
        lda_model = joblib.load(lda_path) if lda_path.exists() else None
        numeric_path = base / "numeric_features.csv"
        if numeric_path.exists():
            numeric = pd.read_csv(numeric_path)
            merged = metadata[["objectID"]].merge(numeric, on="objectID", how="left").fillna(0.0)
            numeric_columns = [col for col in merged.columns if col != "objectID"]
            numeric_features = (
                merged[numeric_columns].to_numpy(dtype=np.float32) if numeric_columns else None
            )
        else:
            numeric_columns = []
            numeric_features = None
        return cls(
            embeddings,
            metadata,
            text_vectorizer,
            ranker,
            numeric_features,
            numeric_columns,
            lda_model,
            embedding_backend=backend,
            clip_model_name=clip_model_name,
            clip_pretrained=clip_pretrained,
            clip_device=clip_device,
            clip_batch_size=clip_batch_size,
            clip_similarity_weight=clip_similarity_weight,
            clip_lexical_weight=clip_lexical_weight,
            clip_prompt_ensemble=clip_prompt_ensemble,
        )

    def recommend_for_theme(
        self,
        theme_query: str,
        n_recommendations: int = 10,
        exclude_ids: Iterable[int] | None = None,
        min_score: float = 0.0,
        diversity_lambda: float = 0.75,
        max_per_artist: int = 2,
        max_per_department: int = 3,
    ) -> pd.DataFrame:
        if not (0.0 <= diversity_lambda <= 1.0):
            raise ValueError("diversity_lambda must be in [0, 1].")
        if max_per_artist < 1:
            raise ValueError("max_per_artist must be >= 1.")
        if max_per_department < 1:
            raise ValueError("max_per_department must be >= 1.")
        if n_recommendations < 1:
            raise ValueError("n_recommendations must be >= 1.")

        positive_tokens, negative_clauses = self._parse_query_intent(theme_query)
        scores = self._score_with_negation(positive_tokens, negative_clauses)

        excluded = set(exclude_ids or [])
        if exclude_ids:
            mask = self.metadata["objectID"].astype("Int64").isin(excluded)
            scores[mask.to_numpy()] = -1.0

        pool_size = min(len(self.metadata), max(n_recommendations * 8, 80))
        ranked = np.argsort(scores)[::-1][:pool_size]
        qarr = self._query_vector(positive_tokens)
        reranked_scores = self._rerank_scores(theme_query, qarr, scores, ranked)
        calibrated = self._calibrate_scores(reranked_scores)
        return self._select_diverse_recommendations(
            ranked_indices=ranked,
            ranked_scores=calibrated,
            raw_scores=reranked_scores,
            similarity_scores=scores,
            n_recommendations=n_recommendations,
            min_score=min_score,
            excluded_ids=excluded,
            diversity_lambda=diversity_lambda,
            max_per_artist=max_per_artist,
            max_per_department=max_per_department,
        )

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
        query = qarr.reshape(1, -1)
        clip_scores = cosine_similarity(self.embeddings, query).ravel()
        if self.embedding_backend != "clip":
            return clip_scores
        lexical_scores = self._lexical_scores(tokens)
        return (
            (self.clip_similarity_weight * clip_scores)
            + (self.clip_lexical_weight * lexical_scores)
        ).astype(np.float32)

    def _score_with_negation(
        self,
        positive_tokens: list[str],
        negative_clauses: list[list[str]],
        negative_weight: float = 0.35,
    ) -> np.ndarray:
        base_scores = self.score_by_tokens(positive_tokens)
        if not negative_clauses:
            return base_scores

        penalties = np.zeros_like(base_scores, dtype=np.float32)
        for clause in negative_clauses:
            if not clause:
                continue
            clause_scores = self.score_by_tokens(clause).astype(np.float32)
            penalties = np.maximum(penalties, clause_scores)
        return (base_scores.astype(np.float32) - (np.float32(negative_weight) * penalties)).astype(
            np.float32
        )

    def _query_vector(self, tokens: list[str]) -> np.ndarray:
        query = " ".join(tokens)
        if self.embedding_backend == "clip":
            encoder = self._get_clip_encoder()
            if self.clip_prompt_ensemble:
                prompts = self._clip_query_prompts(query)
            else:
                prompts = [query]
            emb = encoder.encode_texts(prompts).astype(np.float32)
            if emb.size == 0:
                return np.zeros((self.embeddings.shape[1],), dtype=np.float32)
            combined = emb.mean(axis=0, keepdims=True)
            return normalize(combined, norm="l2", axis=1).astype(np.float32)[0]

        if self.text_vectorizer is None:
            raise RuntimeError("TF-IDF backend requires a loaded text_vectorizer.")
        qvec = self.text_vectorizer.transform([query])
        tfidf_norm = normalize(qvec, norm="l2", axis=1).toarray().astype(np.float32)
        if self.lda_model is not None:
            topic_vec = self.lda_model.transform(qvec).astype(np.float32)
            combined = np.hstack([tfidf_norm, topic_vec])
            return normalize(combined, norm="l2", axis=1).ravel()
        return tfidf_norm.ravel()

    def _get_clip_encoder(self) -> CLIPEncoder:
        if self._clip_encoder is not None:
            return self._clip_encoder
        if not self.clip_model_name or not self.clip_pretrained:
            raise RuntimeError("CLIP backend metadata is incomplete. Rebuild CLIP artifacts.")
        self._clip_encoder = CLIPEncoder(
            model_name=self.clip_model_name,
            pretrained=self.clip_pretrained,
            device=self.clip_device,
            batch_size=self.clip_batch_size,
        )
        return self._clip_encoder

    def _rerank_scores(
        self,
        theme_query: str,
        qarr: np.ndarray,
        base_scores: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        base = base_scores[candidate_indices].astype(np.float32)
        if self.ranker is None:
            self.last_reranker_status = {
                "reranker_used": False,
                "fallback_reason": "xgboost_ranker_missing",
            }
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
            rank_scores = self.ranker.predict(X).astype(np.float32)
        except Exception as exc:
            self.last_reranker_status = {
                "reranker_used": False,
                "fallback_reason": f"xgboost_ranker_failed:{type(exc).__name__}",
            }
            return base
        self.last_reranker_status = {"reranker_used": True, "fallback_reason": None}
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

    def _select_diverse_recommendations(
        self,
        ranked_indices: np.ndarray,
        ranked_scores: np.ndarray,
        raw_scores: np.ndarray,
        similarity_scores: np.ndarray,
        n_recommendations: int,
        min_score: float,
        excluded_ids: set[int],
        diversity_lambda: float,
        max_per_artist: int,
        max_per_department: int,
    ) -> pd.DataFrame:
        score_by_idx = {
            int(idx): float(score) for idx, score in zip(ranked_indices.tolist(), ranked_scores.tolist())
        }
        raw_score_by_idx = {
            int(idx): float(score) for idx, score in zip(ranked_indices.tolist(), raw_scores.tolist())
        }
        similarity_by_idx = {
            int(idx): float(similarity_scores[int(idx)]) for idx in ranked_indices.tolist()
        }
        candidate_indices = [int(idx) for idx in ranked_indices.tolist()]
        selected: list[int] = []
        selected_rows: list[Recommendation] = []
        artist_counts: dict[str, int] = {}
        department_counts: dict[str, int] = {}

        while candidate_indices and len(selected) < n_recommendations:
            best_idx: int | None = None
            best_mmr = -np.inf
            best_relevance = -np.inf

            for idx in candidate_indices:
                object_id = int(self.metadata.iloc[idx].get("objectID"))
                if object_id in excluded_ids:
                    continue
                relevance = score_by_idx.get(idx, 0.0)
                if relevance < min_score:
                    continue

                item = self.metadata.iloc[idx]
                artist = str(item.get("artist") or "").strip().lower()
                department = str(item.get("department") or "").strip().lower()
                if artist and artist_counts.get(artist, 0) >= max_per_artist:
                    continue
                if department and department_counts.get(department, 0) >= max_per_department:
                    continue

                if not selected:
                    mmr_score = relevance
                else:
                    max_sim_to_selected = max(
                        float(np.dot(self.embeddings[idx], self.embeddings[chosen])) for chosen in selected
                    )
                    mmr_score = (diversity_lambda * relevance) - (
                        (1.0 - diversity_lambda) * max_sim_to_selected
                    )

                if mmr_score > best_mmr or (
                    np.isclose(mmr_score, best_mmr) and relevance > best_relevance
                ):
                    best_idx = idx
                    best_mmr = mmr_score
                    best_relevance = relevance

            if best_idx is None:
                break

            selected.append(best_idx)
            score = score_by_idx.get(best_idx, 0.0)
            raw_score = raw_score_by_idx.get(best_idx, score)
            similarity_score = similarity_by_idx.get(best_idx, score)
            selected_rows.append(self._row(best_idx, score, raw_score, similarity_score))

            chosen = self.metadata.iloc[best_idx]
            chosen_artist = str(chosen.get("artist") or "").strip().lower()
            chosen_department = str(chosen.get("department") or "").strip().lower()
            if chosen_artist:
                artist_counts[chosen_artist] = artist_counts.get(chosen_artist, 0) + 1
            if chosen_department:
                department_counts[chosen_department] = department_counts.get(chosen_department, 0) + 1
            candidate_indices = [idx for idx in candidate_indices if idx != best_idx]

        return pd.DataFrame([row.__dict__ for row in selected_rows])

    def _query_numeric_features(self, query: str) -> np.ndarray:
        if self.numeric_features.shape[1] == 0:
            return np.zeros((0,), dtype=np.float32)
        vector = np.zeros((len(self.numeric_columns),), dtype=np.float32)
        lower = query.lower()
        years = [
            float(tok)
            for tok in lower.replace("-", " ").split()
            if tok.isdigit() and 3 <= len(tok) <= 4
        ]
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

    def _row(
        self,
        idx: int,
        score: float,
        raw_score: float | None = None,
        similarity_score: float | None = None,
    ) -> Recommendation:
        item = self.metadata.iloc[idx]
        return Recommendation(
            object_id=int(item.get("objectID")),
            score=score,
            raw_score=score if raw_score is None else raw_score,
            similarity_score=score if similarity_score is None else similarity_score,
            title=item.get("title"),
            artist=item.get("artist"),
            department=item.get("department"),
            object_date=item.get("objectDate"),
            medium=item.get("medium"),
            image_path=item.get("image_path"),
        )

    @staticmethod
    def _simple_tokenize(text: str) -> list[str]:
        return tokenize_text(text)

    @staticmethod
    def _parse_query_intent(theme_query: str) -> tuple[list[str], list[list[str]]]:
        tokens_raw = re.findall(r"[a-z0-9']+", str(theme_query).lower())
        if not tokens_raw:
            return [], []

        neg_markers = {"no", "not", "without", "exclude", "excluding", "except", "minus"}
        clause_stops = {"and", "or", "but"}

        positive_raw: list[str] = []
        negative_clauses_raw: list[list[str]] = []
        idx = 0
        while idx < len(tokens_raw):
            token = tokens_raw[idx]
            if token in neg_markers:
                idx += 1
                clause: list[str] = []
                while idx < len(tokens_raw):
                    current = tokens_raw[idx]
                    if current in neg_markers or current in clause_stops:
                        break
                    clause.append(current)
                    idx += 1
                if clause:
                    negative_clauses_raw.append(clause)
                continue
            positive_raw.append(token)
            idx += 1

        positive_tokens = tokenize_text(" ".join(positive_raw)) if positive_raw else tokenize_text(theme_query)
        if not positive_tokens:
            positive_tokens = tokenize_text(theme_query)
        negative_clauses = [tokenize_text(" ".join(chunk)) for chunk in negative_clauses_raw]
        negative_clauses = [chunk for chunk in negative_clauses if chunk]
        return positive_tokens, negative_clauses

    @staticmethod
    def _metadata_docs(metadata: pd.DataFrame) -> list[str]:
        cols = ["title", "artist", "department", "objectDate", "medium"]
        docs: list[str] = []
        for row in metadata.to_dict(orient="records"):
            parts = [str(row.get(col) or "").strip() for col in cols]
            docs.append(" ".join(p for p in parts if p).lower())
        return docs

    def _lexical_scores(self, tokens: list[str]) -> np.ndarray:
        if not tokens:
            return np.zeros(len(self.metadata), dtype=np.float32)
        query = " ".join(tokens)
        qvec = self._lexical_vectorizer.transform([query])
        return cosine_similarity(self._lexical_matrix, qvec).ravel().astype(np.float32)

    @staticmethod
    def _clip_query_prompts(query: str) -> list[str]:
        q = query.strip().lower()
        expansion_map = {
            "portrait": ["portrait painting of a person", "self portrait artwork", "painted portrait"],
            "portraits": ["portrait painting of a person", "self portrait artwork", "painted portrait"],
            "christian": [
                "christian religious iconography",
                "christian devotional artwork",
                "christian sacred art",
            ],
            "egypt": ["ancient egyptian artwork", "egyptian sculpture", "pharaonic art"],
        }
        expansions = expansion_map.get(q, [])
        base = [q] + expansions
        prompts: list[str] = []
        for item in base:
            prompts.append(item)
            prompts.append(f"a museum artwork depicting {item}")
            prompts.append(f"a fine art image of {item}")
        # preserve order and uniqueness
        return list(dict.fromkeys(prompts))
