from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models import ExhibitionRecommender


QUERY_VARIANTS: dict[str, list[str]] = {
    "portrait": ["portrait", "portraits", "self portrait", "portrait painting"],
    "christian": ["christian", "christian art", "christian religious iconography"],
    "ancient egypt": ["ancient egypt", "egyptian art", "pharaonic art"],
    "religious art": ["religious art", "devotional icon", "sacred art"],
}


@dataclass
class Report:
    artifacts_dir: str
    backend: str
    top_k: int
    num_themes: int
    recall_at_k: float
    ndcg_at_k: float
    mean_score: float
    intra_list_diversity: float
    artist_coverage: float
    department_coverage: float
    independent_unique_id_coverage: float
    exhibition_unique_id_coverage: float
    exhibition_overlap_rate: float
    robustness_jaccard: float
    p95_latency_ms: float
    median_latency_ms: float


def _dcg(binary_rels: list[int]) -> float:
    gains = np.asarray(binary_rels, dtype=np.float32)
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2, dtype=np.float32))
    return float(np.sum(gains / discounts))


def _backend_name(artifacts_dir: Path) -> str:
    path = artifacts_dir / "embedding_backend.json"
    if not path.exists():
        return "tfidf"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("backend", "tfidf")).strip().lower()


def _theme_relevance_metrics(
    rec: ExhibitionRecommender,
    top_k: int,
    min_department_size: int,
) -> tuple[list[float], list[float], list[pd.DataFrame], list[str]]:
    meta = rec.metadata.copy()
    meta["department"] = meta["department"].fillna("").astype(str)
    meta["objectID"] = meta["objectID"].astype("Int64")
    dept_counts = meta["department"].value_counts()
    themes = [
        dept
        for dept, count in dept_counts.items()
        if dept.strip() and int(count) >= int(min_department_size)
    ]

    recalls: list[float] = []
    ndcgs: list[float] = []
    frames: list[pd.DataFrame] = []
    for theme in themes:
        relevant = set(int(v) for v in meta.loc[meta["department"] == theme, "objectID"].dropna().tolist())
        if not relevant:
            continue
        frame = rec.recommend_for_theme(theme, n_recommendations=top_k, min_score=0.0)
        frames.append(frame)
        if frame.empty:
            recalls.append(0.0)
            ndcgs.append(0.0)
            continue
        predicted = [int(v) for v in frame["object_id"].dropna().tolist()[:top_k]]
        hits = [1 if item in relevant else 0 for item in predicted]
        recalls.append(float(sum(hits) / max(1, min(len(relevant), top_k))))
        ideal = [1] * min(len(relevant), top_k)
        idcg = _dcg(ideal)
        ndcgs.append(float(_dcg(hits) / idcg) if idcg > 0 else 0.0)
    return recalls, ndcgs, frames, themes


def _intra_list_diversity(rec: ExhibitionRecommender, frame: pd.DataFrame) -> float:
    if frame.empty or len(frame) < 2:
        return 0.0
    ids = [int(v) for v in frame["object_id"].tolist() if pd.notna(v)]
    idxs = [rec.id_to_idx[i] for i in ids if i in rec.id_to_idx]
    if len(idxs) < 2:
        return 0.0
    emb = rec.embeddings[idxs]
    sims = emb @ emb.T
    mask = ~np.eye(len(idxs), dtype=bool)
    if not np.any(mask):
        return 0.0
    mean_sim = float(np.mean(sims[mask]))
    return float(max(0.0, 1.0 - mean_sim))


def _coverage_metrics(meta: pd.DataFrame, frames: list[pd.DataFrame]) -> tuple[float, float, float]:
    if not frames:
        return 0.0, 0.0, 0.0
    all_artists = {v.strip().lower() for v in meta["artist"].fillna("").astype(str).tolist() if v.strip()}
    all_depts = {
        v.strip().lower() for v in meta["department"].fillna("").astype(str).tolist() if v.strip()
    }
    all_ids = {int(v) for v in meta["objectID"].dropna().tolist()}
    hit_artists: set[str] = set()
    hit_depts: set[str] = set()
    hit_ids: set[int] = set()
    for frame in frames:
        for v in frame.get("artist", pd.Series(dtype=str)).fillna("").astype(str).tolist():
            if v.strip():
                hit_artists.add(v.strip().lower())
        for v in frame.get("department", pd.Series(dtype=str)).fillna("").astype(str).tolist():
            if v.strip():
                hit_depts.add(v.strip().lower())
        for v in frame.get("object_id", pd.Series(dtype=float)).dropna().tolist():
            hit_ids.add(int(v))
    artist_cov = float(len(hit_artists) / max(1, len(all_artists)))
    dept_cov = float(len(hit_depts) / max(1, len(all_depts)))
    id_cov = float(len(hit_ids) / max(1, len(all_ids)))
    return artist_cov, dept_cov, id_cov


def _exhibition_coverage_metrics(
    rec: ExhibitionRecommender,
    themes: list[str],
    top_k: int,
) -> tuple[float, float]:
    if not themes:
        return 0.0, 0.0
    out = rec.recommend_exhibitions(
        themes,
        max_pieces_per_exhibition=top_k,
        min_pieces_per_exhibition=1,
        min_similarity=0.0,
    )
    all_ids = {int(v) for v in rec.metadata["objectID"].dropna().tolist()}
    slots = 0
    unique_ids: set[int] = set()
    for frame in out.values():
        ids = [int(v) for v in frame.get("object_id", pd.Series(dtype=float)).dropna().tolist()]
        slots += len(ids)
        unique_ids.update(ids)
    overlap_count = max(0, slots - len(unique_ids))
    overlap_rate = float(overlap_count / max(1, slots))
    coverage = float(len(unique_ids) / max(1, len(all_ids)))
    return coverage, overlap_rate


def _robustness_jaccard(rec: ExhibitionRecommender, top_k: int) -> float:
    scores: list[float] = []
    for canonical, variants in QUERY_VARIANTS.items():
        del canonical
        variant_sets: list[set[int]] = []
        for q in variants:
            frame = rec.recommend_for_theme(q, n_recommendations=top_k, min_score=0.0)
            variant_sets.append(set(int(v) for v in frame["object_id"].dropna().tolist()))
        for i in range(len(variant_sets)):
            for j in range(i + 1, len(variant_sets)):
                a, b = variant_sets[i], variant_sets[j]
                union = len(a | b)
                if union == 0:
                    continue
                scores.append(float(len(a & b) / union))
    return float(np.mean(scores)) if scores else 0.0


def _latency_metrics(rec: ExhibitionRecommender, top_k: int, n_runs: int) -> tuple[float, float]:
    queries = [q for variants in QUERY_VARIANTS.values() for q in variants]
    if not queries:
        return 0.0, 0.0
    latencies: list[float] = []
    for idx in range(n_runs):
        q = queries[idx % len(queries)]
        t0 = time.perf_counter()
        rec.recommend_for_theme(q, n_recommendations=top_k, min_score=0.0)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    p95 = float(np.percentile(np.asarray(latencies, dtype=np.float32), 95))
    med = float(np.median(np.asarray(latencies, dtype=np.float32)))
    return p95, med


def evaluate(
    artifacts_dir: str,
    top_k: int,
    min_department_size: int,
    latency_runs: int,
) -> Report:
    rec = ExhibitionRecommender.from_artifacts(artifacts_dir)
    recalls, ndcgs, frames, themes = _theme_relevance_metrics(rec, top_k, min_department_size)
    mean_score = float(
        np.mean(
            [float(frame["score"].astype(float).mean()) for frame in frames if not frame.empty]
            or [0.0]
        )
    )
    ild = float(np.mean([_intra_list_diversity(rec, frame) for frame in frames] or [0.0]))
    artist_cov, dept_cov, independent_id_cov = _coverage_metrics(rec.metadata, frames)
    exhibition_id_cov, exhibition_overlap_rate = _exhibition_coverage_metrics(rec, themes, top_k)
    robust = _robustness_jaccard(rec, top_k)
    p95_ms, med_ms = _latency_metrics(rec, top_k, latency_runs)
    return Report(
        artifacts_dir=artifacts_dir,
        backend=_backend_name(Path(artifacts_dir)),
        top_k=top_k,
        num_themes=len(themes),
        recall_at_k=float(np.mean(recalls) if recalls else 0.0),
        ndcg_at_k=float(np.mean(ndcgs) if ndcgs else 0.0),
        mean_score=mean_score,
        intra_list_diversity=ild,
        artist_coverage=artist_cov,
        department_coverage=dept_cov,
        independent_unique_id_coverage=independent_id_cov,
        exhibition_unique_id_coverage=exhibition_id_cov,
        exhibition_overlap_rate=exhibition_overlap_rate,
        robustness_jaccard=robust,
        p95_latency_ms=p95_ms,
        median_latency_ms=med_ms,
    )


def _to_dict(report: Report) -> dict[str, Any]:
    return {
        "artifacts_dir": report.artifacts_dir,
        "backend": report.backend,
        "top_k": report.top_k,
        "num_themes": report.num_themes,
        "recall_at_k": report.recall_at_k,
        "ndcg_at_k": report.ndcg_at_k,
        "mean_score": report.mean_score,
        "intra_list_diversity": report.intra_list_diversity,
        "artist_coverage": report.artist_coverage,
        "department_coverage": report.department_coverage,
        "independent_unique_id_coverage": report.independent_unique_id_coverage,
        "exhibition_unique_id_coverage": report.exhibition_unique_id_coverage,
        "exhibition_overlap_rate": report.exhibition_overlap_rate,
        "robustness_jaccard": report.robustness_jaccard,
        "p95_latency_ms": report.p95_latency_ms,
        "median_latency_ms": report.median_latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comprehensive retrieval evaluation.")
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-department-size", type=int, default=2)
    parser.add_argument("--latency-runs", type=int, default=24)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--csv-out", type=str, default="")
    args = parser.parse_args()

    reports = [
        evaluate(path, top_k=args.top_k, min_department_size=args.min_department_size, latency_runs=args.latency_runs)
        for path in args.artifacts
    ]
    print(
        "| backend | artifacts | recall@k | ndcg@k | ild | artist_cov | dept_cov | id_cov_ind | id_cov_exh | exh_overlap | robust_jaccard | p95_ms |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rep in reports:
        print(
            f"| {rep.backend} | {rep.artifacts_dir} | {rep.recall_at_k:.4f} | {rep.ndcg_at_k:.4f} | "
            f"{rep.intra_list_diversity:.4f} | {rep.artist_coverage:.4f} | {rep.department_coverage:.4f} | "
            f"{rep.independent_unique_id_coverage:.4f} | {rep.exhibition_unique_id_coverage:.4f} | "
            f"{rep.exhibition_overlap_rate:.4f} | {rep.robustness_jaccard:.4f} | {rep.p95_latency_ms:.2f} |"
        )

    if args.json_out:
        payload = [_to_dict(rep) for rep in reports]
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Saved JSON: {args.json_out}")
    if args.csv_out:
        pd.DataFrame([_to_dict(rep) for rep in reports]).to_csv(args.csv_out, index=False)
        print(f"Saved CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
