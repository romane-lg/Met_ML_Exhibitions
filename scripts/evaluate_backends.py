from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import ExhibitionRecommender


@dataclass
class EvalResult:
    backend: str
    artifacts_dir: str
    recall_at_k: float
    ndcg_at_k: float
    artist_coverage: float
    department_coverage: float
    num_queries: int


def _dcg_at_k(binary_rels: list[int]) -> float:
    gains = np.asarray(binary_rels, dtype=np.float32)
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2, dtype=np.float32))
    return float(np.sum(gains / discounts))


def evaluate_backend(artifacts_dir: str, k: int = 10, min_department_size: int = 2) -> EvalResult:
    recommender = ExhibitionRecommender.from_artifacts(artifacts_dir)
    meta = recommender.metadata.copy()
    meta["department"] = meta["department"].fillna("").astype(str)
    meta["artist"] = meta["artist"].fillna("").astype(str)
    meta["objectID"] = meta["objectID"].astype("Int64")

    dept_counts = meta["department"].value_counts()
    candidate_departments = {
        dept
        for dept, count in dept_counts.items()
        if dept.strip() and int(count) >= int(min_department_size)
    }

    recalls: list[float] = []
    ndcgs: list[float] = []
    all_artist_hits: set[str] = set()
    all_department_hits: set[str] = set()

    for department in sorted(candidate_departments):
        relevant_ids = set(
            int(v)
            for v in meta.loc[meta["department"] == department, "objectID"].dropna().tolist()
        )
        if not relevant_ids:
            continue
        recs = recommender.recommend_for_theme(
            department,
            n_recommendations=k,
            min_score=0.0,
            max_per_artist=max(2, k),
            max_per_department=max(3, k),
        )
        if recs.empty:
            recalls.append(0.0)
            ndcgs.append(0.0)
            continue

        predicted_ids = [int(v) for v in recs["object_id"].dropna().tolist()]
        hits = [1 if obj_id in relevant_ids else 0 for obj_id in predicted_ids[:k]]
        hit_count = int(np.sum(hits))
        denom = max(1, min(len(relevant_ids), k))
        recalls.append(float(hit_count / denom))

        ideal_hits = [1] * min(len(relevant_ids), k)
        dcg = _dcg_at_k(hits)
        idcg = _dcg_at_k(ideal_hits)
        ndcgs.append(float(dcg / idcg) if idcg > 0 else 0.0)

        for val in recs["artist"].dropna().astype(str).tolist():
            if val.strip():
                all_artist_hits.add(val.strip().lower())
        for val in recs["department"].dropna().astype(str).tolist():
            if val.strip():
                all_department_hits.add(val.strip().lower())

    all_artists = {
        val.strip().lower()
        for val in meta["artist"].dropna().astype(str).tolist()
        if val.strip()
    }
    all_departments = {
        val.strip().lower()
        for val in meta["department"].dropna().astype(str).tolist()
        if val.strip()
    }

    backend = "tfidf"
    backend_meta = Path(artifacts_dir) / "embedding_backend.json"
    if backend_meta.exists():
        payload = json.loads(backend_meta.read_text(encoding="utf-8"))
        backend = str(payload.get("backend", "tfidf")).strip().lower()

    return EvalResult(
        backend=backend,
        artifacts_dir=artifacts_dir,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        artist_coverage=float(len(all_artist_hits) / max(1, len(all_artists))),
        department_coverage=float(len(all_department_hits) / max(1, len(all_departments))),
        num_queries=len(recalls),
    )


def _print_table(results: list[EvalResult], k: int) -> None:
    print()
    print(
        f"| Backend | Recall@{k} | NDCG@{k} | Artist Coverage | Department Coverage | Notes |"
    )
    print("|---|---:|---:|---:|---:|---|")
    for result in results:
        notes = f"queries={result.num_queries}; artifacts={result.artifacts_dir}"
        print(
            f"| {result.backend.upper()} | "
            f"{result.recall_at_k:.4f} | "
            f"{result.ndcg_at_k:.4f} | "
            f"{result.artist_coverage:.4f} | "
            f"{result.department_coverage:.4f} | "
            f"{notes} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TF-IDF vs CLIP artifacts with Recall/NDCG/Coverage metrics.",
    )
    parser.add_argument(
        "--artifacts",
        nargs="+",
        required=True,
        help="One or more artifact directories (e.g., artifacts_tfidf artifacts_clip).",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-department-size", type=int, default=2)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    results = [
        evaluate_backend(path, k=args.k, min_department_size=args.min_department_size)
        for path in args.artifacts
    ]
    _print_table(results, k=args.k)

    if args.json_out:
        payload = [
            {
                "backend": item.backend,
                "artifacts_dir": item.artifacts_dir,
                f"recall@{args.k}": item.recall_at_k,
                f"ndcg@{args.k}": item.ndcg_at_k,
                "artist_coverage": item.artist_coverage,
                "department_coverage": item.department_coverage,
                "num_queries": item.num_queries,
            }
            for item in results
        ]
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"\nSaved JSON metrics to: {args.json_out}")


if __name__ == "__main__":
    main()
