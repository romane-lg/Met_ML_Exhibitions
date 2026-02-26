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


def _theme_level_table(
    rec: ExhibitionRecommender,
    top_k: int,
    min_department_size: int,
) -> pd.DataFrame:
    meta = rec.metadata.copy()
    meta["department"] = meta["department"].fillna("").astype(str)
    meta["objectID"] = meta["objectID"].astype("Int64")
    dept_counts = meta["department"].value_counts()
    themes = [
        dept
        for dept, count in dept_counts.items()
        if dept.strip() and int(count) >= int(min_department_size)
    ]
    rows: list[dict[str, Any]] = []
    for theme in themes:
        relevant = set(int(v) for v in meta.loc[meta["department"] == theme, "objectID"].dropna().tolist())
        frame = rec.recommend_for_theme(theme, n_recommendations=top_k, min_score=0.0)
        if frame.empty:
            rows.append(
                {
                    "theme": theme,
                    "recall_at_k": 0.0,
                    "ndcg_at_k": 0.0,
                    "mean_score": 0.0,
                    "top1_id": None,
                    "top1_title": "",
                }
            )
            continue
        predicted = [int(v) for v in frame["object_id"].dropna().tolist()[:top_k]]
        hits = [1 if item in relevant else 0 for item in predicted]
        recall = float(sum(hits) / max(1, min(len(relevant), top_k)))
        ideal = [1] * min(len(relevant), top_k)
        idcg = _dcg(ideal)
        ndcg = float(_dcg(hits) / idcg) if idcg > 0 else 0.0
        top1_id = int(frame.iloc[0]["object_id"]) if len(frame) > 0 else None
        top1_title = str(frame.iloc[0].get("title", "") or "") if len(frame) > 0 else ""
        rows.append(
            {
                "theme": theme,
                "recall_at_k": recall,
                "ndcg_at_k": ndcg,
                "mean_score": float(frame["score"].astype(float).mean()),
                "top1_id": top1_id,
                "top1_title": top1_title,
            }
        )
    return pd.DataFrame(rows)


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


def _paired_bootstrap_stats(
    baseline: np.ndarray,
    candidate: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    if baseline.size == 0 or candidate.size == 0 or baseline.size != candidate.size:
        return 0.0, 0.0, 0.0, 1.0
    diffs = candidate - baseline
    obs = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    n = len(diffs)
    for _ in range(max(1, n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(diffs[idx])))
    arr = np.asarray(boots, dtype=np.float32)
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    p_left = float(np.mean(arr <= 0.0))
    p_right = float(np.mean(arr >= 0.0))
    p_two_sided = float(min(1.0, 2.0 * min(p_left, p_right)))
    return obs, ci_low, ci_high, p_two_sided


def _run_clip_ablations(
    rec: ExhibitionRecommender,
    top_k: int,
    min_department_size: int,
) -> list[dict[str, Any]]:
    if rec.embedding_backend != "clip":
        return []
    original = (rec.clip_similarity_weight, rec.clip_lexical_weight, rec.clip_prompt_ensemble)
    configs = [
        ("current", rec.clip_similarity_weight, rec.clip_lexical_weight, rec.clip_prompt_ensemble),
        ("clip_only_no_prompt", 1.0, 0.0, False),
        ("clip_only_prompt", 1.0, 0.0, True),
        ("hybrid_no_prompt", rec.clip_similarity_weight, rec.clip_lexical_weight, False),
    ]
    rows: list[dict[str, Any]] = []
    try:
        for name, cw, lw, pe in configs:
            rec.clip_similarity_weight = float(cw)
            rec.clip_lexical_weight = float(lw)
            rec.clip_prompt_ensemble = bool(pe)
            table = _theme_level_table(rec, top_k=top_k, min_department_size=min_department_size)
            rows.append(
                {
                    "mode": name,
                    "clip_weight": float(cw),
                    "lexical_weight": float(lw),
                    "prompt_ensemble": bool(pe),
                    "recall_at_k": float(table["recall_at_k"].mean()) if not table.empty else 0.0,
                    "ndcg_at_k": float(table["ndcg_at_k"].mean()) if not table.empty else 0.0,
                }
            )
    finally:
        rec.clip_similarity_weight, rec.clip_lexical_weight, rec.clip_prompt_ensemble = original
    return rows


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
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--per-theme-top-errors", type=int, default=5)
    parser.add_argument("--per-theme-out", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--csv-out", type=str, default="")
    args = parser.parse_args()

    reports: list[Report] = []
    theme_tables: dict[str, pd.DataFrame] = {}
    ablation_rows: list[dict[str, Any]] = []
    for path in args.artifacts:
        reports.append(
            evaluate(
                path,
                top_k=args.top_k,
                min_department_size=args.min_department_size,
                latency_runs=args.latency_runs,
            )
        )
        rec = ExhibitionRecommender.from_artifacts(path)
        table = _theme_level_table(rec, top_k=args.top_k, min_department_size=args.min_department_size)
        table.insert(0, "artifacts_dir", path)
        theme_tables[path] = table
        if args.run_ablations:
            for row in _run_clip_ablations(rec, top_k=args.top_k, min_department_size=args.min_department_size):
                row["artifacts_dir"] = path
                ablation_rows.append(row)

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

    for path, table in theme_tables.items():
        if table.empty:
            continue
        worst = table.sort_values("ndcg_at_k", ascending=True).head(max(1, args.per_theme_top_errors))
        print()
        print(f"Worst themes for {path}:")
        print("| theme | recall@k | ndcg@k | top1_id | top1_title |")
        print("|---|---:|---:|---:|---|")
        for _, row in worst.iterrows():
            print(
                f"| {row['theme']} | {float(row['recall_at_k']):.4f} | "
                f"{float(row['ndcg_at_k']):.4f} | {row['top1_id']} | {row['top1_title']} |"
            )

    if len(args.artifacts) > 1:
        baseline_path = args.artifacts[0]
        baseline = theme_tables.get(baseline_path, pd.DataFrame())
        for cand_path in args.artifacts[1:]:
            candidate = theme_tables.get(cand_path, pd.DataFrame())
            if baseline.empty or candidate.empty:
                continue
            merged = baseline[["theme", "ndcg_at_k"]].merge(
                candidate[["theme", "ndcg_at_k"]],
                on="theme",
                suffixes=("_base", "_cand"),
            )
            obs, low, high, pval = _paired_bootstrap_stats(
                baseline=merged["ndcg_at_k_base"].to_numpy(dtype=np.float32),
                candidate=merged["ndcg_at_k_cand"].to_numpy(dtype=np.float32),
                n_bootstrap=args.bootstrap_samples,
            )
            print()
            print(
                "NDCG significance "
                f"{cand_path} vs {baseline_path}: diff={obs:+.4f}, "
                f"95% CI [{low:+.4f}, {high:+.4f}], p≈{pval:.4f}"
            )

    if ablation_rows:
        ablation = pd.DataFrame(ablation_rows)
        print()
        print("Ablations:")
        print("| artifacts | mode | clip_w | lex_w | prompt | recall@k | ndcg@k |")
        print("|---|---|---:|---:|:---:|---:|---:|")
        for _, row in ablation.sort_values(["artifacts_dir", "ndcg_at_k"], ascending=[True, False]).iterrows():
            print(
                f"| {row['artifacts_dir']} | {row['mode']} | {float(row['clip_weight']):.2f} | "
                f"{float(row['lexical_weight']):.2f} | {'yes' if bool(row['prompt_ensemble']) else 'no'} | "
                f"{float(row['recall_at_k']):.4f} | {float(row['ndcg_at_k']):.4f} |"
            )

    if args.json_out:
        payload = [_to_dict(rep) for rep in reports]
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Saved JSON: {args.json_out}")
    if args.csv_out:
        pd.DataFrame([_to_dict(rep) for rep in reports]).to_csv(args.csv_out, index=False)
        print(f"Saved CSV: {args.csv_out}")
    if args.per_theme_out:
        all_themes = pd.concat(list(theme_tables.values()), ignore_index=True) if theme_tables else pd.DataFrame()
        all_themes.to_csv(args.per_theme_out, index=False)
        print(f"Saved per-theme diagnostics: {args.per_theme_out}")


if __name__ == "__main__":
    main()
