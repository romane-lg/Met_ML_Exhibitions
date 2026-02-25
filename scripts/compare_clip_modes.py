from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models import ExhibitionRecommender


DEFAULT_THEME_KEYWORDS: dict[str, list[str]] = {
    "portrait": ["portrait", "self-portrait", "face", "bust"],
    "portraits": ["portrait", "self-portrait", "face", "bust"],
    "christian": ["christian", "saint", "madonna", "cross", "crucifix", "biblical"],
    "ancient egypt": ["egypt", "egyptian", "pharaoh", "hieroglyph", "dynasty"],
    "religious art": ["religious", "saint", "altar", "devotional", "icon"],
}


@dataclass
class ModeConfig:
    name: str
    clip_weight: float
    lexical_weight: float
    prompt_ensemble: bool


def _keywords_for_theme(theme: str) -> list[str]:
    return DEFAULT_THEME_KEYWORDS.get(theme.lower(), theme.lower().split())


def _hit_rate(frame: pd.DataFrame, keywords: list[str]) -> float:
    if frame.empty:
        return 0.0
    docs = (
        frame[["title", "artist", "department", "medium"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .tolist()
    )
    hits = [1.0 if any(word in doc for word in keywords) else 0.0 for doc in docs]
    return float(np.mean(hits)) if hits else 0.0


def _evaluate_mode(
    rec: ExhibitionRecommender,
    mode: ModeConfig,
    themes: list[str],
    top_k: int,
) -> pd.DataFrame:
    rec.clip_similarity_weight = mode.clip_weight
    rec.clip_lexical_weight = mode.lexical_weight
    rec.clip_prompt_ensemble = mode.prompt_ensemble

    rows: list[dict[str, object]] = []
    for theme in themes:
        recs = rec.recommend_for_theme(theme, n_recommendations=top_k, min_score=0.0)
        keywords = _keywords_for_theme(theme)
        hit_rate = _hit_rate(recs, keywords)
        top_titles = recs["title"].fillna("").astype(str).head(3).tolist() if not recs.empty else []
        top_scores = recs["score"].astype(float).head(3).tolist() if not recs.empty else []
        rows.append(
            {
                "mode": mode.name,
                "theme": theme,
                "keyword_hit_rate": hit_rate,
                "mean_score": float(recs["score"].astype(float).mean()) if not recs.empty else 0.0,
                "top1": top_titles[0] if len(top_titles) > 0 else "",
                "top2": top_titles[1] if len(top_titles) > 1 else "",
                "top3": top_titles[2] if len(top_titles) > 2 else "",
                "top1_score": float(top_scores[0]) if len(top_scores) > 0 else 0.0,
                "top2_score": float(top_scores[1]) if len(top_scores) > 1 else 0.0,
                "top3_score": float(top_scores[2]) if len(top_scores) > 2 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline CLIP retrieval vs improved retrieval side-by-side."
    )
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--themes", type=str, default="portrait,christian,ancient egypt,religious art")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--csv-out", type=str, default="")
    args = parser.parse_args()

    rec = ExhibitionRecommender.from_artifacts(args.artifacts)
    if rec.embedding_backend != "clip":
        raise RuntimeError(
            f"Artifacts at '{args.artifacts}' are backend='{rec.embedding_backend}', expected clip."
        )
    themes = [item.strip() for item in args.themes.split(",") if item.strip()]

    baseline = ModeConfig(name="baseline", clip_weight=1.0, lexical_weight=0.0, prompt_ensemble=False)
    improved = ModeConfig(
        name="improved",
        clip_weight=rec.clip_similarity_weight,
        lexical_weight=rec.clip_lexical_weight,
        prompt_ensemble=rec.clip_prompt_ensemble,
    )

    df_base = _evaluate_mode(rec, baseline, themes, args.top_k)
    df_improved = _evaluate_mode(rec, improved, themes, args.top_k)
    merged = df_base.merge(df_improved, on="theme", suffixes=("_baseline", "_improved"))

    print("| theme | hit_rate_base | hit_rate_improved | top1_base | top1_improved |")
    print("|---|---:|---:|---|---|")
    for _, row in merged.iterrows():
        print(
            f"| {row['theme']} | {float(row['keyword_hit_rate_baseline']):.3f} | "
            f"{float(row['keyword_hit_rate_improved']):.3f} | "
            f"{row['top1_baseline']} | {row['top1_improved']} |"
        )

    mean_base = float(merged["keyword_hit_rate_baseline"].mean()) if not merged.empty else 0.0
    mean_improved = float(merged["keyword_hit_rate_improved"].mean()) if not merged.empty else 0.0
    print()
    print(f"Mean keyword hit rate: baseline={mean_base:.4f} improved={mean_improved:.4f}")

    if args.csv_out:
        merged.to_csv(args.csv_out, index=False)
        print(f"Saved comparison CSV to: {args.csv_out}")


if __name__ == "__main__":
    main()
