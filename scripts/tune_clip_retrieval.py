from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models import ExhibitionRecommender


@dataclass
class TuningResult:
    clip_weight: float
    lexical_weight: float
    prompt_ensemble: bool
    mean_keyword_hit_rate: float
    mean_score: float


DEFAULT_THEME_KEYWORDS: dict[str, list[str]] = {
    "portrait": ["portrait", "self-portrait", "face", "bust"],
    "christian": ["christian", "saint", "madonna", "cross", "crucifix", "biblical"],
    "ancient egypt": ["egypt", "egyptian", "pharaoh", "hieroglyph", "dynasty"],
    "religious art": ["religious", "saint", "altar", "devotional", "icon"],
}


def _hit_rate(frame: pd.DataFrame, keywords: list[str]) -> float:
    if frame.empty:
        return 0.0
    cols = ["title", "artist", "department", "medium"]
    docs = (
        frame[cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .tolist()
    )
    hits = [1.0 if any(word in doc for word in keywords) else 0.0 for doc in docs]
    return float(np.mean(hits)) if hits else 0.0


def evaluate_configs(
    recommender: ExhibitionRecommender,
    themes: list[str],
    top_k: int,
    clip_weights: list[float],
    prompt_modes: list[bool],
) -> list[TuningResult]:
    out: list[TuningResult] = []
    for cw in clip_weights:
        lw = 1.0 - cw
        for prompt_mode in prompt_modes:
            recommender.clip_similarity_weight = float(cw)
            recommender.clip_lexical_weight = float(lw)
            recommender.clip_prompt_ensemble = bool(prompt_mode)
            hit_rates: list[float] = []
            mean_scores: list[float] = []
            for theme in themes:
                keywords = DEFAULT_THEME_KEYWORDS.get(theme.lower(), theme.lower().split())
                recs = recommender.recommend_for_theme(theme, n_recommendations=top_k, min_score=0.0)
                hit_rates.append(_hit_rate(recs, keywords))
                if recs.empty:
                    mean_scores.append(0.0)
                else:
                    mean_scores.append(float(recs["score"].astype(float).mean()))
            out.append(
                TuningResult(
                    clip_weight=float(cw),
                    lexical_weight=float(lw),
                    prompt_ensemble=bool(prompt_mode),
                    mean_keyword_hit_rate=float(np.mean(hit_rates)) if hit_rates else 0.0,
                    mean_score=float(np.mean(mean_scores)) if mean_scores else 0.0,
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CLIP retrieval blend/prompt settings.")
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--themes", type=str, default="portrait,christian,ancient egypt,religious art")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--clip-weights", type=str, default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--include-no-prompt-ensemble", action="store_true")
    args = parser.parse_args()

    rec = ExhibitionRecommender.from_artifacts(args.artifacts)
    if rec.embedding_backend != "clip":
        raise RuntimeError(
            f"Artifacts at '{args.artifacts}' are backend='{rec.embedding_backend}', expected clip."
        )

    themes = [item.strip() for item in args.themes.split(",") if item.strip()]
    weights = [float(item.strip()) for item in args.clip_weights.split(",") if item.strip()]
    prompt_modes = [True, False] if args.include_no_prompt_ensemble else [True]

    results = evaluate_configs(rec, themes, args.top_k, weights, prompt_modes)
    results = sorted(
        results,
        key=lambda x: (x.mean_keyword_hit_rate, x.mean_score),
        reverse=True,
    )
    print("| clip_weight | lexical_weight | prompt_ensemble | keyword_hit_rate | mean_score |")
    print("|---:|---:|:---:|---:|---:|")
    for row in results:
        print(
            f"| {row.clip_weight:.2f} | {row.lexical_weight:.2f} | "
            f"{'yes' if row.prompt_ensemble else 'no'} | "
            f"{row.mean_keyword_hit_rate:.4f} | {row.mean_score:.4f} |"
        )


if __name__ == "__main__":
    main()
