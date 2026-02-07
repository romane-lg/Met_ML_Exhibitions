from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.bootstrap import ensure_artifacts
from src.config import get_settings
from src.models import ExhibitionRecommender

STOPWORDS = {
    "the",
    "and",
    "of",
    "in",
    "a",
    "an",
    "to",
    "for",
    "with",
    "art",
    "arts",
    "from",
}
SETTINGS = get_settings()


@st.cache_resource
def load_recommender() -> tuple[ExhibitionRecommender | None, str | None, str | None]:
    settings = get_settings()
    status = ensure_artifacts(settings)
    if not status.ready:
        return None, status.error, status.warning
    recommender = ExhibitionRecommender.from_artifacts(settings.artifacts_dir)
    return recommender, status.error, status.warning


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.replace("/", " ").replace(";", " ").split() if t and t not in STOPWORDS]


def extract_year(value: str | None) -> int | None:
    if not value:
        return None
    digits = "".join(ch if ch.isdigit() else " " for ch in str(value)).split()
    for token in digits:
        if len(token) == 4:
            return int(token)
    return None


def score_with_filters(
    frame: pd.DataFrame,
    colors: list[str],
    styles: list[str],
    year_min: int | None,
    year_max: int | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    combo = (
        frame[["title", "artist", "department", "medium", "object_date"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    modifier = pd.Series(0.0, index=frame.index)
    if colors:
        modifier += combo.apply(lambda x: sum(c in x for c in colors) * 0.03)
    if styles:
        modifier += combo.apply(lambda x: sum(s in x for s in styles) * 0.03)

    if year_min is not None or year_max is not None:
        years = frame["object_date"].apply(extract_year)
        valid = pd.Series(True, index=frame.index)
        if year_min is not None:
            valid &= years.fillna(-9999) >= year_min
        if year_max is not None:
            valid &= years.fillna(9999) <= year_max
        modifier += valid.astype(float) * 0.05

    frame = frame.copy()
    frame["score"] = (frame["score"].astype(float) + modifier).clip(upper=1.0)
    return frame.sort_values("score", ascending=False)


def image_path(raw: str | None) -> str | None:
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        images_base = Path(SETTINGS.images_dir)
        if candidate.parts and candidate.parts[0].lower() == "images":
            candidate = images_base.parent / candidate
        else:
            candidate = images_base / candidate
    return str(candidate) if candidate.exists() else None


st.set_page_config(page_title="MET Exhibition AI Curator", layout="wide")
st.title("MET Exhibition AI Curator")
st.write("Choose themes and generate grouped exhibition recommendations.")

recommender, bootstrap_error, bootstrap_warning = load_recommender()
if bootstrap_warning:
    st.warning(bootstrap_warning)
if bootstrap_error:
    st.error(bootstrap_error)
    st.stop()
if recommender is None:
    st.warning("Artifacts not found and could not be generated.")
    st.stop()

with st.sidebar:
    st.header("Exhibition Setup")
    st.caption(
        "This recommender works best when your theme uses attributes represented in the collection "
        "(period, material, style, subject, color, culture, or department). If results are weak, "
        "refine your prompt with concrete descriptors that combine what it is, when, and how it looks."
    )
    themes_input = st.text_area(
        "Themes (comma-separated)",
        value="ancient egypt, religious art, portraits",
    )
    pieces = st.slider("Pieces per exhibition", 5, 10, 8)
    min_similarity = st.slider("Minimum similarity", 0.0, 1.0, 0.2, 0.05)
    colors_input = st.text_input("Colors (optional)", value="")
    styles_input = st.text_input("Styles (optional)", value="")
    year_min = st.number_input("Year min (optional, 0=off)", value=0, step=1)
    year_max = st.number_input("Year max (optional, 0=off)", value=0, step=1)
    generate = st.button("Generate Exhibitions")

if generate:
    themes = [t.strip() for t in themes_input.split(",") if t.strip()]
    if not (1 <= len(themes) <= 7):
        st.error("Please enter between 1 and 7 themes.")
        st.stop()

    colors = [c.strip().lower() for c in colors_input.split(",") if c.strip()]
    styles = [s.strip().lower() for s in styles_input.split(",") if s.strip()]
    y_min = None if year_min == 0 else int(year_min)
    y_max = None if year_max == 0 else int(year_max)

    used_ids: set[int] = set()
    for theme in themes:
        frame = recommender.recommend_for_theme(
            theme,
            n_recommendations=pieces,
            exclude_ids=used_ids,
            min_score=min_similarity,
        )
        frame = score_with_filters(frame, colors, styles, y_min, y_max)

        st.subheader(f"Theme: {theme}")
        if frame.empty or frame["score"].max() < min_similarity:
            st.error("No similar pieces of art found for this theme.")
            continue

        used_ids.update(int(v) for v in frame["object_id"].tolist())
        cols = st.columns(4)
        for idx, row in frame.iterrows():
            with cols[idx % 4]:
                img = image_path(row.get("image_path"))
                if img:
                    st.image(img, use_container_width=True)
                st.caption(
                    f"{row.get('title') or 'Untitled'} | {row.get('artist') or 'Unknown'}"
                    f" | score={float(row.get('score', 0.0)):.3f}"
                )
