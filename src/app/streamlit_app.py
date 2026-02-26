from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure `src` imports resolve when Streamlit is launched outside repo-root PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

    try:
        recommender = ExhibitionRecommender.from_artifacts(settings.artifacts_dir)
    except Exception as exc:
        return None, f"Failed to load artifacts: {exc}", status.warning

    if getattr(recommender, "embedding_backend", "") == "clip":
        try:
            # Warm CLIP model on startup to avoid first-query UI stalls.
            recommender._get_clip_encoder()
        except Exception as exc:
            return (
                None,
                "CLIP initialization failed. Install `open_clip_torch` and `torch`, "
                f"or rebuild artifacts with TF-IDF backend. Details: {exc}",
                status.warning,
            )

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
    # Normalize separators so Windows-style paths work on Mac/Linux too
    candidate = Path(raw.replace("\\", "/"))
    if not candidate.is_absolute():
        images_base = Path(SETTINGS.images_dir)
        first = candidate.parts[0].lower() if candidate.parts else ""
        if first == "images":
            candidate = images_base.parent / candidate
        elif len(candidate.parts) > 1:
            # project-root-relative path, e.g. "data/raw/images/398746.jpg"
            candidate = Path.cwd() / candidate
        else:
            candidate = images_base / candidate

    return str(candidate) if candidate.exists() else None


st.set_page_config(page_title="MET Exhibition AI Curator", layout="wide")
col_text, col_logo = st.columns([9, 1])

with col_text:
    st.title("MET Exhibition AI Curator")


with col_logo:
    st.image("https://pdr-assets.b-cdn.net/sources/the-met.png?height=1200", use_container_width=True)
st.markdown("*Intelligent artwork recommendations for themed exhibitions*")
st.divider()

#st.title("MET Exhibition AI Curator")
#st.write("Choose themes and generate grouped exhibition recommendations.")

recommender, bootstrap_error, bootstrap_warning = load_recommender()
if bootstrap_warning:
    st.warning(bootstrap_warning)
if bootstrap_error:
    st.error(bootstrap_error)
    st.stop()
if recommender is None:
    st.warning("Artifacts not found and could not be generated.")
    st.stop()
assert recommender is not None


with st.sidebar:
    st.header("Exhibition Setup")
    
    themes_input = st.text_area(
        "Theme(s) (comma-separated)",
        value="Ancient Egypt, Religious Art, Portraits",
    )
    pieces = st.slider("Target pieces per exhibition", 5, 10, 8)
    min_similarity = st.slider("Minimum similarity", 0.0, 1.0, 0.2, 0.05)
    colors_input = st.text_input("Colors (optional)", value="")
    styles_input = st.text_input("Styles (optional)", value="")
    #year_min = st.number_input("Year min (optional, 0=off)", value=0, step=1)
    #year_max = st.number_input("Year max (optional, 0=off)", value=0, step=1)
    
    show_diagnostics = st.toggle("Show Selection Diagnostics", value=True)
    
    generate = st.button("Generate", type="primary")

    st.divider()
    st.caption(
        "This recommender works best when your theme uses attributes represented in the collection "
        "(period, material, style, subject, color, culture, or department). If results are weak, "
        "refine your prompt with concrete descriptors that combine what it is, when, and how it looks."
    )

if generate:
    try:
        themes = [t.strip() for t in themes_input.split(",") if t.strip()]
        if not (1 <= len(themes) <= 7):
            st.error("Please enter between 1 and 7 themes.")
            st.stop()

        colors = [c.strip().lower() for c in colors_input.split(",") if c.strip()]
        styles = [s.strip().lower() for s in styles_input.split(",") if s.strip()]
        #y_min = None if year_min == 0 else int(year_min)
        #y_max = None if year_max == 0 else int(year_max)

        with st.spinner("Generating recommendations..."):
            used_ids: set[int] = set()
            for theme in themes:
                frame = recommender.recommend_for_theme(
                    theme,
                    n_recommendations=pieces,
                    exclude_ids=used_ids,
                    min_score=min_similarity,
                )
                if frame.empty:
                    frame = recommender.recommend_for_theme(
                        theme,
                        n_recommendations=pieces,
                        exclude_ids=used_ids,
                        min_score=0.0,
                    )
                frame = score_with_filters(frame, colors, styles, 0, 0)

                st.subheader(f"{theme} Exhibition")


                if frame.empty:
                    st.error("No similar pieces of art found for this theme.")
                    continue
                if frame.empty:
                    st.error("No similar pieces of art found for this theme.")
                    continue
                if frame["score"].max() < min_similarity:
                    st.warning("Showing best available matches below the selected minimum similarity.")
                
                # Create the expander with a clear label (e.g., the Theme Name)
                with st.expander(f"View Exhibition Details for: {theme}", expanded=False):
                    used_ids.update(int(v) for v in frame["object_id"].tolist())
                    
                    # Create 4 columns inside the expander
                    cols = st.columns(4)
                    
                    for col_idx, (_, row) in enumerate(frame.iterrows()):
                        # Determine which column to place the current piece in
                        with cols[col_idx % len(cols)]:
                            img = image_path(row.get("image_path"))
                            if img:
                                # Display the artwork image
                                st.image(img, use_container_width=True)
                            
                            # Extract and display the score and metadata
                            shown_score = float(row.get("raw_score", row.get("score", 0.0)))
                            artist = row.get('artist') or 'Unknown'
                            if pd.isna(artist):
                                artist = 'Unknown'
                            st.caption(
                                f"**{row.get('title') or 'Untitled'},** \n"
                                f"**{artist}** |\n"
                                f"{row.get('object_date') or 'Unknown'}"
                            )

                    if show_diagnostics:
                        st.divider()
                        st.header("AI Selection Process & Ranking Insights")

                        st.write(f"**Theme Analysis:** Keywords and visual patterns identified for *'{theme}'*.")
                        
                        # Display insights of filtering process
                        diag_cols = st.columns(2)
                        with diag_cols[0]:
                            st.metric("Initial Candidates Found", "100")
                            st.caption("Retrieved via Cosine Similarity")
                        with diag_cols[1]:
                            st.metric("Final Selection", len(frame))
                            st.caption("Refined via XGBoost Ranker")

                        # Display scoring details
                        st.write("**Top Ranked Selection Insights:**")
                        diag_frame = frame[['title', 'raw_score', 'score']].copy()
                        diag_frame['Filter Bonus'] = (diag_frame['score'] - diag_frame['raw_score']).clip(lower=0.0)
                        diag_frame.columns = ['Artwork Title', 'AI Match (Raw)', 'Final Score', 'Filter Bonus']
                        diag_frame = diag_frame[['Artwork Title', 'AI Match (Raw)', 'Filter Bonus', 'Final Score']]
                        st.dataframe(
                            diag_frame.style.format({
                                'AI Match (Raw)': '{:.3f}',
                                'Filter Bonus': '+{:.3f}',
                                'Final Score': '{:.3f}'
                            }), 
                            use_container_width=True, 
                            hide_index=True
                        )


    except Exception as exc:
        st.error("Theme generation failed. See details below.")
        st.exception(exc)


# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>MET Exhibition AI Curator | INSY 674 Winter 2026</p>
        <p>Data source: Metropolitan Museum of Art Collection API</p>
    </div>
""", unsafe_allow_html=True)


#st.caption(
#    "Backend: "
#    f"`{getattr(recommender, 'embedding_backend', 'unknown')}`"
#    " | Artifacts: "
#    f"`{SETTINGS.artifacts_dir}`"
#    " | Embeddings shape: "
#    f"`{recommender.embeddings.shape}`"
#)



