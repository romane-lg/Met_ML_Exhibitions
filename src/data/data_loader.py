"""Data loading utilities for MET Exhibition Curator."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_met_data(
    data_path: str | Path = "data/raw/met_data.csv",
    validate: bool = True,
) -> pd.DataFrame:
    """Load MET artwork metadata from CSV file."""
    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    if validate:
        df = validate_data(df)

    logger.info("Loaded %d artworks", len(df))
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the MET data structure and content."""
    required_columns = ["objectID", "title", "image_path"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    initial_len = len(df)
    df = df.drop_duplicates(subset="objectID", keep="first")
    if len(df) < initial_len:
        logger.warning("Removed %d duplicate records", initial_len - len(df))

    return df.dropna(subset=["objectID", "image_path"])


def get_image_path(object_id: int, images_dir: str | Path = "data/raw/images") -> Path | None:
    """Get the image path for a given object ID."""
    images_base = Path(images_dir)
    image_path = images_base / f"{object_id}.jpg"
    if image_path.exists():
        return image_path

    logger.warning("Image not found for object %s", object_id)
    return None


def filter_by_department(df: pd.DataFrame, departments: list[str]) -> pd.DataFrame:
    """Filter artworks by department."""
    filtered = df[df["department"].isin(departments)]
    logger.info("Filtered to %d artworks from departments: %s", len(filtered), departments)
    return filtered


def get_data_summary(df: pd.DataFrame) -> dict[str, object]:
    """Get summary statistics of the dataset."""
    return {
        "total_artworks": len(df),
        "departments": df["department"].nunique(),
        "department_counts": df["department"].value_counts().to_dict(),
        "artists": df["artist"].nunique(),
        "missing_titles": int(df["title"].isna().sum()),
        "missing_artists": int(df["artist"].isna().sum()),
        "date_range": (df["objectDate"].min(), df["objectDate"].max()),
    }

