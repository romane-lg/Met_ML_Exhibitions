"""
Data loading utilities for MET Exhibition Curator.

This module provides functions to load and validate the MET artwork dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def load_met_data(
    data_path: str = "data/raw/met_data.csv",
    validate: bool = True
) -> pd.DataFrame:
    """
    Load MET artwork metadata from CSV file.
    
    Parameters
    ----------
    data_path : str
        Path to the CSV file containing MET data
    validate : bool, optional
        Whether to validate the data after loading (default: True)
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing artwork metadata
        
    Raises
    ------
    FileNotFoundError
        If the data file doesn't exist
    ValueError
        If required columns are missing
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    if validate:
        df = validate_data(df)
    
    logger.info(f"Loaded {len(df)} artworks")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the MET data structure and content.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
        
    Returns
    -------
    pd.DataFrame
        Validated DataFrame
        
    Raises
    ------
    ValueError
        If required columns are missing or data is invalid
    """
    required_columns = ['objectID', 'title', 'image_path']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove duplicates based on objectID
    initial_len = len(df)
    df = df.drop_duplicates(subset='objectID', keep='first')
    
    if len(df) < initial_len:
        logger.warning(f"Removed {initial_len - len(df)} duplicate records")
    
    # Remove rows with missing critical values
    df = df.dropna(subset=['objectID', 'image_path'])
    
    return df


def get_image_path(
    object_id: int,
    images_dir: str = "data/raw/images"
) -> Optional[Path]:
    """
    Get the image path for a given object ID.
    
    Parameters
    ----------
    object_id : int
        The MET object ID
    images_dir : str
        Directory containing images
        
    Returns
    -------
    Optional[Path]
        Path to the image file, or None if not found
    """
    images_dir = Path(images_dir)
    image_path = images_dir / f"{object_id}.jpg"
    
    if image_path.exists():
        return image_path
    else:
        logger.warning(f"Image not found for object {object_id}")
        return None


def filter_by_department(
    df: pd.DataFrame,
    departments: List[str]
) -> pd.DataFrame:
    """
    Filter artworks by department.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing artwork data
    departments : List[str]
        List of department names to include
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    filtered = df[df['department'].isin(departments)]
    logger.info(f"Filtered to {len(filtered)} artworks from departments: {departments}")
    return filtered


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing artwork data
        
    Returns
    -------
    dict
        Dictionary containing summary statistics
    """
    summary = {
        'total_artworks': len(df),
        'departments': df['department'].nunique(),
        'department_counts': df['department'].value_counts().to_dict(),
        'artists': df['artist'].nunique(),
        'missing_titles': df['title'].isna().sum(),
        'missing_artists': df['artist'].isna().sum(),
        'date_range': (df['objectDate'].min(), df['objectDate'].max())
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    df = load_met_data()
    summary = get_data_summary(df)
    
    print("\nDataset Summary:")
    print(f"Total artworks: {summary['total_artworks']}")
    print(f"Number of departments: {summary['departments']}")
    print(f"\nTop 5 departments:")
    for dept, count in list(summary['department_counts'].items())[:5]:
        print(f"  {dept}: {count}")
