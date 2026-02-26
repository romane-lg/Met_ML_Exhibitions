"""
Test data loading utilities.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import load_met_data, validate_data, get_data_summary


def test_load_met_data():
    """Test that data loads correctly."""
    # This would need actual test data
    # For now, just test the structure
    pass


def test_validate_data():
    """Test data validation."""
    # Create sample data
    test_data = pd.DataFrame({
        'objectID': [1, 2, 2, 3],  # Has duplicate
        'title': ['Art 1', 'Art 2', 'Art 2', None],  # Has missing
        'image_path': ['img1.jpg', 'img2.jpg', 'img2.jpg', 'img3.jpg']
    })
    
    validated = validate_data(test_data)
    
    # Should remove duplicate
    assert len(validated) == 3
    
    # Should keep only valid records
    assert validated['objectID'].tolist() == [1, 2, 3]


def test_get_data_summary():
    """Test summary statistics."""
    test_data = pd.DataFrame({
        'objectID': [1, 2, 3],
        'title': ['Art 1', 'Art 2', 'Art 3'],
        'artist': ['Artist A', 'Artist B', None],
        'department': ['Dept 1', 'Dept 1', 'Dept 2'],
        'objectDate': ['2020', '2021', '2022'],
        'medium': ['Oil', 'Watercolor', 'Sculpture'],
        'image_path': ['img1.jpg', 'img2.jpg', 'img3.jpg']
    })
    
    summary = get_data_summary(test_data)
    
    assert summary['total_artworks'] == 3
    assert summary['departments'] == 2
    assert summary['artists'] == 2
    assert summary['missing_artists'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
