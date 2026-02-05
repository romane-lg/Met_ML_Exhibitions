"""
Test recommendation system.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import ExhibitionRecommender


def test_recommender_initialization():
    """Test recommender initialization."""
    # Create sample data
    n_artworks = 50
    n_features = 20
    
    features = np.random.rand(n_artworks, n_features)
    metadata = pd.DataFrame({
        'objectID': range(n_artworks),
        'title': [f'Artwork {i}' for i in range(n_artworks)],
        'department': ['Dept A'] * 25 + ['Dept B'] * 25
    })
    
    recommender = ExhibitionRecommender(features, metadata)
    
    assert recommender.features.shape == (n_artworks, n_features)
    assert len(recommender.metadata) == n_artworks
    assert recommender.similarity_matrix.shape == (n_artworks, n_artworks)


def test_recommend_for_theme():
    """Test theme-based recommendations."""
    n_artworks = 50
    features = np.random.rand(n_artworks, 20)
    
    metadata = pd.DataFrame({
        'objectID': range(n_artworks),
        'title': [f'Artwork {i}' for i in range(n_artworks)],
        'department': ['Egyptian Art'] * 10 + ['European Paintings'] * 40
    })
    
    recommender = ExhibitionRecommender(features, metadata)
    
    # Request Egyptian theme
    results = recommender.recommend_for_theme('egyptian', n_recommendations=15)
    
    assert len(results) <= 15
    assert 'similarity_score' in results.columns


def test_coherence_evaluation():
    """Test coherence score calculation."""
    n_artworks = 20
    features = np.random.rand(n_artworks, 10)
    
    metadata = pd.DataFrame({
        'objectID': range(n_artworks),
        'title': [f'Art {i}' for i in range(n_artworks)]
    })
    
    recommender = ExhibitionRecommender(features, metadata)
    
    # Test with a subset
    artwork_ids = [0, 1, 2, 3, 4]
    coherence = recommender.evaluate_coherence(artwork_ids)
    
    assert 0 <= coherence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
