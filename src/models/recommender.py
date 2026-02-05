"""
Recommendation system for exhibition curation.

This module provides the core recommendation engine that suggests artwork
groupings based on thematic queries.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
import logging

logger = logging.getLogger(__name__)


class ExhibitionRecommender:
    """Recommend artworks for themed exhibitions."""
    
    def __init__(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame,
        similarity_metric: str = 'cosine',
        diversity_weight: float = 0.3
    ):
        """
        Initialize the recommender.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_artworks, n_features)
        metadata : pd.DataFrame
            Artwork metadata
        similarity_metric : str
            'cosine', 'euclidean', or 'manhattan'
        diversity_weight : float
            Weight for diversity in recommendations (0-1)
        """
        self.features = features
        self.metadata = metadata
        self.similarity_metric = similarity_metric
        self.diversity_weight = diversity_weight
        
        # Precompute similarity matrix
        self.similarity_matrix = self._compute_similarity_matrix()
        
        logger.info(f"Initialized recommender with {len(metadata)} artworks")
    
    def _compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise similarity matrix.
        
        Returns
        -------
        np.ndarray
            Similarity matrix (n_artworks, n_artworks)
        """
        if self.similarity_metric == 'cosine':
            similarity = cosine_similarity(self.features)
        elif self.similarity_metric == 'euclidean':
            distances = euclidean_distances(self.features)
            # Convert distances to similarities
            similarity = 1 / (1 + distances)
        else:
            similarity = cosine_similarity(self.features)
        
        logger.info("Computed similarity matrix")
        return similarity
    
    def recommend_for_theme(
        self,
        theme_query: str,
        n_recommendations: int = 30,
        exclude_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Recommend artworks for a theme.
        
        Parameters
        ----------
        theme_query : str
            Theme description (e.g., "ancient egypt", "portraits")
        n_recommendations : int
            Number of artworks to recommend
        exclude_ids : List[int], optional
            Object IDs to exclude from recommendations
            
        Returns
        -------
        pd.DataFrame
            Recommended artworks with scores
        """
        # Simple keyword matching for MVP
        # TODO: Implement semantic search with embeddings
        query_lower = theme_query.lower()
        
        # Score based on text matching
        scores = []
        for idx, row in self.metadata.iterrows():
            score = 0
            
            # Check title
            if pd.notna(row.get('title')):
                if query_lower in str(row['title']).lower():
                    score += 0.5
            
            # Check department
            if pd.notna(row.get('department')):
                if query_lower in str(row['department']).lower():
                    score += 0.3
            
            # Check medium
            if pd.notna(row.get('medium')):
                if query_lower in str(row['medium']).lower():
                    score += 0.2
            
            scores.append(score)
        
        # Get seed artworks (those that match query)
        seed_indices = np.argsort(scores)[-5:]  # Top 5 matches as seeds
        
        # Expand recommendations using similarity
        recommendations = self._expand_recommendations(
            seed_indices,
            n_recommendations,
            exclude_ids
        )
        
        return recommendations
    
    def _expand_recommendations(
        self,
        seed_indices: np.ndarray,
        n_recommendations: int,
        exclude_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Expand recommendations from seed artworks.
        
        Parameters
        ----------
        seed_indices : np.ndarray
            Indices of seed artworks
        n_recommendations : int
            Number of recommendations
        exclude_ids : List[int], optional
            Object IDs to exclude
            
        Returns
        -------
        pd.DataFrame
            Recommended artworks
        """
        # Average similarity to seed artworks
        seed_similarities = self.similarity_matrix[seed_indices].mean(axis=0)
        
        # Apply diversity penalty
        # TODO: Implement diversity-aware selection
        
        # Create exclude mask
        exclude_mask = np.ones(len(self.metadata), dtype=bool)
        if exclude_ids:
            exclude_indices = self.metadata[
                self.metadata['objectID'].isin(exclude_ids)
            ].index
            exclude_mask[exclude_indices] = False
        
        # Apply mask
        masked_scores = seed_similarities.copy()
        masked_scores[~exclude_mask] = -np.inf
        
        # Get top recommendations
        top_indices = np.argsort(masked_scores)[-n_recommendations:][::-1]
        
        # Create results DataFrame
        results = self.metadata.iloc[top_indices].copy()
        results['similarity_score'] = masked_scores[top_indices]
        
        return results
    
    def recommend_exhibitions(
        self,
        themes: List[str],
        max_pieces_per_exhibition: int = 30,
        min_pieces_per_exhibition: int = 15
    ) -> Dict[str, pd.DataFrame]:
        """
        Recommend artworks for multiple themed exhibitions.
        
        Parameters
        ----------
        themes : List[str]
            List of exhibition themes
        max_pieces_per_exhibition : int
            Maximum artworks per exhibition
        min_pieces_per_exhibition : int
            Minimum artworks per exhibition
            
        Returns
        -------
        dict
            Dictionary mapping theme to recommended artworks
        """
        recommendations = {}
        used_ids = set()
        
        for theme in themes:
            # Get recommendations excluding already used artworks
            recs = self.recommend_for_theme(
                theme,
                n_recommendations=max_pieces_per_exhibition,
                exclude_ids=list(used_ids)
            )
            
            # Ensure minimum number of pieces
            if len(recs) < min_pieces_per_exhibition:
                logger.warning(
                    f"Only found {len(recs)} artworks for theme '{theme}' "
                    f"(minimum: {min_pieces_per_exhibition})"
                )
            
            recommendations[theme] = recs
            
            # Mark these artworks as used
            used_ids.update(recs['objectID'].values)
        
        return recommendations
    
    def evaluate_coherence(self, artwork_ids: List[int]) -> float:
        """
        Evaluate thematic coherence of a set of artworks.
        
        Parameters
        ----------
        artwork_ids : List[int]
            Object IDs of artworks
            
        Returns
        -------
        float
            Coherence score (0-1, higher is better)
        """
        indices = self.metadata[self.metadata['objectID'].isin(artwork_ids)].index
        
        if len(indices) < 2:
            return 0.0
        
        # Average pairwise similarity
        subset_sim = self.similarity_matrix[np.ix_(indices, indices)]
        
        # Exclude diagonal
        mask = ~np.eye(len(indices), dtype=bool)
        coherence = subset_sim[mask].mean()
        
        return float(coherence)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from src.data import load_met_data
    
    # Load data
    df = load_met_data()
    
    # Create dummy features for testing
    features = np.random.rand(len(df), 50)
    
    # Initialize recommender
    recommender = ExhibitionRecommender(features, df)
    
    # Test recommendations
    themes = ["ancient egypt", "portraits", "religious art"]
    exhibitions = recommender.recommend_exhibitions(themes, max_pieces_per_exhibition=20)
    
    for theme, artworks in exhibitions.items():
        print(f"\n{theme.upper()}: {len(artworks)} artworks")
        print(artworks[['title', 'department', 'similarity_score']].head())
