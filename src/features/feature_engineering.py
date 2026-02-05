"""
Feature engineering - combine image and text features.

This module provides functions to combine features from different sources
and prepare them for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging
import pickle

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Combine and engineer features from multiple sources."""
    
    def __init__(
        self,
        vision_weight: float = 0.5,
        text_weight: float = 0.5,
        use_pca: bool = True,
        n_components: int = 50,
        scaler_type: str = 'standard'
    ):
        """
        Initialize the feature engineer.
        
        Parameters
        ----------
        vision_weight : float
            Weight for vision features (0-1)
        text_weight : float
            Weight for text features (0-1)
        use_pca : bool
            Whether to use PCA for dimensionality reduction
        n_components : int
            Number of PCA components
        scaler_type : str
            Type of scaler: 'standard', 'minmax', or 'robust'
        """
        self.vision_weight = vision_weight
        self.text_weight = text_weight
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler_type = scaler_type
        
        self.scaler = None
        self.pca = None
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        logger.info(f"Initialized FeatureEngineer with {scaler_type} scaler")
    
    def combine_features(
        self,
        vision_features: np.ndarray,
        text_features: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Combine vision and text features with weights.
        
        Parameters
        ----------
        vision_features : np.ndarray
            Vision feature matrix (n_samples, n_vision_features)
        text_features : np.ndarray
            Text feature matrix (n_samples, n_text_features)
        normalize : bool
            Whether to normalize features before combining
            
        Returns
        -------
        np.ndarray
            Combined feature matrix
        """
        if vision_features.shape[0] != text_features.shape[0]:
            raise ValueError("Number of samples must match")
        
        # Normalize each feature type separately
        if normalize:
            vision_norm = self.scaler.fit_transform(vision_features)
            text_norm = self.scaler.fit_transform(text_features)
        else:
            vision_norm = vision_features
            text_norm = text_features
        
        # Apply weights
        vision_weighted = vision_norm * self.vision_weight
        text_weighted = text_norm * self.text_weight
        
        # Concatenate
        combined = np.hstack([vision_weighted, text_weighted])
        
        logger.info(f"Combined features shape: {combined.shape}")
        
        return combined
    
    def reduce_dimensions(
        self,
        features: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix
        fit : bool
            Whether to fit PCA model
            
        Returns
        -------
        np.ndarray
            Reduced feature matrix
        """
        if not self.use_pca:
            return features
        
        if fit:
            self.pca = PCA(n_components=self.n_components, random_state=42)
            reduced = self.pca.fit_transform(features)
            
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA reduced to {self.n_components} components")
            logger.info(f"Explained variance: {explained_var:.2%}")
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted yet")
            reduced = self.pca.transform(features)
        
        return reduced
    
    def fit_transform(
        self,
        vision_features: np.ndarray,
        text_features: np.ndarray
    ) -> np.ndarray:
        """
        Fit and transform features (for training data).
        
        Parameters
        ----------
        vision_features : np.ndarray
            Vision features
        text_features : np.ndarray
            Text features
            
        Returns
        -------
        np.ndarray
            Processed feature matrix
        """
        # Combine features
        combined = self.combine_features(vision_features, text_features, normalize=True)
        
        # Apply PCA if enabled
        if self.use_pca:
            combined = self.reduce_dimensions(combined, fit=True)
        
        return combined
    
    def transform(
        self,
        vision_features: np.ndarray,
        text_features: np.ndarray
    ) -> np.ndarray:
        """
        Transform features using fitted scalers (for new data).
        
        Parameters
        ----------
        vision_features : np.ndarray
            Vision features
        text_features : np.ndarray
            Text features
            
        Returns
        -------
        np.ndarray
            Processed feature matrix
        """
        # Combine features (without fitting)
        combined = self.combine_features(vision_features, text_features, normalize=False)
        
        # Apply PCA if enabled
        if self.use_pca:
            combined = self.reduce_dimensions(combined, fit=False)
        
        return combined
    
    def save(self, filepath: str):
        """Save the feature engineer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved FeatureEngineer to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FeatureEngineer':
        """Load a feature engineer from disk."""
        with open(filepath, 'rb') as f:
            engineer = pickle.load(f)
        logger.info(f"Loaded FeatureEngineer from {filepath}")
        return engineer


def create_feature_pipeline(
    df: pd.DataFrame,
    vision_features_path: Optional[str] = None,
    text_features_path: Optional[str] = None,
    save_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, FeatureEngineer]:
    """
    Create a complete feature engineering pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Artwork metadata DataFrame
    vision_features_path : str, optional
        Path to saved vision features
    text_features_path : str, optional
        Path to saved text features
    save_path : str, optional
        Path to save combined features
    config : dict, optional
        Configuration parameters
        
    Returns
    -------
    tuple
        (combined_features, feature_engineer)
    """
    if config is None:
        config = {
            'vision_weight': 0.5,
            'text_weight': 0.5,
            'use_pca': True,
            'n_components': 50,
            'scaler_type': 'standard'
        }
    
    # Load or create vision features
    if vision_features_path:
        with open(vision_features_path, 'rb') as f:
            vision_data = pickle.load(f)
        # Extract relevant features (e.g., label vectors)
        vision_features = vision_data  # Adjust based on actual structure
    else:
        logger.warning("No vision features provided, using zeros")
        vision_features = np.zeros((len(df), 10))
    
    # Load or create text features
    if text_features_path:
        with open(text_features_path, 'rb') as f:
            text_data = pickle.load(f)
        # Combine TF-IDF and topic features
        text_features = np.hstack([
            text_data.get('tfidf', np.zeros((len(df), 10))),
            text_data.get('topics', np.zeros((len(df), 10)))
        ])
    else:
        logger.warning("No text features provided, using zeros")
        text_features = np.zeros((len(df), 10))
    
    # Create feature engineer
    engineer = FeatureEngineer(**config)
    
    # Fit and transform
    combined_features = engineer.fit_transform(vision_features, text_features)
    
    # Save if requested
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'features': combined_features,
                'engineer': engineer,
                'object_ids': df['objectID'].values
            }, f)
        logger.info(f"Saved combined features to {save_path}")
    
    return combined_features, engineer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy features for testing
    n_samples = 100
    vision_features = np.random.rand(n_samples, 20)
    text_features = np.random.rand(n_samples, 30)
    
    engineer = FeatureEngineer(use_pca=True, n_components=10)
    combined = engineer.fit_transform(vision_features, text_features)
    
    print(f"Original shapes: vision={vision_features.shape}, text={text_features.shape}")
    print(f"Combined shape: {combined.shape}")
