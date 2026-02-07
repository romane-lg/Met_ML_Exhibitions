"""
Text feature extraction and NLP preprocessing.

This module provides functions for extracting features from artwork metadata
using natural language processing techniques.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from src.features.nlp_utils import get_lemmatizer, get_stopwords, tokenize_text

logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """Extract text features from artwork metadata."""
    
    def __init__(
        self,
        max_features: int = 500,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.8
    ):
        """
        Initialize the text feature extractor.
        
        Parameters
        ----------
        max_features : int
            Maximum number of features for TF-IDF
        ngram_range : tuple
            N-gram range for TF-IDF
        min_df : int
            Minimum document frequency
        max_df : float
            Maximum document frequency
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.lemmatizer = get_lemmatizer()
        self.stop_words = get_stopwords()
        
        logger.info("Initialized TextFeatureExtractor")
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Parameters
        ----------
        text : str
            Raw text to preprocess
            
        Returns
        -------
        str
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        tokens = tokenize_text(text, stop_words=self.stop_words, lemmatizer=self.lemmatizer, min_len=2)
        return ' '.join(tokens)
    
    def combine_text_fields(
        self,
        df: pd.DataFrame,
        columns: List[str] = ['title', 'artist', 'medium', 'department', 'objectDate']
    ) -> pd.Series:
        """
        Combine multiple text columns into one.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing text columns
        columns : List[str]
            Columns to combine
            
        Returns
        -------
        pd.Series
            Combined text
        """
        combined = []
        
        for _, row in df.iterrows():
            text_parts = []
            for col in columns:
                if col in df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            combined.append(' '.join(text_parts))
        
        return pd.Series(combined, index=df.index)
    
    def extract_tfidf_features(
        self,
        texts: pd.Series,
        fit: bool = True
    ) -> np.ndarray:
        """
        Extract TF-IDF features from texts.
        
        Parameters
        ----------
        texts : pd.Series
            Series of text documents
        fit : bool
            Whether to fit the vectorizer (True for training, False for transform only)
            
        Returns
        -------
        np.ndarray
            TF-IDF feature matrix
        """
        # Preprocess texts
        processed_texts = texts.apply(self.preprocess_text)
        
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english'
            )
            features = self.tfidf_vectorizer.fit_transform(processed_texts)
            logger.info(f"Fitted TF-IDF with {features.shape[1]} features")
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("Vectorizer not fitted. Set fit=True first.")
            features = self.tfidf_vectorizer.transform(processed_texts)
        
        return features.toarray()
    
    def extract_topic_features(
        self,
        tfidf_features: np.ndarray,
        n_topics: int = 10,
        fit: bool = True
    ) -> np.ndarray:
        """
        Extract topic features using LDA.
        
        Parameters
        ----------
        tfidf_features : np.ndarray
            TF-IDF feature matrix
        n_topics : int
            Number of topics
        fit : bool
            Whether to fit the LDA model
            
        Returns
        -------
        np.ndarray
            Topic distribution matrix
        """
        if fit:
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=20,
                learning_method='online',
                random_state=42
            )
            topic_features = self.lda_model.fit_transform(tfidf_features)
            logger.info(f"Fitted LDA with {n_topics} topics")
        else:
            if self.lda_model is None:
                raise ValueError("LDA model not fitted. Set fit=True first.")
            topic_features = self.lda_model.transform(tfidf_features)
        
        return topic_features
    
    def get_top_words_per_topic(self, n_words: int = 10) -> Dict[int, List[str]]:
        """
        Get top words for each topic.
        
        Parameters
        ----------
        n_words : int
            Number of top words per topic
            
        Returns
        -------
        dict
            Dictionary mapping topic index to list of top words
        """
        if self.lda_model is None or self.tfidf_vectorizer is None:
            raise ValueError("Models not fitted yet")
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics[topic_idx] = top_words
        
        return topics
    
    def extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract categorical and numerical features from metadata.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with metadata
            
        Returns
        -------
        pd.DataFrame
            DataFrame with encoded features
        """
        features = pd.DataFrame(index=df.index)
        
        # One-hot encode department
        if 'department' in df.columns:
            dept_dummies = pd.get_dummies(df['department'], prefix='dept')
            features = pd.concat([features, dept_dummies], axis=1)
        
        # Extract century from objectDate if possible
        if 'objectDate' in df.columns:
            features['has_date'] = df['objectDate'].notna().astype(int)
        
        # Artist presence
        if 'artist' in df.columns:
            features['has_artist'] = df['artist'].notna().astype(int)
        
        return features


def extract_all_text_features(
    df: pd.DataFrame,
    text_columns: List[str] = ['title', 'artist', 'medium', 'department'],
    n_topics: int = 10,
    save_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Extract all text features from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with artwork metadata
    text_columns : List[str]
        Columns to use for text extraction
    n_topics : int
        Number of LDA topics
    save_path : str, optional
        Path to save features
        
    Returns
    -------
    dict
        Dictionary with different feature types
    """
    extractor = TextFeatureExtractor()
    
    # Combine text fields
    combined_text = extractor.combine_text_fields(df, text_columns)
    
    # Extract TF-IDF features
    tfidf_features = extractor.extract_tfidf_features(combined_text, fit=True)
    
    # Extract topic features
    topic_features = extractor.extract_topic_features(tfidf_features, n_topics, fit=True)
    
    # Extract metadata features
    metadata_features = extractor.extract_metadata_features(df)
    
    features = {
        'tfidf': tfidf_features,
        'topics': topic_features,
        'metadata': metadata_features.values,
        'extractor': extractor  # Save for later use
    }
    
    if save_path:
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(features, f)
        logger.info(f"Saved text features to {save_path}")
    
    return features


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    from src.data import load_met_data
    
    df = load_met_data()
    features = extract_all_text_features(df[:100])  # Test on subset
    
    print(f"\nTF-IDF features shape: {features['tfidf'].shape}")
    print(f"Topic features shape: {features['topics'].shape}")
    print(f"Metadata features shape: {features['metadata'].shape}")
