"""
Image feature extraction using Google Vision API.

This module provides functions to extract visual features from artwork images
using the Google Cloud Vision API.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
from google.cloud import vision
from google.oauth2 import service_account
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('config/.env')


class ImageFeatureExtractor:
    """Extract features from images using Google Vision API."""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize the Vision API client.
        
        Parameters
        ----------
        credentials_path : str, optional
            Path to Google Cloud credentials JSON file
        """
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            # Will use GOOGLE_APPLICATION_CREDENTIALS env variable
            self.client = vision.ImageAnnotatorClient()
        
        logger.info("Initialized Google Vision API client")
    
    def extract_features(
        self,
        image_path: str,
        max_results: int = 10
    ) -> Dict:
        """
        Extract features from a single image.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        max_results : int
            Maximum number of results per feature type
            
        Returns
        -------
        dict
            Dictionary containing extracted features:
            - labels: List of detected labels with scores
            - objects: List of detected objects with bounding boxes
            - colors: Dominant colors
            - web_entities: Related web entities
            - text: Detected text in image
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return {}
        
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        # Request multiple feature types
        features = [
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=max_results),
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=max_results),
            vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            vision.Feature(type_=vision.Feature.Type.WEB_DETECTION, max_results=max_results),
            vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
        ]
        
        request = vision.AnnotateImageRequest(image=image, features=features)
        
        try:
            response = self.client.annotate_image(request)
            
            if response.error.message:
                logger.error(f"API error: {response.error.message}")
                return {}
            
            features_dict = {
                'labels': self._extract_labels(response.label_annotations),
                'objects': self._extract_objects(response.localized_object_annotations),
                'colors': self._extract_colors(response.image_properties_annotation),
                'web_entities': self._extract_web_entities(response.web_detection),
                'text': self._extract_text(response.text_annotations)
            }
            
            return features_dict
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {}
    
    def _extract_labels(self, annotations) -> List[Dict]:
        """Extract label annotations."""
        return [
            {'description': label.description, 'score': label.score}
            for label in annotations
        ]
    
    def _extract_objects(self, annotations) -> List[Dict]:
        """Extract object localization annotations."""
        return [
            {
                'name': obj.name,
                'score': obj.score,
                'bbox': [(v.x, v.y) for v in obj.bounding_poly.normalized_vertices]
            }
            for obj in annotations
        ]
    
    def _extract_colors(self, annotation) -> List[Dict]:
        """Extract dominant colors."""
        if not annotation:
            return []
        
        return [
            {
                'color': {
                    'red': color.color.red,
                    'green': color.color.green,
                    'blue': color.color.blue
                },
                'score': color.score,
                'pixel_fraction': color.pixel_fraction
            }
            for color in annotation.dominant_colors.colors[:5]  # Top 5 colors
        ]
    
    def _extract_web_entities(self, detection) -> List[Dict]:
        """Extract web entities."""
        if not detection or not detection.web_entities:
            return []
        
        return [
            {'entity': entity.description, 'score': entity.score}
            for entity in detection.web_entities
            if entity.description  # Filter out entities without description
        ]
    
    def _extract_text(self, annotations) -> str:
        """Extract text from image."""
        if not annotations:
            return ""
        
        # First annotation contains full text
        return annotations[0].description if annotations else ""
    
    def batch_extract(
        self,
        image_paths: List[str],
        max_results: int = 10,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract features from multiple images.
        
        Parameters
        ----------
        image_paths : List[str]
            List of image paths
        max_results : int
            Maximum results per feature type
        save_path : str, optional
            Path to save results as pickle file
            
        Returns
        -------
        pd.DataFrame
            DataFrame with image paths and extracted features
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            features = self.extract_features(image_path, max_results)
            features['image_path'] = str(image_path)
            
            results.append(features)
        
        df = pd.DataFrame(results)
        
        if save_path:
            df.to_pickle(save_path)
            logger.info(f"Saved features to {save_path}")
        
        return df


def extract_label_vector(features_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Convert label features to numerical vectors.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with extracted features
    top_n : int
        Number of top labels to include
        
    Returns
    -------
    pd.DataFrame
        DataFrame with label vectors
    """
    from collections import Counter
    
    # Collect all labels
    all_labels = []
    for labels in features_df['labels']:
        if labels:
            all_labels.extend([l['description'] for l in labels])
    
    # Get top N labels
    top_labels = [label for label, _ in Counter(all_labels).most_common(top_n)]
    
    # Create binary vectors
    vectors = []
    for labels in features_df['labels']:
        vector = {label: 0 for label in top_labels}
        if labels:
            for label_info in labels:
                if label_info['description'] in top_labels:
                    vector[label_info['description']] = label_info['score']
        vectors.append(vector)
    
    return pd.DataFrame(vectors)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    extractor = ImageFeatureExtractor()
    
    # Test on a single image
    test_image = "data/raw/images/398746.jpg"
    if Path(test_image).exists():
        features = extractor.extract_features(test_image)
        print(f"\nExtracted features from {test_image}:")
        print(f"Labels: {features.get('labels', [])[:3]}")
        print(f"Objects: {features.get('objects', [])[:3]}")
        print(f"Top color: {features.get('colors', [])[:1]}")
