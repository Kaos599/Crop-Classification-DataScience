"""
Data Preprocessing Module for Crop vs. Non-Crop Land Segmentation

CS30305 - Foundations of Data Science Core
Student: Her Saleh 22 BCE 10564
Instructor: Dr. Preema Cha

This module provides comprehensive data preprocessing utilities for satellite
and UAV imagery used in crop vs. non-crop land segmentation tasks.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import cv2
from typing import Tuple, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteImageProcessor:
    """
    Comprehensive satellite image preprocessing for crop segmentation.
    
    This class handles multi-temporal satellite imagery preprocessing including
    normalization, alignment, and feature extraction for crop classification.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the satellite image processor.
        
        Args:
            config (dict): Configuration parameters for preprocessing
        """
        self.config = config
        self.scaler = None
        
    def load_satellite_image(self, image_path: str) -> np.ndarray:
        """
        Load satellite image from file path.
        
        Args:
            image_path (str): Path to satellite image file
            
        Returns:
            np.ndarray: Loaded satellite image array
        """
        try:
            with rasterio.open(image_path) as src:
                image = src.read()
                # Convert from (bands, height, width) to (height, width, bands)
                image = np.transpose(image, (1, 2, 0))
            logger.info(f"Successfully loaded image: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def normalize_image(self, image: np.ndarray, method: str = 'min_max') -> np.ndarray:
        """
        Normalize satellite image data.
        
        Args:
            image (np.ndarray): Input satellite image
            method (str): Normalization method ('min_max' or 'standard')
            
        Returns:
            np.ndarray: Normalized image
        """
        original_shape = image.shape
        image_flat = image.reshape(-1, image.shape[-1])
        
        if method == 'min_max':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                normalized = self.scaler.fit_transform(image_flat)
            else:
                normalized = self.scaler.transform(image_flat)
        elif method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                normalized = self.scaler.fit_transform(image_flat)
            else:
                normalized = self.scaler.transform(image_flat)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.reshape(original_shape)
    
    def calculate_ndvi(self, image: np.ndarray, red_band: int = 0, nir_band: int = 3) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            image (np.ndarray): Multi-spectral satellite image
            red_band (int): Index of red band
            nir_band (int): Index of near-infrared band
            
        Returns:
            np.ndarray: NDVI values
        """
        red = image[:, :, red_band].astype(float)
        nir = image[:, :, nir_band].astype(float)
        
        # Avoid division by zero
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
        
        # Clip NDVI values to valid range [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize satellite image to target dimensions.
        
        Args:
            image (np.ndarray): Input image
            target_size (Tuple[int, int]): Target (height, width)
            
        Returns:
            np.ndarray: Resized image
        """
        return cv2.resize(image, (target_size[1], target_size[0]))
    
    def apply_cloud_mask(self, image: np.ndarray, cloud_threshold: float = 0.3) -> np.ndarray:
        """
        Apply cloud masking to satellite imagery.
        
        Args:
            image (np.ndarray): Input satellite image
            cloud_threshold (float): Threshold for cloud detection
            
        Returns:
            np.ndarray: Cloud-masked image
        """
        # Simple cloud detection based on brightness in visible bands
        brightness = np.mean(image[:, :, :3], axis=2)
        cloud_mask = brightness > cloud_threshold
        
        # Apply mask
        masked_image = image.copy()
        masked_image[cloud_mask] = 0
        
        return masked_image


def preprocess_satellite_data(data_path: str, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main preprocessing function for satellite data.
    
    Args:
        data_path (str): Path to satellite data directory
        config (dict): Configuration parameters
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed images and labels
    """
    processor = SatelliteImageProcessor(config)
    
    # Implementation would load and process all satellite images
    # This is a template structure for the actual implementation
    
    logger.info("Starting satellite data preprocessing...")
    
    # Placeholder for actual implementation
    processed_images = []
    labels = []
    
    logger.info("Satellite data preprocessing completed.")
    
    return np.array(processed_images), np.array(labels)


def create_temporal_features(image_sequence: List[np.ndarray]) -> np.ndarray:
    """
    Create temporal features from multi-temporal satellite imagery.
    
    Args:
        image_sequence (List[np.ndarray]): Sequence of satellite images
        
    Returns:
        np.ndarray: Temporal features
    """
    # Calculate temporal statistics
    temporal_mean = np.mean(image_sequence, axis=0)
    temporal_std = np.std(image_sequence, axis=0)
    temporal_max = np.max(image_sequence, axis=0)
    temporal_min = np.min(image_sequence, axis=0)
    
    # Stack temporal features
    temporal_features = np.stack([
        temporal_mean, temporal_std, temporal_max, temporal_min
    ], axis=-1)
    
    return temporal_features


if __name__ == "__main__":
    # Example usage
    config = {
        'image_size': [256, 256],
        'normalization': 'min_max',
        'cloud_threshold': 0.3
    }
    
    processor = SatelliteImageProcessor(config)
    print("Satellite Image Processor initialized successfully.")