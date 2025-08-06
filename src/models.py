"""
Machine Learning and Deep Learning Models for Crop Segmentation

CS30305 - Foundations of Data Science Core
Student: Her Saleh 22 BCE 10564
Instructor: Dr. Preema Cha

This module implements various machine learning and deep learning models
for crop vs. non-crop land segmentation using satellite imagery.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassicalMLModels:
    """
    Classical machine learning models for crop classification.
    
    This class implements traditional ML algorithms including Random Forest,
    SVM, and Gradient Boosting for baseline comparison.
    """
    
    def __init__(self, config: dict):
        """
        Initialize classical ML models.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all classical ML models."""
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train a specific classical ML model.
        
        Args:
            model_name (str): Name of the model to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        logger.info(f"Training {model_name} model...")
        self.models[model_name].fit(X_train, y_train)
        logger.info(f"{model_name} training completed.")
    
    def predict(self, model_name: str, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Predictions
        """
        return self.models[model_name].predict(X_test)
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        y_pred = self.predict(model_name, X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics


class CropSegmentationCNN:
    """
    Convolutional Neural Network for crop segmentation.
    
    This class implements a U-Net architecture for semantic segmentation
    of crop vs. non-crop areas in satellite imagery.
    """
    
    def __init__(self, config: dict):
        """
        Initialize CNN model.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.model = None
        self.history = None
        
    def build_unet_model(self, input_shape: Tuple[int, int, int], num_classes: int = 2) -> tf.keras.Model:
        """
        Build U-Net architecture for semantic segmentation.
        
        Args:
            input_shape (Tuple[int, int, int]): Input image shape
            num_classes (int): Number of output classes
            
        Returns:
            tf.keras.Model: U-Net model
        """
        inputs = layers.Input(shape=input_shape)
        
        # Encoder (Contracting Path)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        
        # Bottleneck
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        
        # Decoder (Expanding Path)
        up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
        up6 = layers.concatenate([up6, conv4])
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
        
        up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
        up7 = layers.concatenate([up7, conv3])
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
        
        up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
        up8 = layers.concatenate([up8, conv2])
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
        
        up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
        up9 = layers.concatenate([up9, conv1])
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
        
        # Output layer
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv9)
        
        model = models.Model(inputs=[inputs], outputs=[outputs])
        
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate (float): Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_unet_model first.")
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              epochs: int = 100, batch_size: int = 32) -> tf.keras.callbacks.History:
        """
        Train the CNN model.
        
        Args:
            X_train (np.ndarray): Training images
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation images
            y_val (np.ndarray): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        logger.info("Starting CNN model training...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("CNN model training completed.")
        
        return self.history
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X_test (np.ndarray): Test images
            
        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test images
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        
        return metrics


class NDVIClassifier:
    """
    NDVI-based crop classification model.
    
    This class implements vegetation index-based classification for
    distinguishing crop from non-crop areas.
    """
    
    def __init__(self, threshold: float = 0.3):
        """
        Initialize NDVI classifier.
        
        Args:
            threshold (float): NDVI threshold for crop classification
        """
        self.threshold = threshold
    
    def classify(self, ndvi_image: np.ndarray) -> np.ndarray:
        """
        Classify crop vs. non-crop based on NDVI threshold.
        
        Args:
            ndvi_image (np.ndarray): NDVI values
            
        Returns:
            np.ndarray: Binary classification (1=crop, 0=non-crop)
        """
        return (ndvi_image > self.threshold).astype(int)
    
    def evaluate(self, ndvi_image: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Evaluate NDVI classification performance.
        
        Args:
            ndvi_image (np.ndarray): NDVI values
            ground_truth (np.ndarray): Ground truth labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        predictions = self.classify(ndvi_image)
        
        # Flatten arrays for metric calculation
        y_true = ground_truth.flatten()
        y_pred = predictions.flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics


if __name__ == "__main__":
    # Example usage
    config = {
        'models': ['RandomForest', 'SVM', 'GradientBoosting'],
        'input_shape': (256, 256, 5),
        'num_classes': 2
    }
    
    # Initialize models
    classical_models = ClassicalMLModels(config)
    cnn_model = CropSegmentationCNN(config)
    ndvi_classifier = NDVIClassifier(threshold=0.3)
    
    print("All models initialized successfully.")