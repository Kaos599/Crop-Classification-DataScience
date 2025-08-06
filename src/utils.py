"""
Utility Functions for Crop Segmentation Project

CS30305 - Foundations of Data Science Core
Student: Her Saleh 22 BCE 10564
Instructor: Dr. Preema Cha

This module provides utility functions for data visualization, evaluation,
and general helper functions for the crop segmentation project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import jaccard_score, accuracy_score
import cv2
from typing import Dict, List, Tuple, Optional, Union
import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) for segmentation.
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        float: IoU score
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    if union == 0:
        return 1.0  # Perfect score when both are empty
    
    return intersection / union


def calculate_dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Dice coefficient for segmentation evaluation.
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        float: Dice coefficient
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()
    
    if total == 0:
        return 1.0  # Perfect score when both are empty
    
    return (2.0 * intersection) / total


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = None, 
                         save_path: str = None) -> plt.Figure:
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str]): Names of classes
        save_path (str): Path to save the plot
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_training_history(history: dict, save_path: str = None) -> plt.Figure:
    """
    Plot training history for deep learning models.
    
    Args:
        history (dict): Training history from model.fit()
        save_path (str): Path to save the plot
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Plot training & validation loss
    axes[0, 1].plot(history['loss'], label='Training Loss')
    axes[0, 1].plot(history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Plot precision
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Training Precision')
        axes[1, 0].plot(history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
    
    # Plot recall
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Training Recall')
        axes[1, 1].plot(history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    return fig


def visualize_segmentation_results(image: np.ndarray, ground_truth: np.ndarray, 
                                 prediction: np.ndarray, save_path: str = None) -> plt.Figure:
    """
    Visualize segmentation results with original image, ground truth, and prediction.
    
    Args:
        image (np.ndarray): Original satellite image
        ground_truth (np.ndarray): Ground truth segmentation
        prediction (np.ndarray): Predicted segmentation
        save_path (str): Path to save the visualization
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image (RGB bands)
    if image.shape[-1] >= 3:
        rgb_image = image[:, :, :3]
        # Normalize for display
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        axes[0].imshow(rgb_image)
    else:
        axes[0].imshow(image[:, :, 0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(ground_truth, cmap='RdYlGn', alpha=0.7)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction, cmap='RdYlGn', alpha=0.7)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Segmentation visualization saved to {save_path}")
    
    return fig


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for segmentation.
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Flatten arrays for metric calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    metrics = {
        'accuracy': accuracy_score(y_true_flat, y_pred_flat),
        'iou': calculate_iou(y_true, y_pred),
        'dice_coefficient': calculate_dice_coefficient(y_true, y_pred),
        'jaccard_score': jaccard_score(y_true_flat, y_pred_flat, average='weighted')
    }
    
    return metrics


def save_metrics_to_csv(metrics_dict: Dict[str, Dict[str, float]], 
                       save_path: str):
    """
    Save evaluation metrics to CSV file.
    
    Args:
        metrics_dict (Dict[str, Dict[str, float]]): Nested dictionary of metrics
        save_path (str): Path to save CSV file
    """
    df = pd.DataFrame(metrics_dict).T
    df.to_csv(save_path, index=True)
    logger.info(f"Metrics saved to {save_path}")


def create_ndvi_visualization(image: np.ndarray, ndvi: np.ndarray, 
                            threshold: float = 0.3, save_path: str = None) -> plt.Figure:
    """
    Create NDVI visualization with threshold overlay.
    
    Args:
        image (np.ndarray): Original satellite image
        ndvi (np.ndarray): NDVI values
        threshold (float): NDVI threshold for crop classification
        save_path (str): Path to save the visualization
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if image.shape[-1] >= 3:
        rgb_image = image[:, :, :3]
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        axes[0].imshow(rgb_image)
    else:
        axes[0].imshow(image[:, :, 0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # NDVI
    ndvi_plot = axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title('NDVI')
    axes[1].axis('off')
    plt.colorbar(ndvi_plot, ax=axes[1], fraction=0.046, pad=0.04)
    
    # NDVI with threshold
    crop_mask = ndvi > threshold
    axes[2].imshow(crop_mask, cmap='RdYlGn', alpha=0.7)
    axes[2].set_title(f'Crop Classification (NDVI > {threshold})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"NDVI visualization saved to {save_path}")
    
    return fig


def ensure_directory_exists(directory_path: str):
    """
    Ensure that a directory exists, create if it doesn't.
    
    Args:
        directory_path (str): Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")


def log_experiment_results(experiment_name: str, metrics: Dict[str, float], 
                         config: dict, log_file: str = "experiment_log.txt"):
    """
    Log experiment results to file.
    
    Args:
        experiment_name (str): Name of the experiment
        metrics (Dict[str, float]): Evaluation metrics
        config (dict): Configuration used
        log_file (str): Path to log file
    """
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Results:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"{'='*50}\n")
    
    logger.info(f"Experiment results logged to {log_file}")


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully.")
    
    # Test configuration loading
    try:
        config = load_config("config/config.yaml")
        print("Configuration loaded successfully.")
    except:
        print("Configuration file not found - this is expected in testing.")