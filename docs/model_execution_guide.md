# Model Execution Guide

**CS30305 - Foundations of Data Science Core**  
**Student:** Her Saleh 22 BCE 10564  
**Instructor:** Dr. Preema Cha  

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Model Execution Workflow](#model-execution-workflow)
4. [Running Individual Models](#running-individual-models)
5. [Evaluation and Results](#evaluation-and-results)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

## Environment Setup

### 1. System Requirements

**Minimum Requirements:**
- Python 3.7 or higher
- 8GB RAM
- 10GB free disk space
- Internet connection for data download

**Recommended Requirements:**
- Python 3.8+
- 16GB+ RAM
- CUDA-compatible GPU (for deep learning models)
- 50GB+ free disk space
- SSD storage for faster I/O

### 2. Installation Steps

#### Step 1: Create Virtual Environment
```bash
# Using conda (recommended)
conda create --name crop_segmentation python=3.8
conda activate crop_segmentation

# Alternative: Using venv
python -m venv crop_segmentation
# Windows
crop_segmentation\Scripts\activate
# Linux/Mac
source crop_segmentation/bin/activate
```

#### Step 2: Install System Dependencies

**Windows:**
```bash
# Install GDAL using conda (easier on Windows)
conda install -c conda-forge gdal rasterio geopandas
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev python3-gdal
sudo apt-get install libspatialindex-dev
```

**macOS:**
```bash
brew install gdal
brew install spatialindex
```

#### Step 3: Install Python Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# For GPU support (optional)
pip install tensorflow-gpu
# OR for PyTorch GPU
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Verify Installation
```python
# Run this Python script to verify installation
import numpy as np
import pandas as pd
import sklearn
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt

print("✅ All core libraries installed successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

## Data Preparation

### 1. Download Dataset
```bash
# Create data directory
mkdir -p Dataset/raw
mkdir -p Dataset/processed

# Download from Google Drive link provided in Dataset/download.txt
# Extract to Dataset/raw/
```

### 2. Verify Data Structure
```
Dataset/
├── raw/
│   ├── satellite_images/
│   │   ├── image_001.tif
│   │   ├── image_002.tif
│   │   └── ...
│   └── ground_truth/
│       ├── labels_001.tif
│       ├── labels_002.tif
│       └── ...
└── processed/
    └── (will be created during preprocessing)
```

## Model Execution Workflow

### Complete Pipeline Execution

#### Option 1: Using Jupyter Notebooks (Recommended for Learning)
```bash
# Start Jupyter Lab
jupyter lab

# Execute notebooks in order:
# 1. data-preprocessing.ipynb
# 2. NDVI based/NDVI_based.ipynb  
# 3. DL model/Crop_classification_DL_model.ipynb
```

#### Option 2: Using Python Scripts (Recommended for Production)
```bash
# Run complete pipeline
python run_pipeline.py --config config/config.yaml
```

### Step-by-Step Execution

#### Step 1: Data Preprocessing
```python
from src.data_preprocessing import SatelliteImageProcessor, preprocess_satellite_data
from src.utils import load_config

# Load configuration
config = load_config('config/config.yaml')

# Initialize processor
processor = SatelliteImageProcessor(config)

# Process satellite data
X_processed, y_processed = preprocess_satellite_data('Dataset/raw/', config)

print(f"Processed data shape: {X_processed.shape}")
print(f"Labels shape: {y_processed.shape}")
```

#### Step 2: NDVI-Based Classification
```python
from src.models import NDVIClassifier
from src.data_preprocessing import SatelliteImageProcessor

# Calculate NDVI
processor = SatelliteImageProcessor(config)
ndvi_image = processor.calculate_ndvi(satellite_image)

# Initialize NDVI classifier
ndvi_classifier = NDVIClassifier(threshold=0.3)

# Make predictions
ndvi_predictions = ndvi_classifier.classify(ndvi_image)

# Evaluate
ndvi_metrics = ndvi_classifier.evaluate(ndvi_image, ground_truth)
print("NDVI Classification Results:", ndvi_metrics)
```

#### Step 3: Classical Machine Learning
```python
from src.models import ClassicalMLModels
from sklearn.model_selection import train_test_split

# Initialize classical ML models
classical_models = ClassicalMLModels(config)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed.reshape(-1, X_processed.shape[-1]), 
    y_processed.flatten(), 
    test_size=0.2, 
    random_state=42
)

# Train and evaluate each model
for model_name in ['RandomForest', 'SVM', 'GradientBoosting']:
    print(f"\n--- Training {model_name} ---")
    
    # Train model
    classical_models.train_model(model_name, X_train, y_train)
    
    # Evaluate model
    metrics = classical_models.evaluate_model(model_name, X_test, y_test)
    print(f"{model_name} Results:", metrics)
```

#### Step 4: Deep Learning Model
```python
from src.models import CropSegmentationCNN
from sklearn.model_selection import train_test_split

# Initialize CNN model
cnn_model = CropSegmentationCNN(config)

# Build U-Net architecture
input_shape = (256, 256, 5)  # height, width, channels
model = cnn_model.build_unet_model(input_shape, num_classes=2)
cnn_model.model = model

# Compile model
cnn_model.compile_model(learning_rate=0.001)

# Prepare data for CNN
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_processed, test_size=0.2, random_state=42
)

# Train model
history = cnn_model.train(
    X_train, y_train, 
    X_val, y_val, 
    epochs=100, 
    batch_size=32
)

# Evaluate model
cnn_metrics = cnn_model.evaluate(X_val, y_val)
print("CNN Results:", cnn_metrics)
```

## Running Individual Models

### 1. Quick NDVI Classification
```bash
python -c "
from src.models import NDVIClassifier
from src.data_preprocessing import SatelliteImageProcessor
from src.utils import load_config
import numpy as np

config = load_config('config/config.yaml')
processor = SatelliteImageProcessor(config)

# Load sample image (replace with actual path)
image = processor.load_satellite_image('Dataset/raw/sample_image.tif')
ndvi = processor.calculate_ndvi(image)

classifier = NDVIClassifier(threshold=0.3)
result = classifier.classify(ndvi)

print(f'Crop pixels: {np.sum(result)}')
print(f'Non-crop pixels: {np.sum(1-result)}')
"
```

### 2. Train Single Classical Model
```bash
python -c "
from src.models import ClassicalMLModels
from src.utils import load_config
import numpy as np

config = load_config('config/config.yaml')
models = ClassicalMLModels(config)

# Generate sample data (replace with actual data loading)
X_sample = np.random.rand(1000, 10)
y_sample = np.random.randint(0, 2, 1000)

models.train_model('RandomForest', X_sample, y_sample)
print('Random Forest trained successfully!')
"
```

### 3. Quick CNN Training
```bash
python -c "
from src.models import CropSegmentationCNN
from src.utils import load_config
import numpy as np

config = load_config('config/config.yaml')
cnn = CropSegmentationCNN(config)

# Build model
model = cnn.build_unet_model((256, 256, 5), 2)
cnn.model = model
cnn.compile_model()

print('CNN model built and compiled successfully!')
print(f'Model parameters: {model.count_params():,}')
"
```

## Evaluation and Results

### 1. Generate Comprehensive Evaluation
```python
from src.utils import calculate_comprehensive_metrics, plot_confusion_matrix
from src.utils import visualize_segmentation_results, save_metrics_to_csv

# Calculate metrics for all models
all_metrics = {}

# NDVI metrics
all_metrics['NDVI'] = ndvi_classifier.evaluate(ndvi_image, ground_truth)

# Classical ML metrics
for model_name in ['RandomForest', 'SVM', 'GradientBoosting']:
    all_metrics[model_name] = classical_models.evaluate_model(model_name, X_test, y_test)

# CNN metrics
all_metrics['CNN'] = cnn_model.evaluate(X_val, y_val)

# Save results
save_metrics_to_csv(all_metrics, 'results/metrics/all_model_metrics.csv')
```

### 2. Generate Visualizations
```python
from src.utils import plot_training_history, create_ndvi_visualization

# Plot CNN training history
if hasattr(cnn_model, 'history') and cnn_model.history:
    plot_training_history(
        cnn_model.history.history, 
        save_path='results/figures/training_history.png'
    )

# Create NDVI visualization
create_ndvi_visualization(
    satellite_image, ndvi_image, threshold=0.3,
    save_path='results/figures/ndvi_analysis.png'
)

# Visualize segmentation results
cnn_predictions = cnn_model.predict(X_val[:1])
visualize_segmentation_results(
    X_val[0], y_val[0], cnn_predictions[0],
    save_path='results/figures/segmentation_results.png'
)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. GDAL Installation Issues
```bash
# Windows: Use conda-forge
conda install -c conda-forge gdal

# Linux: Install system packages first
sudo apt-get install gdal-bin libgdal-dev

# Mac: Use homebrew
brew install gdal
```

#### 2. Memory Issues
```python
# Reduce batch size for CNN training
cnn_model.train(X_train, y_train, X_val, y_val, batch_size=16)  # Instead of 32

# Process images in chunks
def process_in_chunks(data, chunk_size=100):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]
```

#### 3. GPU Not Detected
```python
# Check GPU availability
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Force CPU usage if needed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

#### 4. Data Loading Issues
```python
# Verify data paths
import os
data_path = 'Dataset/raw/'
if not os.path.exists(data_path):
    print(f"Data path {data_path} does not exist!")
    print("Please check the dataset download and extraction.")
```

### Performance Monitoring
```python
# Monitor memory usage
import psutil
import time

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.2f} MB")

# Use during training
monitor_memory()
```

## Performance Optimization

### 1. Data Loading Optimization
```python
# Use data generators for large datasets
def data_generator(data_path, batch_size=32):
    while True:
        # Load batch of data
        batch_x, batch_y = load_batch(data_path, batch_size)
        yield batch_x, batch_y

# Use with CNN training
cnn_model.model.fit(
    data_generator('Dataset/processed/', 32),
    steps_per_epoch=100,
    epochs=50
)
```

### 2. Model Optimization
```python
# Use mixed precision training for faster GPU training
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
```

### 3. Parallel Processing
```python
# Use multiprocessing for data preprocessing
from multiprocessing import Pool
import multiprocessing as mp

def process_image_parallel(image_paths):
    with Pool(mp.cpu_count()) as pool:
        results = pool.map(processor.load_satellite_image, image_paths)
    return results
```

## Expected Results

### Performance Benchmarks
- **NDVI Classification**: ~75-80% accuracy (baseline)
- **Random Forest**: ~82-85% accuracy
- **SVM**: ~80-83% accuracy  
- **Gradient Boosting**: ~84-87% accuracy
- **CNN (U-Net)**: ~89-92% accuracy (best performance)

### Execution Times (approximate)
- **Data Preprocessing**: 10-30 minutes
- **NDVI Classification**: 1-2 minutes
- **Classical ML Training**: 5-15 minutes each
- **CNN Training**: 2-6 hours (depending on GPU)

### Output Files
```
results/
├── figures/
│   ├── confusion_matrices/
│   ├── training_history.png
│   ├── ndvi_analysis.png
│   └── segmentation_results.png
├── metrics/
│   ├── all_model_metrics.csv
│   └── detailed_evaluation.json
└── models/
    ├── best_cnn_model.h5
    ├── random_forest_model.pkl
    └── svm_model.pkl
```

---

**Note**: This guide provides comprehensive instructions for executing the crop vs. non-crop land segmentation models. For specific issues or advanced configurations, refer to the individual module documentation in the `src/` directory.