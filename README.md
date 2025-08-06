# Crop vs. Non-Crop Land Segmentation (Satellite/UAV Images)

**Academic Assignment Submission**  
**Course:** Foundations of Data Science Core (CS3005)  
**Student:** Harsh Dayal 22BCE10564  
**Instructor:** Dr. Prerna Jha 

## Project Overview

This project implements advanced machine learning and deep learning techniques for crop vs. non-crop land segmentation using satellite and UAV imagery. The work focuses on developing robust classification models that can distinguish between agricultural and non-agricultural land areas using multi-temporal satellite data and spectral analysis.

### Key Features
- Multi-temporal satellite image analysis for crop segmentation
- NDVI-based vegetation index classification
- Deep learning models for improved accuracy
- Comprehensive data preprocessing pipeline
- Performance evaluation with multiple metrics

![Project Overview](cover1.png)

## Methodology

This project employs a comprehensive approach to crop vs. non-crop land segmentation:

1. **Data Preprocessing**: Multi-temporal satellite image preparation and normalization
2. **Feature Engineering**: NDVI calculation and spectral band analysis
3. **Classical ML Approaches**: Traditional machine learning methods for baseline comparison
4. **Deep Learning Models**: Advanced neural networks for improved segmentation accuracy
5. **Evaluation**: Comprehensive performance assessment using multiple metrics

The methodology is based on research by **Rose M. Rustowicz** and incorporates modern deep learning techniques for enhanced accuracy in agricultural land classification.

## Project Structure

```
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration parameters
├── Dataset/               # Raw dataset storage
│   └── download.txt      # Dataset download instructions
├── DL model/             # Deep learning model implementations
│   └── Crop_classification_DL_model.ipynb
├── docs/                 # Documentation and references
│   └── references.md     # Academic references and citations
├── NDVI based/           # NDVI-based classification methods
│   └── NDVI_based.ipynb
├── notebooks/            # Jupyter notebooks for analysis
│   └── __init__.py
├── results/              # Output results and artifacts
│   ├── figures/          # Generated plots and visualizations
│   ├── metrics/          # Performance metrics and evaluations
│   └── models/           # Trained model artifacts
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data preprocessing utilities
│   ├── models.py         # ML/DL model implementations
│   └── utils.py          # Utility functions
├── data-preprocessing.ipynb   # Main preprocessing notebook
├── cover1.png            # Project overview image
├── test3.kml            # Sample KML file for testing
└── README.md            # This file
```

## Installation and Setup

### Prerequisites
- Python 3.7+
- GDAL library for geospatial data processing
- CUDA-compatible GPU (recommended for deep learning models)

### Environment Setup
```bash
# Create conda environment
conda create --name crop_segmentation python=3.7

# Activate environment
conda activate crop_segmentation

# Install geospatial dependencies
conda install gdal rasterio

# Install core data science packages
conda install numpy pandas geopandas scikit-learn jupyterlab matplotlib seaborn

# Install additional packages
conda install xarray rasterstats tqdm pytest sqlalchemy scikit-image scipy
conda install pysal beautifulsoup4 boto3 cython statsmodels future graphviz
conda install pylint line_profiler nodejs sphinx

# For deep learning (optional - choose based on your system)
conda install pytorch torchvision torchaudio -c pytorch
# OR
pip install tensorflow
```

## Dataset

The project utilizes multi-temporal satellite imagery for crop segmentation:

- **Satellite Data**: 10 RapidEye satellite images from Planet.com
- **Ground Truth**: USDA Cropland Data Layer for pixel-level crop labels
- **Format**: Multi-spectral imagery with temporal sequences
- **Coverage**: Agricultural regions with diverse crop types

### Data Access
Dataset can be downloaded from the provided Google Drive link in `Dataset/download.txt`.

## Quick Start

### 1. Install Dependencies
```bash
# Create and activate virtual environment
conda create --name crop_segmentation python=3.8
conda activate crop_segmentation

# Install requirements
pip install -r requirements.txt
```

### 2. Basic Usage

#### Option A: Jupyter Notebooks (Recommended for Learning)
```bash
# Start Jupyter Lab
jupyter lab

# Execute notebooks in order:
# 1. data-preprocessing.ipynb
# 2. NDVI based/NDVI_based.ipynb  
# 3. DL model/Crop_classification_DL_model.ipynb
```

#### Option B: Python Scripts (Production)
```python
# Import preprocessing utilities
from src.data_preprocessing import preprocess_satellite_data
from src.models import CropSegmentationCNN, ClassicalMLModels, NDVIClassifier
from src.utils import load_config, calculate_comprehensive_metrics

# Load configuration
config = load_config('config/config.yaml')

# Process data and train models
X_processed, y_processed = preprocess_satellite_data('Dataset/raw/', config)
```

### 3. Detailed Execution Guide

**📖 For comprehensive setup, troubleshooting, and execution instructions, see:**
**[Model Execution Guide](docs/model_execution_guide.md)**

This detailed guide covers:
- Complete environment setup and installation
- Step-by-step model execution workflow
- Performance optimization techniques
- Troubleshooting common issues
- Expected results and benchmarks

## Technical Implementation

### Data Preprocessing
- Multi-temporal image alignment and normalization
- Spectral band extraction and processing
- NDVI calculation for vegetation analysis
- Data augmentation for improved model robustness

### Model Architecture
- **Classical ML**: Random Forest, SVM, and ensemble methods
- **Deep Learning**: Convolutional Neural Networks (CNN) for image segmentation
- **Feature Engineering**: Spectral indices and temporal features
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, IoU

### Key Algorithms
1. **NDVI-Based Classification**: Vegetation index thresholding
2. **Multi-Temporal Analysis**: Time-series pattern recognition
3. **Deep CNN**: Semantic segmentation for pixel-level classification
4. **Ensemble Methods**: Combining multiple models for improved accuracy

## Results and Performance

The project achieves significant improvements over traditional mono-temporal approaches:

- **Baseline Accuracy**: ~75% (traditional spectral methods)
- **NDVI-Enhanced**: ~82% (with vegetation indices)
- **Deep Learning**: ~89% (CNN-based segmentation)
- **Multi-Temporal**: ~92% (temporal pattern analysis)

Detailed results and visualizations are available in the `results/` directory.

## Academic Contributions

This project demonstrates:
1. **Methodological Innovation**: Integration of multi-temporal analysis with deep learning
2. **Technical Implementation**: Robust preprocessing pipeline for satellite imagery
3. **Performance Analysis**: Comprehensive evaluation of different approaches
4. **Practical Application**: Real-world agricultural monitoring capabilities

## Documentation

### 📚 Complete Documentation
- **[Model Execution Guide](docs/model_execution_guide.md)** - Comprehensive setup and execution instructions
- **[Academic References](docs/references.md)** - Complete academic references and related work

### 📋 Additional Resources
- **[requirements.txt](requirements.txt)** - Complete list of Python dependencies
- **[Configuration Guide](config/config.yaml)** - Model and data processing parameters

## Future Work

- Integration of additional satellite data sources (Sentinel-2, Landsat)
- Real-time processing capabilities for operational deployment
- Extension to multi-class crop type classification
- Integration with IoT sensors for ground-truth validation

## License

This project is submitted as part of academic coursework for CS30305 - Foundations of Data Science Core.

---

**Note**: This implementation focuses on demonstrating advanced data science techniques for agricultural applications, combining traditional machine learning with modern deep learning approaches for improved crop vs. non-crop land segmentation accuracy.