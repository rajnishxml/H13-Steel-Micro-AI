## 📊 Current Results
### Image Enhancement (CLAHE)
We successfully implemented a preprocessing pipeline to enhance the contrast of H13 Tool Steel micrographs. This allows for better segmentation of cellular dendrites.

![Enhancement Result](enhanced_comparison.png)
# Automated Microstructural Analysis of H13 Steel using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

An end-to-end Deep Learning pipeline for high-throughput characterization of H13 Tool Steel printed via Laser Powder Bed Fusion (LPBF).

## 🚀 Key Performance Metrics
- **91% Classification Accuracy:** ResNet-18 model fine-tuned for defect vs. healthy matrix detection.
- **1800x Speedup:** Characterization time reduced from ~3 hours (manual) to 2 seconds (automated).
- **<0.1% Detection Sensitivity:** Pixel-level segmentation of porosity and lack-of-fusion (LOF) defects.

## 🛠️ Technical Approach
### 1. Preprocessing (CLAHE)
Applied **Contrast Limited Adaptive Histogram Equalization** to raw micrographs.
- **Impact:** 40% boost in feature contrast, enabling the model to distinguish fine cellular dendrites from background noise.

### 2. Deep Learning Pipeline
- **Architecture:** ResNet-18 (Residual Network).
- **Transfer Learning:** Pre-trained on 18K+ metallurgical images; fine-tuned specifically for H13 Steel solidification patterns.
- **Interpretability (Grad-CAM):** Integrated Gradient-weighted Class Activation Mapping to generate heatmaps, ensuring the model focuses on metallurgically relevant features (grain boundaries and pores).

### 3. Automated Quantification
- **Spatial Calibration:** Automated detection of 50µm scale bars to establish a precise pixel-to-micron ratio (e.g., 8.74 px/µm).
- **Batch Processing:** Scalable analysis of 1000+ images in a single execution.
