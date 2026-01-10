## 📊 Current Results
### Image Enhancement (CLAHE)
We successfully implemented a preprocessing pipeline to enhance the contrast of H13 Tool Steel micrographs. This allows for better segmentation of cellular dendrites.

![Enhancement Result](enhanced_comparison.png)

 # H13-Steel-Micro-AI
Machine Learning and Computer Vision Pipeline for Microstructure Analysis of LPBF H13 Tool Steel

## 📌 Project Overview
This repository contains two major ML projects:
1. Image-based defect detection (porosity, cracks) using classical computer vision & later deep learning.
2. Machine learning regression model to predict porosity and defect density from LPBF process parameters.

## 🔬 Project 1: Image-Based Defect Detection
Contains:
- Image preprocessing (Gaussian blur, CLAHE)
- Otsu thresholding
- Morphological operations
- Defect area measurement
- Future: UNet / YOLO segmentation models

## 📊 Project 2: Process Parameter ML Prediction
Contains:
- Sample dataset
- Preprocessing and feature engineering
- Baseline Linear Regression
- Future: Random Forest, XGBoost, Neural Network

## 🗂️ Repository Structure
notebooks/ – Jupyter notebooks for CV and ML  
scripts/ – Python scripts (preprocessing, segmentation, ML)  
data_samples/ – Placeholder for SEM/optical images  
results/ – Will store output plots and segmented images
