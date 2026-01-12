"""
Image preprocessing for steel microstructure analysis.
Simple and easy to understand.
"""

import cv2
import numpy as np


def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """
    Apply CLAHE to enhance image contrast.
    
    Args:
        image: Input image (grayscale or color)
        clip_limit: Contrast limit (default: 2.0)
        tile_size: Grid size (default: 8x8)
    
    Returns:
        Enhanced image
    
    Example:
        >>> enhanced = apply_clahe(image)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(gray)
    
    return enhanced


def calculate_metrics(image):
    """
    Calculate simple contrast metrics.
    
    Args:
        image: Input image
    
    Returns:
        Dictionary with metrics
    """
    return {
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'min': int(np.min(image)),
        'max': int(np.max(image))
    }


def resize_image(image, width=224, height=224):
    """Resize image to target size."""
    return cv2.resize(image, (width, height))


def normalize_image(image):
    """Normalize image to 0-1 range."""
    return image.astype(np.float32) / 255.0