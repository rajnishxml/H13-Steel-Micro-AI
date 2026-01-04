import cv2  # OpenCV library for image processing operations
import numpy as np  # NumPy for numerical array operations
import matplotlib.pyplot as plt  # Matplotlib for displaying images

def enhance_micrograph(image_path):
    """
    Enhances metallurgical micrographs by reducing noise and improving contrast.
    
    This function processes microscope images of metal structures to make
    features like grain boundaries and phases more visible for analysis.
    
    Parameters:
    -----------
    image_path : str
        Path to the input micrograph image file
        
    Returns:
    --------
    tuple : (original_image, enhanced_image)
        Returns both the original and processed images, or (None, error_message) if failed
    """
    
    # Step 1: Load the image in grayscale mode (single channel, 0-255 intensity values)
    img = cv2.imread(image_path, 0)
    
    # Check if image was loaded successfully
    if img is None: 
        return None, "Error: Image file not found or could not be loaded!"
    
    # Step 2: Apply noise reduction to remove scanning artifacts and random noise
    # Parameters: image, None (output array), h=10 (filter strength), 
    # templateWindowSize=7, searchWindowSize=21
    denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    
    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This enhances local contrast without over-amplifying noise
    # clipLimit=3.0 prevents over-contrast, tileGridSize=(8,8) defines processing blocks
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    return img, enhanced


# ========== MAIN EXECUTION SECTION ==========

# Process the micrograph image (change '1.tif' to your actual filename)
original, processed = enhance_micrograph('1.tif')

# Display results if processing was successful
if original is not None:
    # Create a figure with two side-by-side subplots
    plt.figure(figsize=(12, 6))
    
    # Left subplot: Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image: 1.tif')
    plt.axis('off')
    
    # Right subplot: Enhanced image
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title('Enhanced Microstructure')
    plt.axis('off')
    
    plt.show()
    print("✓ Image processing completed successfully!")
else:
    print("✗ Error: Please verify that '1.tif' is uploaded in the Colab file browser.")
