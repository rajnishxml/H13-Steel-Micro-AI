"""
Simple data loading for images.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path


class ImageDataset(Dataset):
    """
    Simple dataset for loading images.
    """
    
    def __init__(self, image_folder, image_size=224):
        """
        Args:
            image_folder: Folder with images
            image_size: Size to resize images to
        """
        self.image_folder = Path(image_folder)
        self.image_size = image_size
        
        # Get all image files
        self.images = list(self.image_folder.glob('*.jpg'))
        self.images += list(self.image_folder.glob('*.png'))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # To tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, 0  # Return dummy label for now


def create_dataloader(folder, batch_size=32, shuffle=True):
    """
    Create simple dataloader.
    
    Args:
        folder: Image folder path
        batch_size: Batch size
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader
    """
    dataset = ImageDataset(folder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader