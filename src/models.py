"""
Simple neural network models for classification.
"""

import torch
import torch.nn as nn
from torchvision import models


class SteelClassifier(nn.Module):
    """
    Simple ResNet-18 classifier.
    Transfer learning ready.
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Load pretrained ResNet-18
        self.model = models.resnet18(pretrained=True)
        
        # Change final layer for our classes
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.model(x)


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for quick testing.
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Simple 3-layer CNN
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(name='resnet18', num_classes=2):
    """
    Get model by name.
    
    Args:
        name: 'resnet18' or 'simple'
        num_classes: Number of output classes
    
    Returns:
        Model
    """
    if name == 'resnet18':
        return SteelClassifier(num_classes)
    elif name == 'simple':
        return SimpleCNN(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")