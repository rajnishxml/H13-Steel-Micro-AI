"""
Simple utility functions.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def save_model(model, path):
    """Save model to file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """Load model from file."""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model


def plot_history(train_loss, val_loss):
    """
    Plot training history.
    
    Args:
        train_loss: List of training losses
        val_loss: List of validation losses
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.show()


def calculate_accuracy(predictions, labels):
    """
    Calculate accuracy.
    
    Args:
        predictions: Model predictions
        labels: True labels
    
    Returns:
        Accuracy percentage
    """
    correct = (predictions == labels).sum()
    total = len(labels)
    accuracy = (correct / total) * 100
    return accuracy


class ProgressTracker:
    """Track training progress."""
    
    def __init__(self):
        self.losses = []
        self.accuracies = []
    
    def update(self, loss, accuracy):
        """Add new values."""
        self.losses.append(loss)
        self.accuracies.append(accuracy)
    
    def get_average_loss(self):
        """Get average loss."""
        return np.mean(self.losses)
    
    def get_average_accuracy(self):
        """Get average accuracy."""
        return np.mean(self.accuracies)

