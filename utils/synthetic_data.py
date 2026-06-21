import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles

def generate_synthetic_data(n_samples=1000, cluster_std=3, centers=[[-3.0, 0.0], [3.0, 0.0]], random_state=42):
    """
    Generates a 2D synthetic dataset where natural spatial clusters
    conflict with the assigned labels.

    Args:
        n_samples (int): The number of samples to generate.
        random_state (int): The random state for reproducibility.

    Returns:
        tuple: A tuple containing the input data, task labels, and natural labels.
    """
    # 1. Generate natural spatial clusters (Left and Right blobs)
    # We use a large standard deviation so they overlap slightly,
    # but the unsupervised structure is clearly left vs. right.
    X, natural_labels = make_blobs(
        n_samples=n_samples,
        centers=centers, # Centers on the X-axis
        cluster_std=cluster_std,
        random_state=random_state
    )

    # 2. Assign task labels that completely ignore the natural clusters
    # Here, the task label is determined by the Y-axis (Top vs. Bottom)
    # Class 0: y > 0
    # Class 1: y <= 0
    task_labels = (X[:, 1] <= 0).astype(np.int64)

    return X, task_labels

def generate_complex_synthetic_data(n_samples=1000, noise=0.1, random_state=42):
    """
    Generates a non-linearly separable 2D synthetic dataset where the classes
    are intertwined. 

    Args:
        n_samples (int): The number of samples to generate.
        noise (float): Standard deviation of Gaussian noise added to the data.
        random_state (int): The random state for reproducibility.

    Returns:
        tuple: A tuple containing the input data (X) and task labels (y).
    """
    # Two intertwining half circles
    X, task_labels = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, task_labels