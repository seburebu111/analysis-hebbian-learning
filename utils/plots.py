import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs

custom_cmap = ListedColormap(['#1f77b4', '#ff7f0e'])

def plot_synthetic_data(X_train, y_train, X_test, y_test):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 12))

    # Plot 1: Training Data Clusters
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=custom_cmap, alpha=0.7, edgecolors='k')
    ax1.set_title("Training Data Clusters")
    ax1.set_xlabel("Feature 1 (X-axis)")
    ax1.set_ylabel("Feature 2 (Y-axis)")
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Plot 2: Test Data Clusters
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=custom_cmap, alpha=0.7, edgecolors='k')
    ax2.set_title("Test Data Clusters")
    ax2.set_xlabel("Feature 1 (X-axis)")
    ax2.set_ylabel("Feature 2 (Y-axis)")

    plt.suptitle("Synthetic Dataset")
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(model, X_train, y_train, X_test, y_test):
    # Set min and max values and give it some padding
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    h = 0.05 # grid step size
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        outputs = model(grid_tensor)
        _, Z = torch.max(outputs, 1)
        Z = Z.numpy().reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=custom_cmap, alpha=0.3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=custom_cmap, edgecolors='k', alpha=0.7)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=custom_cmap, edgecolors='k', alpha=0.7)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_loss(train_loss_history, test_loss_history):
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(test_loss_history, label="Test Loss")
    plt.title("Training and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_accuracy(train_acc_history, test_acc_history):
    plt.plot(train_acc_history, label="Training Accuracy")
    plt.plot(test_acc_history, label="Test Accuracy")
    plt.title("Training and Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

def plot_comparative_losses(losses, labels):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, loss, label in zip(axes, losses, labels):
        ax.plot(loss[0], label="Training Loss")
        ax.plot(loss[1], label="Test Loss")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    plt.suptitle("Comparative Losses")
    plt.tight_layout()
    plt.show()

def plot_comparative_accuracies(accuracies, labels):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, acc, label in zip(axes, accuracies, labels):
        ax.plot(acc[0], label="Training Accuracy")
        ax.plot(acc[1], label="Test Accuracy")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
    plt.suptitle("Comparative Accuracies")
    plt.tight_layout()
    plt.show()

def plot_comparative_boundaries(models, titles, X_train, y_train, X_test, y_test):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, model, title in zip(axes, models, titles):
        with torch.no_grad():
            outputs = model(grid_tensor)
            _, Z = torch.max(outputs, 1)
            Z = Z.numpy().reshape(xx.shape)
            
        ax.contourf(xx, yy, Z, cmap=custom_cmap, alpha=0.3)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=custom_cmap, edgecolors='k', alpha=0.7)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=custom_cmap, edgecolors='k', alpha=0.7)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Feature 1 (X-axis)")
        ax.set_ylabel("Feature 2 (Y-axis)")
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.suptitle("Comparative Decision Boundaries")
    plt.tight_layout()
    plt.show()
    
def plot_initial_dataset_boundary(X_train, y_train, X_test, y_test):
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = (yy <= 0).astype(int)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=custom_cmap, alpha=0.3)
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=custom_cmap, edgecolors='k', alpha=0.7, rasterized=True)
    
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    import os
    os.makedirs('renders', exist_ok=True)
    plt.savefig('renders/initial_dataset_boundary.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
