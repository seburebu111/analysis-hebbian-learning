# Comparative Analysis of Internal Representations in Hebbian Learning Algorithms

> **Note:** This repository is part of the 2026 Research Project of the Computer Science and Engineering department at [TU Delft](https://github.com/TU-Delft-CSE).

## Abstract

This repository contains the official implementation and experiments for the accompanying research [paper](research_paper.pdf), evaluating the efficacy, biological plausibility, and computational efficiency of Hebbian-based learning paradigms compared to standard error backpropagation.

The backpropagation (BP) algorithm, though fundamental to modern deep learning, faces severe biological, computational, and physical limitations that hinder its applicability on energy-efficient neuromorphic systems. This has motivated the search for BP-free learning paradigms, with Hebbian-based algorithms being a notable alternative. Existing approaches range from purely local, unsupervised schemes such as SoftHebb, which naturally clusters data based on structural variance, to supervised, error-modulated approaches like PEPITA, which provides task-specific feedback through a second forward pass. However, the tradeoffs between these extremes remain largely unexplored. This study systematically compares the internal representations and hardware efficiencies of three Multi-Layer Perceptrons (MLP) trained with Backpropagation, PEPITA, and SoftHebb.

Our evaluation, utilizing geometric metrics such as Centered Kernel Alignment (CKA) and Principal Component Analysis (PCA), showcases how PEPITA exhibits higher similarity to BP, but is more geometrically aligned with the unsupervised SoftHebb. Furthermore, empirical hardware profiling exposes a significant implementation paradox: despite the theoretical efficiency of BP-free methods, high-level framework bottlenecks currently make algorithms like PEPITA computationally expensive on traditional digital architectures.

## Repository Structure

The project is structured to separate core learning algorithms from experimental evaluation and utility scripts:

```text
hebbian-coding/
│
├── data/                       # Datasets (MNIST, synthetic clusters)
├── experiments/                # Experimental setups and evaluations
│   ├── checkpoints/            # Model weight checkpoints
│   ├── renders/                # Rendered plots and figures
│   ├── run_benchmark.py        # Automated benchmarking script
│   ├── mnist_metrics.ipynb     # Evaluation metrics and visualizations on MNIST
│   ├── synthetic_baselines.ipynb # Baseline experiments on synthetic data
│   ├── synthetic_metrics.ipynb # Detailed metrics on synthetic data
│   └── performance_benchmark.ipynb # Performance overview and profiling
│
├── models/                     # Core neural network architectures
│   ├── baseline_bp.py          # Standard Multi-Layer Perceptron (Backpropagation)
│   ├── pepita.py               # PEPITA learning algorithm (Random Projections)
│   └── softhebb.py             # SoftHebb unsupervised learning (with Delta Rule Readout)
│
├── utils/                      # Auxiliary and helper functions
│   ├── data_utils.py           # Data loading and preprocessing routines
│   └── plots.py                # Plotting utilities for decision boundaries and learning curves
│
├── requirements.txt            # Python dependencies
└── README.md                   # Repository documentation
```

## Methodology

### 1. Baseline Backpropagation (BP)
The standard multi-layer perceptron trained via stochastic gradient descent with backpropagation. This serves as the upper-bound baseline for classification accuracy and the primary point of comparison for computational overhead.

### 2. PEPITA
PEPITA utilizes a forward pass combined with a modulated second pass to compute weight updates without relying on symmetric weight matrices (the weight transport problem). Error signals are modulated by a fixed random projection matrix (B), offering a more biologically realistic error assignment mechanism.

### 3. SoftHebb
SoftHebb introduces a purely local learning mechanism. Hidden layer weights are updated via an unsupervised, winner-take-all SoftHebb rule, allowing the network to form distinct feature representations without label information. A final linear classification layer is trained via the standard supervised Delta rule.

## Experimental Setup

The evaluations are divided into two primary phases:

1.  **Synthetic Data Experiments:**
    Models are trained on generated 2D clusters with varying degrees of linear separability and noise. This allows for clear visualizations of the learned decision boundaries and the trajectory of weight updates. Detailed in `synthetic_baselines.ipynb` and `synthetic_metrics.ipynb`.
2.  **MNIST Benchmark:**
    A scale-up evaluation using the standard MNIST handwritten digit dataset. The focus is on classification accuracy, convergence speed, and the properties of the learned hidden representations. Detailed in `mnist_metrics.ipynb`.

### Plots and Renders
To ensure reproducibility and clean organization, all generated figures (loss curves, decision boundaries, representation t-SNEs) are automatically saved to the `experiments/renders/` directory.

## Usage and Instructions

### Environment Setup
Install the necessary dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### Exploring Experiments
The Jupyter notebooks provide interactive environments to explore the metrics and generate visualizations.
- Start a Jupyter server: `jupyter notebook`
- Navigate to the `experiments/` directory and open the desired `.ipynb` file.

## References
*   Contributions and methodology details are outlined in the accompanying paper, found here: [paper](research_paper.pdf)