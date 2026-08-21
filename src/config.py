"""
src/config.py
Configuration and Hyperparameter Management for MNIST Digit Recognition.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# Base Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

# Ensure runtime directories exist
for directory in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture and input specification."""
    image_height: int = 28
    image_width: int = 28
    channels: int = 1
    num_classes: int = 10
    input_shape: Tuple[int, int, int] = (28, 28, 1)

    # CNN layer hyperparameters
    conv1_filters: int = 32
    conv1_kernel: Tuple[int, int] = (3, 3)
    conv2_filters: int = 64
    conv2_kernel: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    dense_units: int = 128
    dropout_rate: float = 0.3
    activation: str = "relu"
    output_activation: str = "softmax"


@dataclass
class TrainingConfig:
    """Hyperparameters for model training."""
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    validation_split: float = 0.1
    random_seed: int = 42
    optimizer_name: str = "adam"
    loss_function: str = "sparse_categorical_crossentropy"
    metrics: list = field(default_factory=lambda: ["accuracy"])

    # Callback parameters
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6


@dataclass
class PathsConfig:
    """Paths to artifacts and saved assets."""
    project_root: Path = PROJECT_ROOT
    artifacts_dir: Path = ARTIFACTS_DIR
    models_dir: Path = MODELS_DIR
    plots_dir: Path = PLOTS_DIR
    metrics_dir: Path = METRICS_DIR
    predictions_dir: Path = PREDICTIONS_DIR

    # Specific file paths
    model_save_path: Path = MODELS_DIR / "mnist_cnn.keras"
    baseline_model_save_path: Path = MODELS_DIR / "mnist_baseline_mlp.keras"
    best_weights_path: Path = MODELS_DIR / "best_weights.weights.h5"
    history_json_path: Path = METRICS_DIR / "training_history.json"
    evaluation_json_path: Path = METRICS_DIR / "evaluation_metrics.json"
    baseline_metrics_json_path: Path = METRICS_DIR / "baseline_metrics.json"

    # Plot paths
    loss_curve_path: Path = PLOTS_DIR / "loss_curves.png"
    accuracy_curve_path: Path = PLOTS_DIR / "accuracy_curves.png"
    combined_training_plot_path: Path = PLOTS_DIR / "training_history.png"
    confusion_matrix_path: Path = PLOTS_DIR / "confusion_matrix.png"
    misclassifications_path: Path = PLOTS_DIR / "misclassified_examples.png"
    sample_grid_path: Path = PLOTS_DIR / "sample_digits_grid.png"


# Global instances for convenience
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_PATHS = PathsConfig()
