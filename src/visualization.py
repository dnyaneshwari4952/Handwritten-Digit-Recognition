"""
src/visualization.py
Publication-Grade Visualizations: Training Dynamics, Confusion Matrix Heatmaps, and Misclassification Grids.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless and script execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import DEFAULT_PATHS, PathsConfig
from src.data_loader import load_mnist_raw
from src.preprocessing import preprocess_pipeline
from src.utils import load_json, setup_logger

logger = setup_logger("visualization")

# Style configurations
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 150,
})


def plot_training_history(
    history: Dict[str, Any],
    save_path: Optional[Path] = None
) -> Path:
    """
    Plot training and validation Loss and Accuracy curves over epochs.

    Args:
        history: Dictionary containing 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
        save_path: Path to save the PNG file.

    Returns:
        Path to the saved figure.
    """
    save_path = Path(save_path or DEFAULT_PATHS.combined_training_plot_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

    # 1. Loss Plot
    ax1.plot(epochs, history.get("loss", []), "o-", color="#2563EB", label="Train Loss", linewidth=2, markersize=5)
    if "val_loss" in history:
        ax1.plot(epochs, history.get("val_loss", []), "s--", color="#DC2626", label="Val Loss", linewidth=2, markersize=5)
    ax1.set_title("Cross-Entropy Loss vs Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")

    # 2. Accuracy Plot
    ax2.plot(epochs, [acc * 100 for acc in history.get("accuracy", [])], "o-", color="#16A34A", label="Train Acc (%)", linewidth=2, markersize=5)
    if "val_accuracy" in history:
        ax2.plot(epochs, [acc * 100 for acc in history.get("val_accuracy", [])], "s--", color="#9333EA", label="Val Acc (%)", linewidth=2, markersize=5)
    ax2.set_title("Classification Accuracy vs Epochs", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training history plot saved to: {save_path}")
    return save_path


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[Path] = None,
    normalize: bool = False
) -> Path:
    """
    Generate an annotated heatmap of the digit confusion matrix.

    Args:
        cm: 10x10 confusion matrix array.
        save_path: Destination path.
        normalize: If True, normalize row-wise to display percentages.

    Returns:
        Path to the saved figure.
    """
    save_path = Path(save_path or DEFAULT_PATHS.confusion_matrix_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm, dtype=float)
    if normalize:
        cm_arr = cm_arr / cm_arr.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized Confusion Matrix (Row %)"
    else:
        fmt = "d"
        cm_arr = cm_arr.astype(int)
        title = "MNIST Digit Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
        linewidths=0.5,
        linecolor="#E2E8F0",
        ax=ax
    )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Digit", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Digit", fontweight="bold", labelpad=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved to: {save_path}")
    return save_path


def plot_sample_digits(
    images: np.ndarray,
    labels: np.ndarray,
    num_samples: int = 16,
    save_path: Optional[Path] = None
) -> Path:
    """
    Plot a grid of sample MNIST digits with their ground-truth labels.

    Args:
        images: Array of images (N, 28, 28) or (N, 28, 28, 1).
        labels: Corresponding true labels (N,).
        num_samples: Number of images to display (default 16, 4x4).
        save_path: Destination path.

    Returns:
        Path to saved figure.
    """
    save_path = Path(save_path or DEFAULT_PATHS.sample_grid_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cols = 4
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(8, 8), dpi=200)
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].squeeze()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Digit: {labels[i]}", fontsize=10, fontweight="bold")
        axes[i].axis("off")

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle("MNIST Sample Handwritten Digits", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Sample digits grid saved to: {save_path}")
    return save_path


def plot_misclassified_examples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
    num_samples: int = 12,
    save_path: Optional[Path] = None
) -> Path:
    """
    Plot a grid of misclassified test examples with true labels, predicted labels, and confidence.

    Args:
        images: Test images array.
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        y_conf: Prediction confidence probabilities.
        num_samples: Number of examples to show.
        save_path: Destination path.

    Returns:
        Path to saved figure.
    """
    save_path = Path(save_path or DEFAULT_PATHS.misclassifications_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        logger.warning("No misclassified examples found to plot!")
        return save_path

    num_to_plot = min(num_samples, len(mis_idx))
    cols = 4
    rows = int(np.ceil(num_to_plot / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows), dpi=200)
    axes = np.array(axes).flatten()

    for i in range(num_to_plot):
        idx = mis_idx[i]
        img = images[idx].squeeze()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"True: {y_true[idx]} | Pred: {y_pred[idx]}\nConf: {y_conf[idx] * 100:.1f}%",
            fontsize=9,
            color="#DC2626",
            fontweight="bold"
        )
        axes[i].axis("off")

    for j in range(num_to_plot, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Misclassified Test Examples (Error Analysis)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Misclassified examples plot saved to: {save_path}")
    return save_path


def generate_all_plots(paths_config: Optional[PathsConfig] = None) -> None:
    """
    Generate all visualization assets from trained model history and test data.
    """
    import keras
    paths_config = paths_config or DEFAULT_PATHS

    # 1. Plot History if available
    if paths_config.history_json_path.exists():
        history = load_json(paths_config.history_json_path)
        plot_training_history(history, paths_config.combined_training_plot_path)

    # 2. Plot Sample Grid & Confusion Matrix
    (x_train, y_train), (x_test_raw, y_test) = load_mnist_raw()
    plot_sample_digits(x_train, y_train, num_samples=16, save_path=paths_config.sample_grid_path)

    if paths_config.model_save_path.exists():
        model = keras.models.load_model(str(paths_config.model_save_path))
        x_test = preprocess_pipeline(x_test_raw)
        y_prob = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_conf = np.max(y_prob, axis=1)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
        plot_confusion_matrix(cm, paths_config.confusion_matrix_path)
        plot_misclassified_examples(x_test_raw, y_test, y_pred, y_conf, num_samples=12, save_path=paths_config.misclassifications_path)


if __name__ == "__main__":
    generate_all_plots()
