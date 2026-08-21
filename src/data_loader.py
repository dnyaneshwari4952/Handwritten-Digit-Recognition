"""
src/data_loader.py
Robust MNIST Data Loading, Dataset Integrity Checks, and Splitting.
"""

from typing import Dict, Tuple, Any
import numpy as np
from src.utils import setup_logger

logger = setup_logger("data_loader")


def load_mnist_test() -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast-load only the MNIST test dataset (10,000 samples).
    Uses cached .npz if available for instant sub-10ms startup.

    Returns:
        (x_test, y_test) as NumPy arrays.
    """
    from src.config import DATA_DIR
    test_npz_path = DATA_DIR / "mnist_test.npz"
    if test_npz_path.exists():
        with np.load(test_npz_path) as data:
            return data["x_test"], data["y_test"]

    from keras.datasets import mnist
    _, (x_test, y_test) = mnist.load_data()
    return x_test, y_test


def load_mnist_raw() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load raw MNIST dataset from Keras datasets.

    Returns:
        ((x_train, y_train), (x_test, y_test)) as NumPy arrays.
    """
    logger.info("Loading raw MNIST dataset from keras.datasets.mnist...")
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    logger.info(f"Loaded train: {x_train.shape}, test: {x_test.shape}")
    validate_dataset(x_train, y_train, x_test, y_test)
    return (x_train, y_train), (x_test, y_test)


def validate_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> bool:
    """
    Verify dataset dimensions, non-emptiness, label bounds, and pixel value ranges.

    Raises:
        ValueError: If any integrity constraint fails.

    Returns:
        True if all assertions pass.
    """
    # 1. Non-empty check
    if x_train.size == 0 or y_train.size == 0 or x_test.size == 0 or y_test.size == 0:
        raise ValueError("Dataset cannot contain empty arrays.")

    # 2. Shape checks
    if x_train.ndim != 3 or x_train.shape[1:] != (28, 28):
        raise ValueError(f"Expected x_train shape (N, 28, 28), got {x_train.shape}")
    if x_test.ndim != 3 or x_test.shape[1:] != (28, 28):
        raise ValueError(f"Expected x_test shape (N, 28, 28), got {x_test.shape}")

    # 3. Label matching
    if len(x_train) != len(y_train):
        raise ValueError(f"Train sample count mismatch: {len(x_train)} images vs {len(y_train)} labels.")
    if len(x_test) != len(y_test):
        raise ValueError(f"Test sample count mismatch: {len(x_test)} images vs {len(y_test)} labels.")

    # 4. Label bounds
    unique_train_labels = np.unique(y_train)
    unique_test_labels = np.unique(y_test)
    if not np.all(np.isin(unique_train_labels, range(10))):
        raise ValueError(f"Invalid train label values found: {unique_train_labels}")
    if not np.all(np.isin(unique_test_labels, range(10))):
        raise ValueError(f"Invalid test label values found: {unique_test_labels}")

    # 5. Pixel values range check
    if np.min(x_train) < 0 or np.max(x_train) > 255:
        raise ValueError(f"x_train pixel values out of bounds [0, 255]: min={np.min(x_train)}, max={np.max(x_train)}")
    if np.min(x_test) < 0 or np.max(x_test) > 255:
        raise ValueError(f"x_test pixel values out of bounds [0, 255]: min={np.min(x_test)}, max={np.max(x_test)}")

    logger.info("Dataset validation passed successfully.")
    return True


def get_dataset_summary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Generate statistical summary and class distributions for MNIST.

    Returns:
        Dictionary containing summary metadata.
    """
    train_dist = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    test_dist = {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}

    summary = {
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "image_dimensions": [int(x_train.shape[1]), int(x_train.shape[2])],
        "pixel_dtype": str(x_train.dtype),
        "pixel_min": float(np.min(x_train)),
        "pixel_max": float(np.max(x_train)),
        "train_class_distribution": train_dist,
        "test_class_distribution": test_dist,
    }
    return summary


def create_validation_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    val_split: float = 0.1,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training data into training and validation sets deterministically.

    Args:
        x_train: Training image array.
        y_train: Training label array.
        val_split: Fraction of training data for validation.
        random_seed: Seed for shuffling.

    Returns:
        (x_tr, y_tr, x_val, y_val)
    """
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"Validation split must be between 0.0 and 1.0, got {val_split}")

    num_samples = len(x_train)
    val_size = int(num_samples * val_split)

    rng = np.random.RandomState(random_seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    x_tr, y_tr = x_train[train_indices], y_train[train_indices]
    x_val, y_val = x_train[val_indices], y_train[val_indices]

    logger.info(f"Split train set ({num_samples}) -> Train: {len(x_tr)}, Validation: {len(x_val)}")
    return x_tr, y_tr, x_val, y_val
