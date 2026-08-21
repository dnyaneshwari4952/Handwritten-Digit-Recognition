"""
tests/test_data.py
Unit tests for data loading, validation, and splitting.
"""

import numpy as np
import pytest
from src.data_loader import create_validation_split, get_dataset_summary, load_mnist_raw, validate_dataset


def test_validate_dataset_valid():
    """Verify that a synthetically valid dataset passes validation."""
    x_train = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
    y_train = np.random.randint(0, 10, (100,), dtype=np.uint8)
    x_test = np.random.randint(0, 256, (20, 28, 28), dtype=np.uint8)
    y_test = np.random.randint(0, 10, (20,), dtype=np.uint8)

    assert validate_dataset(x_train, y_train, x_test, y_test) is True


def test_validate_dataset_empty_fails():
    """Verify that empty arrays trigger a ValueError."""
    x_empty = np.empty((0, 28, 28))
    y_empty = np.empty((0,))
    x_valid = np.random.randint(0, 256, (10, 28, 28))
    y_valid = np.random.randint(0, 10, (10,))

    with pytest.raises(ValueError, match="empty arrays"):
        validate_dataset(x_empty, y_empty, x_valid, y_valid)


def test_validate_dataset_invalid_shape():
    """Verify that invalid shapes trigger a ValueError."""
    x_invalid = np.random.randint(0, 256, (10, 32, 32))  # Not 28x28
    y_invalid = np.random.randint(0, 10, (10,))
    x_valid = np.random.randint(0, 256, (10, 28, 28))
    y_valid = np.random.randint(0, 10, (10,))

    with pytest.raises(ValueError, match="Expected x_train shape"):
        validate_dataset(x_invalid, y_invalid, x_valid, y_valid)


def test_validate_dataset_invalid_labels():
    """Verify that out-of-range labels trigger a ValueError."""
    x_valid = np.random.randint(0, 256, (10, 28, 28))
    y_invalid = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 15])  # 15 is invalid
    y_valid = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    with pytest.raises(ValueError, match="Invalid train label"):
        validate_dataset(x_valid, y_invalid, x_valid, y_valid)


def test_create_validation_split():
    """Verify train/validation split maintains counts and determinism."""
    x_train = np.random.randint(0, 256, (1000, 28, 28))
    y_train = np.random.randint(0, 10, (1000,))

    x_tr, y_tr, x_val, y_val = create_validation_split(x_train, y_train, val_split=0.2, random_seed=42)

    assert len(x_tr) == 800
    assert len(y_tr) == 800
    assert len(x_val) == 200
    assert len(y_val) == 200

    # Test determinism
    x_tr2, _, x_val2, _ = create_validation_split(x_train, y_train, val_split=0.2, random_seed=42)
    np.testing.assert_array_equal(x_tr, x_tr2)
    np.testing.assert_array_equal(x_val, x_val2)


def test_get_dataset_summary():
    """Verify dataset summary metadata dictionary."""
    x_train = np.zeros((100, 28, 28), dtype=np.uint8)
    y_train = np.zeros((100,), dtype=np.uint8)
    x_test = np.zeros((20, 28, 28), dtype=np.uint8)
    y_test = np.zeros((20,), dtype=np.uint8)

    summary = get_dataset_summary(x_train, y_train, x_test, y_test)
    assert summary["train_samples"] == 100
    assert summary["test_samples"] == 20
    assert summary["image_dimensions"] == [28, 28]
