"""
tests/test_evaluation.py
Unit tests for evaluation metrics computation and report formatting.
"""

import numpy as np
import pytest
from src.model import build_cnn_model, compile_model


def test_evaluation_metric_shapes():
    """Verify evaluation metric calculations and confusion matrix structure."""
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    # Generate synthetic predictions
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 5)
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0] * 5)  # 9 is misclassified as 0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    assert cm.shape == (10, 10)
    assert cm[9, 0] == 5  # all digit 9s predicted as 0

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
