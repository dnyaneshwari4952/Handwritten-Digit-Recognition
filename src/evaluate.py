"""
src/evaluate.py
Comprehensive Model Evaluation Suite: Classification Metrics, Per-Class Breakdown, and Confusion Analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from src.config import DEFAULT_PATHS, PathsConfig
from src.data_loader import load_mnist_raw
from src.preprocessing import preprocess_pipeline
from src.utils import load_json, save_json, setup_logger

logger = setup_logger("evaluate")


def evaluate_model(
    model: Optional[keras.Model] = None,
    model_path: Optional[Path] = None,
    paths_config: Optional[PathsConfig] = None
) -> Dict[str, Any]:
    """
    Evaluate trained MNIST CNN model on the test dataset.

    Computes:
        - Overall accuracy and cross-entropy loss
        - Per-class precision, recall, f1-score, and support
        - Macro and weighted average metrics
        - 10x10 Confusion matrix
        - Top confused digit pairs
        - Misclassified instances with prediction confidences

    Args:
        model: Optional pre-loaded keras Model.
        model_path: Optional path to saved .keras model.
        paths_config: Paths configuration object.

    Returns:
        Dictionary containing all evaluation metrics and metadata.
    """
    paths_config = paths_config or DEFAULT_PATHS

    # 1. Load Model
    if model is None:
        path = model_path or paths_config.model_save_path
        logger.info(f"Loading model from {path}...")
        model = keras.models.load_model(str(path))

    # 2. Load and Preprocess Test Data
    _, (x_test_raw, y_test) = load_mnist_raw()
    x_test = preprocess_pipeline(x_test_raw)

    # 3. Model Predictions
    logger.info("Computing predictions on test set (10,000 samples)...")
    y_prob = model.predict(x_test, batch_size=128, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_conf = np.max(y_prob, axis=1)

    # 4. Standard Evaluation
    eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=0)

    # 5. Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(10)), zero_division=0
    )

    per_class_metrics = {}
    for digit in range(10):
        per_class_metrics[str(digit)] = {
            "precision": float(precision[digit]),
            "recall": float(recall[digit]),
            "f1_score": float(f1[digit]),
            "support": int(support[digit]),
        }

    # Macro & Weighted Averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))

    # Find Top Confusion Pairs (excluding diagonal)
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    top_confusion_pairs: List[Dict[str, Any]] = []

    # Get sorted indices of highest confusion
    unraveled_indices = np.argsort(cm_off_diag.ravel())[::-1]
    for idx in unraveled_indices[:5]:  # Top 5 confusion pairs
        true_digit, pred_digit = np.unravel_index(idx, cm_off_diag.shape)
        count = int(cm_off_diag[true_digit, pred_digit])
        if count > 0:
            top_confusion_pairs.append({
                "true_digit": int(true_digit),
                "predicted_digit": int(pred_digit),
                "count": count,
            })

    # 7. Collect Misclassifications
    misclassified_indices = np.where(y_pred != y_test)[0]
    misclassifications: List[Dict[str, Any]] = []

    for idx in misclassified_indices[:50]:  # Capture first 50 misclassified instances
        misclassifications.append({
            "test_index": int(idx),
            "true_label": int(y_test[idx]),
            "predicted_label": int(y_pred[idx]),
            "confidence": float(y_conf[idx]),
        })

    # Compile Final Report
    report = {
        "test_samples": len(y_test),
        "test_loss": float(eval_loss),
        "test_accuracy": float(eval_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "total_misclassified": int(len(misclassified_indices)),
        "error_rate": float(len(misclassified_indices) / len(y_test)),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "top_confusion_pairs": top_confusion_pairs,
        "misclassifications_sample": misclassifications,
    }

    # Save to metrics directory
    save_json(report, paths_config.evaluation_json_path)
    logger.info(f"Evaluation report successfully saved to {paths_config.evaluation_json_path}")
    logger.info(
        f"Evaluation Summary -> Test Accuracy: {eval_acc * 100:.2f}%, "
        f"Misclassified: {len(misclassified_indices)}/{len(y_test)}"
    )

    return report


if __name__ == "__main__":
    evaluate_model()
