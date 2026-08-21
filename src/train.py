"""
src/train.py
Model Training Pipeline with Checkpoints, Early Stopping, Learning Rate Scheduling, and Metrics Serialization.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import keras
from keras import callbacks
import numpy as np

from src.config import DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG, DEFAULT_PATHS, ModelConfig, TrainingConfig, PathsConfig
from src.data_loader import create_validation_split, load_mnist_raw
from src.model import build_baseline_model, build_cnn_model, compile_model
from src.preprocessing import preprocess_pipeline
from src.utils import get_device_info, save_json, set_seed, setup_logger, timeit

logger = setup_logger("train")


def train_model(
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    paths_config: Optional[PathsConfig] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: int = 1
) -> Tuple[keras.Model, Dict[str, Any], Dict[str, float]]:
    """
    Execute full training pipeline for MNIST CNN.

    Args:
        model_config: Model architecture settings.
        training_config: Training hyperparameters.
        paths_config: Artifact destination paths.
        epochs: Optional epoch override.
        batch_size: Optional batch size override.
        verbose: Verbosity level for model.fit.

    Returns:
        (trained_model, history_dict, test_evaluation_metrics)
    """
    import copy
    model_config = model_config or DEFAULT_MODEL_CONFIG
    training_config = copy.copy(training_config or DEFAULT_TRAINING_CONFIG)
    paths_config = paths_config or DEFAULT_PATHS

    if epochs is not None:
        training_config.epochs = epochs
    if batch_size is not None:
        training_config.batch_size = batch_size

    # 1. Setup reproducibility & hardware
    set_seed(training_config.random_seed)
    device_info = get_device_info()
    logger.info(f"Training initialized on device: {device_info['device']} (Python {device_info['python_version']})")

    # 2. Load and Preprocess Data
    (x_train_raw, y_train), (x_test_raw, y_test) = load_mnist_raw()

    logger.info("Preprocessing train and test images...")
    x_train_all = preprocess_pipeline(x_train_raw)
    x_test = preprocess_pipeline(x_test_raw)

    # 3. Create Validation Split
    x_train, y_train, x_val, y_val = create_validation_split(
        x_train=x_train_all,
        y_train=y_train,
        val_split=training_config.validation_split,
        random_seed=training_config.random_seed
    )

    # 4. Build and Compile Model
    model = build_cnn_model(config=model_config)
    model = compile_model(
        model=model,
        learning_rate=training_config.learning_rate,
        loss=training_config.loss_function,
        metrics=training_config.metrics
    )

    # 5. Setup Callbacks
    paths_config.models_dir.mkdir(parents=True, exist_ok=True)
    paths_config.metrics_dir.mkdir(parents=True, exist_ok=True)

    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=str(paths_config.model_save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min"
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training_config.early_stopping_patience,
            min_delta=training_config.early_stopping_min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=training_config.reduce_lr_factor,
            patience=training_config.reduce_lr_patience,
            min_lr=training_config.min_learning_rate,
            verbose=1
        ),
    ]

    # 6. Setup Data Augmentation Pipeline
    try:
        import tensorflow as tf
        aug_model = keras.Sequential([
            keras.layers.RandomRotation(0.06, fill_mode="constant", fill_value=0.0),
            keras.layers.RandomTranslation(0.06, 0.06, fill_mode="constant", fill_value=0.0),
            keras.layers.RandomZoom((-0.06, 0.06), fill_mode="constant", fill_value=0.0),
        ], name="data_augmentation")

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.shuffle(buffer_size=min(10000, len(x_train)), seed=training_config.random_seed)
        train_ds = train_ds.batch(training_config.batch_size)
        train_ds = train_ds.map(lambda x, y: (aug_model(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(training_config.batch_size)
    except Exception as e:
        logger.warning(f"Could not initialize tf.data augmentation pipeline ({e}); training on raw arrays.")
        train_ds = None

    # 7. Model Training
    logger.info(
        f"Starting training: {training_config.epochs} epochs, "
        f"batch_size={training_config.batch_size}, "
        f"initial_lr={training_config.learning_rate}"
    )

    with timeit("Model Training"):
        if train_ds is not None:
            history = model.fit(
                train_ds,
                epochs=training_config.epochs,
                validation_data=val_ds,
                callbacks=callback_list,
                verbose=verbose
            )
        else:
            history = model.fit(
                x=x_train,
                y=y_train,
                batch_size=training_config.batch_size,
                epochs=training_config.epochs,
                validation_data=(x_val, y_val),
                callbacks=callback_list,
                verbose=verbose
            )

    # Ensure best model is saved
    model.save(str(paths_config.model_save_path))
    logger.info(f"Model successfully saved to {paths_config.model_save_path}")

    # 7. Serialize Training History
    history_dict = {
        key: [float(val) for val in values]
        for key, values in history.history.items()
    }
    history_dict["epochs_trained"] = len(history_dict.get("loss", []))
    history_dict["hyperparameters"] = {
        "batch_size": training_config.batch_size,
        "epochs": training_config.epochs,
        "learning_rate": training_config.learning_rate,
        "validation_split": training_config.validation_split,
        "random_seed": training_config.random_seed,
        "dropout_rate": model_config.dropout_rate,
    }

    save_json(history_dict, paths_config.history_json_path)
    logger.info(f"Training history saved to {paths_config.history_json_path}")

    # 8. Evaluate on Unseen Test Data
    test_results = model.evaluate(x_test, y_test, verbose=0)
    test_metrics = {
        "test_loss": float(test_results[0]),
        "test_accuracy": float(test_results[1]),
    }
    logger.info(
        f"Test Results -> Loss: {test_metrics['test_loss']:.4f}, "
        f"Accuracy: {test_metrics['test_accuracy'] * 100:.2f}%"
    )

    return model, history_dict, test_metrics


def train_baseline(
    paths_config: Optional[PathsConfig] = None,
    epochs: int = 8,
    batch_size: int = 64
) -> Tuple[keras.Model, Dict[str, float]]:
    """
    Train a baseline MLP model for comparative analysis against the CNN.

    Args:
        paths_config: Paths configuration.
        epochs: Number of epochs.
        batch_size: Batch size.

    Returns:
        (trained_baseline_model, baseline_test_metrics)
    """
    paths_config = paths_config or DEFAULT_PATHS
    set_seed(42)

    logger.info("Training baseline MLP model for comparison...")
    (x_train_raw, y_train), (x_test_raw, y_test) = load_mnist_raw()

    x_train = preprocess_pipeline(x_train_raw)
    x_test = preprocess_pipeline(x_test_raw)

    baseline_model = build_baseline_model()
    baseline_model = compile_model(baseline_model, learning_rate=0.001)

    baseline_model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    baseline_model.save(str(paths_config.baseline_model_save_path))
    eval_res = baseline_model.evaluate(x_test, y_test, verbose=0)
    metrics = {
        "baseline_loss": float(eval_res[0]),
        "baseline_accuracy": float(eval_res[1]),
    }
    save_json(metrics, paths_config.baseline_metrics_json_path)
    logger.info(f"Baseline MLP Test Accuracy: {metrics['baseline_accuracy'] * 100:.2f}%")
    return baseline_model, metrics


if __name__ == "__main__":
    train_model()
