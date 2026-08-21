"""
src/model.py
Neural Network Architecture Builders (Production CNN & Baseline MLP) for MNIST Digit Recognition.
"""

from typing import Dict, Optional, Tuple
import keras
from keras import layers, models, optimizers
from src.config import DEFAULT_MODEL_CONFIG, ModelConfig
from src.utils import setup_logger

logger = setup_logger("model")


def build_cnn_model(config: Optional[ModelConfig] = None) -> keras.Model:
    """
    Construct the primary Convolutional Neural Network (CNN) for MNIST digit classification.

    Architecture:
        1. Input Layer: (28, 28, 1)
        2. Conv2D (32 filters, 3x3 kernel, ReLU, padding="same") -> (28, 28, 32)
        3. MaxPooling2D (2x2 pool) -> (14, 14, 32)
        4. Conv2D (64 filters, 3x3 kernel, ReLU, padding="same") -> (14, 14, 64)
        5. MaxPooling2D (2x2 pool) -> (7, 7, 64)
        6. Flatten -> (3136,)
        7. Dense (128 units, ReLU) -> (128,)
        8. Dropout (0.3)
        9. Dense (10 units, Softmax) -> (10,)

    Args:
        config: Optional ModelConfig dataclass.

    Returns:
        Uncompiled Keras Model instance.
    """
    if config is None:
        config = DEFAULT_MODEL_CONFIG

    logger.info(f"Building CNN Model with input shape {config.input_shape}...")

    model = models.Sequential([
        layers.Input(shape=config.input_shape, name="input_image"),

        # Block 1
        layers.Conv2D(
            filters=config.conv1_filters,
            kernel_size=config.conv1_kernel,
            padding="same",
            name="conv1"
        ),
        layers.BatchNormalization(name="bn1"),
        layers.Activation(config.activation, name="act1"),
        layers.MaxPooling2D(pool_size=config.pool_size, name="maxpool1"),

        # Block 2
        layers.Conv2D(
            filters=config.conv2_filters,
            kernel_size=config.conv2_kernel,
            padding="same",
            name="conv2"
        ),
        layers.BatchNormalization(name="bn2"),
        layers.Activation(config.activation, name="act2"),
        layers.MaxPooling2D(pool_size=config.pool_size, name="maxpool2"),

        # Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(units=config.dense_units, name="dense1"),
        layers.BatchNormalization(name="bn3"),
        layers.Activation(config.activation, name="act3"),
        layers.Dropout(rate=config.dropout_rate, name="dropout"),
        layers.Dense(units=config.num_classes, activation=config.output_activation, name="predictions"),
    ], name="MNIST_CNN_Classifier")

    return model


def build_baseline_model(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10
) -> keras.Model:
    """
    Construct a baseline Fully Connected (MLP) Neural Network for architectural comparison.

    Architecture:
        1. Input: (28, 28, 1)
        2. Flatten -> (784,)
        3. Dense (128 units, ReLU)
        4. Dense (64 units, ReLU)
        5. Dense (10 units, Softmax)

    Args:
        input_shape: Shape tuple.
        num_classes: Number of target categories.

    Returns:
        Uncompiled Keras Model instance.
    """
    logger.info(f"Building Baseline MLP Model with input shape {input_shape}...")

    model = models.Sequential([
        layers.Input(shape=input_shape, name="baseline_input"),
        layers.Flatten(name="baseline_flatten"),
        layers.Dense(128, activation="relu", name="baseline_dense1"),
        layers.Dense(64, activation="relu", name="baseline_dense2"),
        layers.Dense(num_classes, activation="softmax", name="baseline_predictions"),
    ], name="MNIST_Baseline_MLP")

    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    loss: str = "sparse_categorical_crossentropy",
    metrics: Optional[list] = None
) -> keras.Model:
    """
    Compile a Keras model with Adam optimizer and loss configuration.

    Args:
        model: Keras Model instance.
        learning_rate: Initial learning rate for Adam.
        loss: Loss function string or callable.
        metrics: List of evaluation metrics (default ["accuracy"]).

    Returns:
        Compiled Keras Model instance.
    """
    if metrics is None:
        metrics = ["accuracy"]

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info(f"Compiled model '{model.name}' with lr={learning_rate}, loss='{loss}'")
    return model


def count_parameters(model: keras.Model) -> Dict[str, int]:
    """
    Compute trainable, non-trainable, and total parameter counts.

    Args:
        model: Keras Model instance.

    Returns:
        Dictionary with parameter counts.
    """
    trainable = sum(w.shape.num_elements() for w in model.trainable_weights)
    non_trainable = sum(w.shape.num_elements() for w in model.non_trainable_weights)
    total = trainable + non_trainable

    return {
        "trainable_params": int(trainable),
        "non_trainable_params": int(non_trainable),
        "total_params": int(total),
    }


def get_model_summary_str(model: keras.Model) -> str:
    """
    Return model summary as a clean string.

    Args:
        model: Keras Model instance.

    Returns:
        Formatted summary string.
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)
