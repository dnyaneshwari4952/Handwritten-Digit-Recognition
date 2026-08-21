"""
tests/test_model.py
Unit tests for CNN and Baseline MLP model builders, parameter counting, and compilation.
"""

import numpy as np
import pytest
from src.config import ModelConfig
from src.model import build_baseline_model, build_cnn_model, compile_model, count_parameters, get_model_summary_str


def test_build_cnn_model():
    """Verify CNN model layers, input shape, and output shape."""
    cfg = ModelConfig()
    model = build_cnn_model(cfg)

    assert model.name == "MNIST_CNN_Classifier"
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)

    # Verify forward pass on dummy batch
    dummy_input = np.zeros((4, 28, 28, 1), dtype=np.float32)
    output = model(dummy_input)
    assert output.shape == (4, 10)
    # Check softmax sum is ~1.0
    assert np.allclose(np.sum(output.numpy(), axis=1), 1.0, atol=1e-5)


def test_build_baseline_model():
    """Verify Baseline MLP model layers and output shape."""
    model = build_baseline_model()
    assert model.name == "MNIST_Baseline_MLP"
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)

    dummy_input = np.zeros((2, 28, 28, 1), dtype=np.float32)
    output = model(dummy_input)
    assert output.shape == (2, 10)


def test_compile_model():
    """Verify compilation sets optimizer and loss properly."""
    model = build_cnn_model()
    compiled = compile_model(model, learning_rate=0.005)
    assert compiled.optimizer is not None


def test_count_parameters():
    """Verify parameter count helper accurately computes weights."""
    model = build_cnn_model()
    params = count_parameters(model)
    assert "trainable_params" in params
    assert "total_params" in params
    assert params["trainable_params"] > 10000
    assert params["total_params"] == params["trainable_params"] + params["non_trainable_params"]


def test_get_model_summary_str():
    """Verify model summary formatting."""
    model = build_cnn_model()
    summary_str = get_model_summary_str(model)
    assert "MNIST_CNN_Classifier" in summary_str
    assert "conv1" in summary_str
    assert "predictions" in summary_str
