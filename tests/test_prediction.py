"""
tests/test_prediction.py
Unit and integration tests for DigitPredictor inference engine.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw
from src.model import build_cnn_model, compile_model
from src.predict import DigitPredictor


@pytest.fixture
def mock_predictor():
    """Create a predictor with initialized weights for fast unit testing."""
    model = build_cnn_model()
    model = compile_model(model)
    return DigitPredictor(model=model)


def test_predictor_predict_numpy(mock_predictor):
    """Verify inference on synthetic numpy image with drawn shape."""
    dummy_digit = np.zeros((100, 100), dtype=np.uint8)
    dummy_digit[20:80, 45:55] = 255  # vertical stroke
    res = mock_predictor.predict(dummy_digit, top_k=3)

    assert "predicted_digit" in res
    assert 0 <= res["predicted_digit"] <= 9
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_k"]) == 3
    assert len(res["probabilities"]) == 10
    assert res["preprocessed_image"].shape == (28, 28)
    assert not res["is_blank"]


def test_predictor_predict_blank_image(mock_predictor):
    """Verify predictor cleanly handles blank canvas."""
    blank_img = np.zeros((200, 200), dtype=np.uint8)
    res = mock_predictor.predict(blank_img, top_k=3)

    assert res["is_blank"] is True
    assert res["predicted_digit"] is None
    assert res["confidence"] == 0.0
    assert "No recognizable digit" in res["message"]


def test_predictor_predict_pil(mock_predictor):
    """Verify inference on PIL Image with drawn digit shape."""
    pil_img = Image.new("RGB", (64, 64), color="black")
    draw = ImageDraw.Draw(pil_img)
    draw.ellipse([15, 15, 50, 50], outline="white", width=6)
    res = mock_predictor.predict(pil_img, top_k=5)

    assert 0 <= res["predicted_digit"] <= 9
    assert len(res["top_k"]) == 5
    assert not res["is_blank"]


def test_predictor_batch_predict(mock_predictor):
    """Verify vectorized batch prediction."""
    batch = np.zeros((10, 28, 28, 1), dtype=np.float32)
    preds, confs = mock_predictor.predict_batch(batch)

    assert len(preds) == 10
    assert len(confs) == 10
    assert all(0 <= p <= 9 for p in preds)
