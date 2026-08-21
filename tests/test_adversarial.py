"""
tests/test_adversarial.py
Mentor Adversarial Test Suite: Edge Cases, Malformed Inputs, Extreme Resolutions, and Noise Robustness.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw
from src.model import build_cnn_model, compile_model
from src.predict import DigitPredictor
from src.preprocessing import preprocess_single_image


@pytest.fixture(scope="module")
def predictor():
    """Module-level predictor instance for adversarial testing."""
    model = build_cnn_model()
    model = compile_model(model)
    return DigitPredictor(model=model)


def test_adversarial_all_black_image(predictor):
    """Ensure all-black image (0s) processes cleanly as blank without crashing."""
    black_img = np.zeros((100, 100), dtype=np.uint8)
    res = predictor.predict(black_img)
    assert res["is_blank"] is True
    assert res["preprocessed_image"].shape == (28, 28)


def test_adversarial_all_white_image(predictor):
    """Ensure all-white image (255s) handles smart inversion and detects blank without NaN."""
    white_img = np.full((100, 100), 255, dtype=np.uint8)
    res = predictor.predict(white_img)
    assert res["is_blank"] is True
    assert not np.isnan(res["preprocessed_image"]).any()


def test_adversarial_random_noise(predictor):
    """Ensure pure random noise returns valid probabilities or blank without error."""
    noise_img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    res = predictor.predict(noise_img)
    assert res["preprocessed_image"].shape == (28, 28)
    assert np.isclose(sum(res["probabilities"]), 1.0, atol=1e-3)


def test_adversarial_high_resolution(predictor):
    """Ensure extreme resolution (1024x1024) with drawn stroke is downsampled safely."""
    large_img = Image.new("RGB", (1024, 1024), color=(240, 240, 240))
    draw = ImageDraw.Draw(large_img)
    draw.line([(200, 200), (800, 800)], fill="black", width=40)
    res = predictor.predict(large_img)
    assert res["preprocessed_image"].shape == (28, 28)
    assert not res["is_blank"]


def test_adversarial_tiny_resolution(predictor):
    """Ensure tiny 4x4 image is processed safely."""
    tiny_img = Image.new("L", (4, 4), color=255)
    res = predictor.predict(tiny_img)
    assert res["preprocessed_image"].shape == (28, 28)


def test_adversarial_extreme_aspect_ratio(predictor):
    """Ensure non-square panoramic image (1000x50) resizes without crashing."""
    pano_img = Image.new("L", (1000, 50), color=0)
    res = predictor.predict(pano_img)
    assert res["preprocessed_image"].shape == (28, 28)


def test_adversarial_rgba_alpha_composite(predictor):
    """Ensure RGBA transparent image with drawn stroke handles alpha flattening cleanly."""
    rgba_img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rgba_img)
    draw.line([(20, 20), (80, 80)], fill=(255, 255, 255, 255), width=10)
    res = predictor.predict(rgba_img)
    assert res["preprocessed_image"].shape == (28, 28)
    assert not res["is_blank"]


def test_adversarial_nonexistent_file():
    """Ensure passing nonexistent file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_single_image("non_existent_file_12345.png")


def test_adversarial_invalid_input_type():
    """Ensure passing unsupported types (e.g. integer or list) raises TypeError."""
    with pytest.raises(TypeError):
        preprocess_single_image(12345)
