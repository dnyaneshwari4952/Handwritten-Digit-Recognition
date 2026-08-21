"""
tests/test_preprocessing.py
Unit tests for image normalization, reshaping, bounding-box cropping, aspect-ratio scaling,
center-of-mass centering, and blank canvas detection.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.preprocessing import (
    extract_preprocessing_stages,
    normalize_images,
    preprocess_external_digit,
    preprocess_pipeline,
    preprocess_single_image,
    reshape_for_cnn,
    to_one_hot,
)


def test_normalize_images_uint8():
    """Verify uint8 normalization into [0.0, 1.0] float32."""
    raw = np.array([0, 127.5, 255], dtype=np.float32)
    norm = normalize_images(raw)
    assert norm.dtype == np.float32
    assert np.isclose(norm[0], 0.0)
    assert np.isclose(norm[1], 0.5, atol=1e-3)
    assert np.isclose(norm[2], 1.0)


def test_normalize_images_nan_inf_rejected():
    """Verify that NaNs and Infs raise ValueError."""
    arr_nan = np.array([0, np.nan, 255])
    with pytest.raises(ValueError, match="contain NaN or Inf"):
        normalize_images(arr_nan)

    arr_inf = np.array([0, np.inf, 255])
    with pytest.raises(ValueError, match="contain NaN or Inf"):
        normalize_images(arr_inf)


def test_reshape_for_cnn():
    """Verify dimension expansion for various valid shapes."""
    single_2d = np.zeros((28, 28))
    assert reshape_for_cnn(single_2d).shape == (1, 28, 28, 1)

    batch_3d = np.zeros((50, 28, 28))
    assert reshape_for_cnn(batch_3d).shape == (50, 28, 28, 1)

    batch_4d = np.zeros((50, 28, 28, 1))
    assert reshape_for_cnn(batch_4d).shape == (50, 28, 28, 1)


def test_reshape_for_cnn_invalid_shape():
    """Verify invalid shapes raise ValueError."""
    wrong_dim = np.zeros((10, 30, 30))
    with pytest.raises(ValueError, match="Unexpected 3D image shape"):
        reshape_for_cnn(wrong_dim)


def test_preprocess_single_image_pil():
    """Verify preprocessing of PIL Image inputs with bounding box and centering."""
    white_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(white_img)
    draw.line([(50, 50), (150, 150)], fill="black", width=12)

    processed, meta = preprocess_single_image(white_img, auto_invert=True, return_metadata=True)

    assert processed.shape == (1, 28, 28, 1)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0 and processed.max() <= 1.0
    assert not meta["is_blank"]
    assert meta["digit_size"][0] > 0 and meta["digit_size"][1] > 0


def test_preprocess_blank_canvas_detection():
    """Verify that empty/blank canvas is detected correctly."""
    blank_canvas = Image.new("RGBA", (280, 280), (0, 0, 0, 255))
    tensor, meta = preprocess_single_image(blank_canvas, return_metadata=True)

    assert meta["is_blank"] is True
    assert np.all(tensor == 0.0)


def test_preprocess_aspect_ratio_preservation():
    """Verify non-square rectangular input maintains aspect ratio inside 20x20 box."""
    rect_img = Image.new("RGB", (400, 200), (255, 255, 255))
    draw = ImageDraw.Draw(rect_img)
    draw.line([(50, 100), (350, 100)], fill="black", width=10)

    tensor, meta = preprocess_single_image(rect_img, return_metadata=True)
    assert not meta["is_blank"]
    scaled_w, scaled_h = meta["scaled_size"]
    assert scaled_w <= 20 and scaled_h <= 20
    assert scaled_w > scaled_h


def test_extract_preprocessing_stages():
    """Verify intermediate diagnostic stages are extracted cleanly."""
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 70, 70], outline="white", width=6)

    stages = extract_preprocessing_stages(img)
    assert "original" in stages
    assert "grayscale" in stages
    assert "polarity_corrected" in stages
    assert "bbox_overlay" in stages
    assert "cropped_digit" in stages
    assert "aspect_preserved_20x20" in stages
    assert "centered_28x28" in stages
    assert "final_tensor" in stages
    assert stages["final_tensor"].shape == (1, 28, 28, 1)


def test_to_one_hot():
    """Verify one-hot encoding logic."""
    labels = np.array([0, 3, 9])
    one_hot = to_one_hot(labels, num_classes=10)
    assert one_hot.shape == (3, 10)
    assert np.array_equal(one_hot[0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(one_hot[1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(one_hot[2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
