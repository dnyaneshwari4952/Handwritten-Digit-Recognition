"""
tests/test_regression.py
Regression tests to verify robustness of custom digit recognitions (2, 7, 8, 9, etc.)
and prevent regression of canvas and upload pipeline fixes.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.config import DEFAULT_PATHS
from src.predict import DigitPredictor
from src.preprocessing import preprocess_single_image


@pytest.fixture(scope="module")
def real_predictor():
    """Load the actual trained CNN model for regression validation."""
    if not DEFAULT_PATHS.model_save_path.exists():
        pytest.skip("Trained model checkpoint not found; skipping regression tests.")
    return DigitPredictor(model_path=DEFAULT_PATHS.model_save_path)


def test_regression_digit_8_canvas_variations(real_predictor):
    """
    Verify that an 8 drawn on canvas with various shifts and stroke thicknesses
    correctly classifies as 8 (fixing the previous 8 -> 9 misclassification).
    """
    for shift_x in [-20, 0, 20]:
        for shift_y in [-20, 0, 20]:
            canvas = Image.new("RGBA", (280, 280), (0, 0, 0, 255))
            draw = ImageDraw.Draw(canvas)
            # Top loop
            draw.ellipse([100 + shift_x, 60 + shift_y, 180 + shift_x, 140 + shift_y], outline="white", width=16)
            # Bottom loop
            draw.ellipse([95 + shift_x, 130 + shift_y, 185 + shift_x, 220 + shift_y], outline="white", width=16)

            res = real_predictor.predict(np.array(canvas))
            assert res["predicted_digit"] == 8, f"Failed for shift ({shift_x}, {shift_y}): got {res['predicted_digit']}"
            assert res["confidence"] > 0.70


def test_regression_digit_2_upload_variations(real_predictor):
    """
    Verify that an uploaded image of digit 2 across various sizes, aspect ratios,
    and positions correctly classifies as 2 (fixing the previous 2 -> 7/3 misclassification).
    """
    for scale in [0.5, 0.8, 1.0]:
        for offset_y in [60, 100, 140]:
            img = Image.new("RGB", (400, 400), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            w = int(120 * scale)
            h = int(180 * scale)
            x0 = 140
            y0 = offset_y
            draw.arc([x0, y0, x0 + w, y0 + h // 2], start=180, end=0, fill="black", width=max(4, int(14 * scale)))
            draw.line([(x0 + w, y0 + h // 4), (x0, y0 + h)], fill="black", width=max(4, int(14 * scale)))
            draw.line([(x0, y0 + h), (x0 + w + int(20 * scale), y0 + h)], fill="black", width=max(4, int(14 * scale)))

            res = real_predictor.predict(img, auto_invert=True)
            assert res["predicted_digit"] == 2, f"Failed for scale {scale}, offset {offset_y}: got {res['predicted_digit']}"


def test_regression_digit_6_thin_pen_upload(real_predictor):
    """
    Verify that thin blue/black pen strokes of digit 6 on white/off-white paper
    correctly classify as 6 (fixing the 6 -> 0/8 misclassification).
    """
    for bg_color in [(255, 255, 255), (235, 238, 242)]:
        for pen_width in [3, 5, 8]:
            for stem_tilt in [-20, 0, 20]:
                img = Image.new("RGB", (400, 400), bg_color)
                draw = ImageDraw.Draw(img)
                # Natural handwritten digit 6 with stem and loop
                draw.line([(240 + stem_tilt, 60), (180, 200), (160, 280)], fill=(25, 60, 150), width=pen_width)
                draw.ellipse([150, 180, 290, 340], outline=(25, 60, 150), width=pen_width)

                res = real_predictor.predict(img, auto_invert=True)
                assert res["predicted_digit"] == 6, (
                    f"Digit 6 misclassified as {res['predicted_digit']} "
                    f"(bg={bg_color}, pen_width={pen_width}, tilt={stem_tilt})"
                )
                assert res["confidence"] >= 0.40


def test_regression_all_digits_synthetic_drawings(real_predictor):
    """
    Verify generalization across all 10 digits (0 through 9).
    """
    def make_digit(digit):
        img = Image.new("RGB", (280, 280), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        w = 16
        if digit == 0:
            draw.ellipse([60, 40, 220, 240], outline="black", width=w)
        elif digit == 1:
            draw.line([(140, 40), (140, 240)], fill="black", width=w)
            draw.line([(100, 70), (140, 40)], fill="black", width=w)
            draw.line([(100, 240), (180, 240)], fill="black", width=w)
        elif digit == 2:
            draw.arc([70, 40, 210, 150], 180, 0, fill="black", width=w)
            draw.line([(210, 95), (70, 240)], fill="black", width=w)
            draw.line([(70, 240), (210, 240)], fill="black", width=w)
        elif digit == 3:
            draw.arc([80, 40, 200, 140], 270, 90, fill="black", width=w)
            draw.arc([80, 130, 200, 240], 270, 90, fill="black", width=w)
        elif digit == 4:
            draw.line([(180, 40), (70, 170)], fill="black", width=w)
            draw.line([(70, 170), (220, 170)], fill="black", width=w)
            draw.line([(180, 40), (180, 240)], fill="black", width=w)
        elif digit == 5:
            draw.line([(200, 40), (80, 40)], fill="black", width=w)
            draw.line([(80, 40), (80, 130)], fill="black", width=w)
            draw.arc([70, 120, 210, 240], 270, 90, fill="black", width=w)
        elif digit == 6:
            draw.arc([80, 40, 180, 240], 90, 270, fill="black", width=w)
            draw.ellipse([80, 120, 200, 240], outline="black", width=w)
        elif digit == 7:
            draw.line([(70, 40), (210, 40)], fill="black", width=w)
            draw.line([(210, 40), (110, 240)], fill="black", width=w)
        elif digit == 8:
            draw.ellipse([85, 40, 195, 135], outline="black", width=w)
            draw.ellipse([75, 130, 205, 240], outline="black", width=w)
        elif digit == 9:
            draw.ellipse([80, 40, 200, 150], outline="black", width=w)
            draw.arc([100, 40, 200, 240], 270, 90, fill="black", width=w)
        return img

    for d in range(10):
        img = make_digit(d)
        res = real_predictor.predict(img)
        assert res["predicted_digit"] == d, f"Digit {d} misclassified as {res['predicted_digit']}"
        assert res["confidence"] >= 0.50


def test_regression_multi_digit_recognition_42(real_predictor):
    """
    Verify automatic segmentation and recognition of multi-digit number '42'.
    """
    img = Image.new("RGB", (600, 300), (240, 242, 245))
    draw = ImageDraw.Draw(img)
    # Digit 4
    draw.line([(150, 50), (90, 180)], fill=(30, 60, 150), width=6)
    draw.line([(90, 180), (220, 180)], fill=(30, 60, 150), width=6)
    draw.line([(170, 50), (170, 250)], fill=(30, 60, 150), width=6)
    # Digit 2
    draw.arc([(340, 50), (480, 160)], 180, 0, fill=(30, 60, 150), width=6)
    draw.line([(480, 105), (340, 250)], fill=(30, 60, 150), width=6)
    draw.line([(340, 250), (480, 250)], fill=(30, 60, 150), width=6)

    res = real_predictor.predict(img)
    assert res["is_multi_digit"] is True
    assert res["full_number"] == "42"
    assert len(res["digits"]) == 2
    assert res["digits"][0]["predicted_digit"] == 4
    assert res["digits"][1]["predicted_digit"] == 2


def test_regression_multi_digit_recognition_789(real_predictor):
    """
    Verify automatic segmentation and recognition of 3-digit number '789'.
    """
    img = Image.new("RGB", (900, 300), (250, 250, 250))
    d = ImageDraw.Draw(img)
    # 7
    d.line([(80, 50), (220, 50)], fill="black", width=6)
    d.line([(220, 50), (120, 250)], fill="black", width=6)
    # 8
    d.ellipse([(360, 50), (480, 150)], outline="black", width=6)
    d.ellipse([(350, 140), (490, 250)], outline="black", width=6)
    # 9
    d.ellipse([(620, 50), (750, 155)], outline="black", width=6)
    d.line([(750, 100), (750, 250)], fill="black", width=6)

    res = real_predictor.predict(img)
    assert res["is_multi_digit"] is True
    assert res["full_number"] == "789"
    assert len(res["digits"]) == 3
    assert [d_info["predicted_digit"] for d_info in res["digits"]] == [7, 8, 9]


def test_regression_slanted_handwriting_deskew(real_predictor):
    """
    Verify that slanted handwriting (e.g. forward tilted 1 and 7) correctly deskews and predicts.
    """
    # Slanted 1
    img1 = Image.new("RGB", (280, 280), (255, 255, 255))
    d1 = ImageDraw.Draw(img1)
    d1.line([(180, 40), (110, 240)], fill="black", width=14)
    res1 = real_predictor.predict(img1)
    assert res1["predicted_digit"] == 1

    # Slanted 7
    img7 = Image.new("RGB", (280, 280), (255, 255, 255))
    d7 = ImageDraw.Draw(img7)
    d7.line([(70, 50), (220, 50)], fill="black", width=14)
    d7.line([(220, 50), (90, 240)], fill="black", width=14)
    res7 = real_predictor.predict(img7)
    assert res7["predicted_digit"] == 7
