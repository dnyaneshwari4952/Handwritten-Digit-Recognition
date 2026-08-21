"""
src/preprocessing.py
Robust Image Normalization, Inversion, Bounding-Box Cropping, Aspect-Ratio Scaling,
Center-of-Mass Centering, and Preprocessing Diagnostics for MNIST CNN.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw
import scipy.ndimage

from src.utils import setup_logger

logger = setup_logger("preprocessing")


def normalize_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] to [0.0, 1.0] float32.

    Args:
        images: Input uint8 or float array.

    Returns:
        Normalized float32 array in range [0.0, 1.0].
    """
    if not isinstance(images, np.ndarray):
        images = np.array(images)

    if np.isnan(images).any() or np.isinf(images).any():
        raise ValueError("Input images contain NaN or Inf values.")

    # Check if already normalized
    if images.dtype in (np.float32, np.float64) and images.max() <= 1.0 and images.min() >= 0.0:
        return images.astype(np.float32)

    norm_images = images.astype(np.float32) / 255.0

    # Strict bounds verification
    if norm_images.min() < 0.0 or norm_images.max() > 1.0:
        raise ValueError(f"Normalized values out of bounds [0.0, 1.0]: min={norm_images.min()}, max={norm_images.max()}")

    return norm_images


def reshape_for_cnn(images: np.ndarray) -> np.ndarray:
    """
    Ensure images have a channel dimension (N, 28, 28, 1).

    Args:
        images: Array of shape (N, 28, 28), (28, 28), or (N, 28, 28, 1).

    Returns:
        Array with 4 dimensions (N, 28, 28, 1).
    """
    if images.ndim == 2:  # Single image (28, 28)
        return np.expand_dims(images, axis=(0, -1))
    elif images.ndim == 3:
        if images.shape[1:] == (28, 28):  # (N, 28, 28)
            return np.expand_dims(images, axis=-1)
        elif images.shape[0] == 28 and images.shape[1] == 28 and images.shape[2] == 1:  # (28, 28, 1)
            return np.expand_dims(images, axis=0)
        else:
            raise ValueError(f"Unexpected 3D image shape: {images.shape}")
    elif images.ndim == 4:
        if images.shape[1:] == (28, 28, 1):
            return images
        raise ValueError(f"Unexpected 4D image shape: {images.shape}, expected (N, 28, 28, 1)")
    else:
        raise ValueError(f"Invalid dimensions: {images.ndim}D image array provided.")


def preprocess_pipeline(images: np.ndarray) -> np.ndarray:
    """
    Standard preprocessing pipeline for dataset batches (e.g. MNIST test set).
    Validates, normalizes, and reshapes.

    Args:
        images: Raw images array of shape (N, 28, 28).

    Returns:
        Preprocessed (N, 28, 28, 1) float32 array.
    """
    normalized = normalize_images(images)
    reshaped = reshape_for_cnn(normalized)
    return reshaped


def deskew_image(image_arr: np.ndarray) -> np.ndarray:
    """
    Deskew a grayscale digit image (bright stroke on dark background)
    by computing central image moments (mu11 / mu02).
    """
    total_mass = float(np.sum(image_arr))
    if total_mass < 1e-3:
        return image_arr

    h, w = image_arr.shape
    y_coords, x_coords = np.mgrid[:h, :w]

    cx = float(np.sum(x_coords * image_arr) / total_mass)
    cy = float(np.sum(y_coords * image_arr) / total_mass)

    mu11 = float(np.sum((x_coords - cx) * (y_coords - cy) * image_arr) / total_mass)
    mu02 = float(np.sum((y_coords - cy) ** 2 * image_arr) / total_mass)

    if abs(mu02) < 1e-3:
        return image_arr

    skew = float(mu11 / mu02)
    # Clamp extreme skew values to avoid over-distortion
    skew = max(-0.55, min(0.55, skew))

    if abs(skew) < 0.05:
        return image_arr

    transform = np.array([
        [1.0, 0.0],
        [-skew, 1.0]
    ])
    offset = np.array([0.0, skew * cy])

    deskewed = scipy.ndimage.affine_transform(
        image_arr,
        transform,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0
    )
    return np.clip(deskewed, 0.0, 255.0)


def preprocess_single_image(
    image_input: Union[np.ndarray, Image.Image, Path, str],
    auto_invert: bool = True,
    return_metadata: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Preprocess an arbitrary single image (canvas, file upload, or numpy array) for MNIST CNN inference.

    Follows the official MNIST construction standard (LeCun et al.) with robust enhancements:
    1. Grayscale conversion with intelligent RGBA alpha compositing.
    2. Dynamic background estimation and adaptive paper background subtraction.
    3. Dynamic contrast stretching (ensuring canonical MNIST peak intensity).
    4. Blank canvas / empty image detection.
    5. Bounding box detection with margin.
    6. Aspect-ratio preserving scaling into a 20x20 pixel bounding box.
    7. Morphological stroke thickness regulation (thickening thin pen strokes).
    8. Embedding inside a 28x28 canvas.
    9. Slant deskewing (moment-based shear correction).
    10. Center-of-mass alignment (translating digit center of mass to (14.0, 14.0)).
    11. Normalization to [0.0, 1.0] float32.
    12. Reshaping to (1, 28, 28, 1).

    Args:
        image_input: Filepath, PIL Image, or NumPy array.
        auto_invert: If True, automatically detect background brightness and invert if light.
        return_metadata: If True, return (tensor, metadata_dict) tuple.

    Returns:
        NumPy array of shape (1, 28, 28, 1) with dtype float32 (or tuple if return_metadata=True).
    """
    # 1. Load to PIL Image
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at: {path}")
        pil_img = Image.open(path)
    elif isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(axis=2)

        pil_img = Image.fromarray(arr)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # 2. Handle RGBA alpha transparency
    if pil_img.mode == "RGBA":
        r, g, b, a = pil_img.split()
        rgb = Image.merge("RGB", (r, g, b))
        a_arr = np.array(a)
        rgb_arr = np.array(rgb.convert("L"))
        mask = a_arr > 20
        if np.any(mask):
            fg_mean = np.mean(rgb_arr[mask])
            if fg_mean > 127:  # Bright stroke on transparent background
                bg = Image.new("RGBA", pil_img.size, (0, 0, 0, 255))
            else:  # Dark stroke on transparent background
                bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
            pil_img = Image.alpha_composite(bg, pil_img).convert("L")
        else:
            pil_img = pil_img.convert("L")
    else:
        pil_img = pil_img.convert("L")

    img_arr = np.array(pil_img, dtype=np.float32)

    # 3. Dynamic Background Estimation & Clean Inversion
    if auto_invert:
        border_pixels = np.concatenate([
            img_arr[0, :], img_arr[-1, :],
            img_arr[:, 0], img_arr[:, -1]
        ])
        border_mean = float(np.mean(border_pixels))
        border_p90 = float(np.percentile(border_pixels, 90))
        if border_mean > 120:
            # White / off-white paper: subtract background level
            img_arr = np.clip(border_p90 - img_arr, 0.0, 255.0)

    # 4. Dynamic Contrast Stretching & Noise Floor Cleanup
    if img_arr.max() > 20:
        img_arr = (img_arr / img_arr.max()) * 255.0
        # Suppress faint paper texture noise
        img_arr[img_arr < 35.0] = 0.0

    # 5. Blank Canvas Detection
    is_blank = False
    if img_arr.max() < 20 or np.sum(img_arr > 30) < 5:
        is_blank = True
        blank_tensor = np.zeros((1, 28, 28, 1), dtype=np.float32)
        meta = {
            "is_blank": True,
            "bbox": (0, 0, 0, 0),
            "digit_size": (0, 0),
            "scaled_size": (0, 0),
            "center_of_mass": (14.0, 14.0),
            "shift": (0, 0),
            "mean_intensity": float(img_arr.mean()),
            "max_intensity": float(img_arr.max()),
        }
        return (blank_tensor, meta) if return_metadata else blank_tensor

    # 6. Bounding Box Detection
    thresh = max(30.0, float(img_arr.max()) * 0.15)
    active_rows = np.any(img_arr > thresh, axis=1)
    active_cols = np.any(img_arr > thresh, axis=0)

    if not active_rows.any() or not active_cols.any():
        is_blank = True
        blank_tensor = np.zeros((1, 28, 28, 1), dtype=np.float32)
        meta = {
            "is_blank": True,
            "bbox": (0, 0, 0, 0),
            "digit_size": (0, 0),
            "scaled_size": (0, 0),
            "center_of_mass": (14.0, 14.0),
            "shift": (0, 0),
            "mean_intensity": float(img_arr.mean()),
            "max_intensity": float(img_arr.max()),
        }
        return (blank_tensor, meta) if return_metadata else blank_tensor

    rmin, rmax = np.where(active_rows)[0][[0, -1]]
    cmin, cmax = np.where(active_cols)[0][[0, -1]]

    # Add a slight 2-pixel margin if within bounds
    h_orig, w_orig = img_arr.shape
    rmin = max(0, rmin - 2)
    rmax = min(h_orig - 1, rmax + 2)
    cmin = max(0, cmin - 2)
    cmax = min(w_orig - 1, cmax + 2)

    digit_crop = img_arr[rmin:rmax+1, cmin:cmax+1]
    crop_h, crop_w = digit_crop.shape

    # 7. Aspect-Ratio Preserving Scaling into 20x20 Box
    crop_pil = Image.fromarray(digit_crop.astype(np.uint8))
    scale = 20.0 / max(crop_h, crop_w)
    new_w = max(1, min(20, int(round(crop_w * scale))))
    new_h = max(1, min(20, int(round(crop_h * scale))))

    resized_digit = crop_pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
    resized_arr = np.array(resized_digit, dtype=np.float32)

    if resized_arr.max() > 0:
        resized_arr = (resized_arr / resized_arr.max()) * 255.0

    # 8. Morphological Stroke Regulation (Thicken thin pen strokes)
    stroke_fraction = np.sum(resized_arr > 50) / (new_w * new_h)
    if stroke_fraction < 0.20:
        resized_arr = scipy.ndimage.grey_dilation(resized_arr, size=(2, 2))
        if resized_arr.max() > 0:
            resized_arr = (resized_arr / resized_arr.max()) * 255.0

    # 9. Embedding inside 28x28 Canvas
    canvas28 = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas28[start_y:start_y+new_h, start_x:start_x+new_w] = resized_arr

    # 10. Slant Deskewing
    canvas28 = deskew_image(canvas28)

    # 11. Center of Mass Centering
    total_mass = float(np.sum(canvas28))
    shift_x, shift_y = 0, 0
    cx, cy = 14.0, 14.0
    if total_mass > 0:
        cy_calc, cx_calc = scipy.ndimage.center_of_mass(canvas28)
        cx, cy = float(cx_calc), float(cy_calc)
        shift_y = int(np.round(14.0 - cy))
        shift_x = int(np.round(14.0 - cx))
        # Keep shift within safe bounds [-4, 4] to prevent clipping
        shift_y = max(-4, min(4, shift_y))
        shift_x = max(-4, min(4, shift_x))
        if shift_y != 0 or shift_x != 0:
            canvas28 = scipy.ndimage.shift(canvas28, (shift_y, shift_x), mode="constant", cval=0.0)

    # Clean negative artifacts or overflow from interpolation
    canvas28 = np.clip(canvas28, 0.0, 255.0)

    # 12. Normalize to [0.0, 1.0] and Reshape
    norm_tensor = (canvas28 / 255.0).astype(np.float32)
    tensor = reshape_for_cnn(norm_tensor)

    if return_metadata:
        metadata = {
            "is_blank": is_blank,
            "bbox": (int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)),
            "digit_size": (int(crop_w), int(crop_h)),
            "scaled_size": (int(new_w), int(new_h)),
            "center_of_mass": (round(cx, 2), round(cy, 2)),
            "shift": (int(shift_x), int(shift_y)),
            "mean_intensity": float(norm_tensor.mean()),
            "max_intensity": float(norm_tensor.max()),
        }
        return tensor, metadata

    return tensor


def segment_digit_components(
    image_input: Union[np.ndarray, Image.Image, Path, str],
    auto_invert: bool = True
) -> list:
    """
    Detect, isolate, and standardize all individual handwritten digit components in an image.
    Supports single and multi-digit recognition (e.g. '42', '2024', '789').

    Returns a list of dicts sorted left-to-right, each containing:
    - 'box': (x, y, w, h)
    - 'tensor': (1, 28, 28, 1) float32 array
    - 'metadata': dict with size, center of mass, etc.
    - 'crop': cropped PIL Image
    - 'preprocessed_image': 28x28 numpy array
    """
    # 1. Load image to PIL
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at: {path}")
        pil_img = Image.open(path)
    elif isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(axis=2)
        pil_img = Image.fromarray(arr)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # 2. Alpha & Grayscale
    if pil_img.mode == "RGBA":
        r, g, b, a = pil_img.split()
        rgb = Image.merge("RGB", (r, g, b))
        a_arr = np.array(a)
        rgb_arr = np.array(rgb.convert("L"))
        mask = a_arr > 20
        if np.any(mask):
            fg_mean = np.mean(rgb_arr[mask])
            bg_col = (0, 0, 0, 255) if fg_mean > 127 else (255, 255, 255, 255)
            gray_pil = Image.alpha_composite(Image.new("RGBA", pil_img.size, bg_col), pil_img).convert("L")
        else:
            gray_pil = pil_img.convert("L")
    else:
        gray_pil = pil_img.convert("L")

    gray_arr = np.array(gray_pil, dtype=np.float32)

    # 3. Dynamic background & contrast
    if auto_invert:
        border_pixels = np.concatenate([
            gray_arr[0, :], gray_arr[-1, :],
            gray_arr[:, 0], gray_arr[:, -1]
        ])
        border_mean = float(np.mean(border_pixels))
        border_p90 = float(np.percentile(border_pixels, 90))
        if border_mean > 120:
            inv = np.clip(border_p90 - gray_arr, 0.0, 255.0)
        else:
            inv = gray_arr.copy()
    else:
        inv = gray_arr.copy()

    if inv.max() > 20:
        inv = (inv / inv.max()) * 255.0
        inv[inv < 35.0] = 0.0

    # 4. Check for blank
    if inv.max() < 20 or np.sum(inv > 30) < 5:
        blank_t = np.zeros((1, 28, 28, 1), dtype=np.float32)
        return [{
            "box": (0, 0, 0, 0),
            "tensor": blank_t,
            "metadata": {"is_blank": True, "center_of_mass": (14.0, 14.0)},
            "crop": Image.new("L", (28, 28), 0),
            "preprocessed_image": np.zeros((28, 28), dtype=np.float32),
        }]

    # 5. Connected Component Analysis with morphological bridge
    binary = inv > 35.0
    struct = np.ones((5, 5))
    closed = scipy.ndimage.binary_closing(binary, structure=struct)
    labeled, num_features = scipy.ndimage.label(closed)

    # Collect initial candidate boxes
    raw_boxes = []
    for fid in range(1, num_features + 1):
        rows = np.any(labeled == fid, axis=1)
        cols = np.any(labeled == fid, axis=0)
        if not rows.any() or not cols.any():
            continue
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        w = cmax - cmin + 1
        h = rmax - rmin + 1
        # Minimum size filter for noise
        if w >= 6 and h >= 10:
            raw_boxes.append([int(cmin), int(rmin), int(cmax), int(rmax)])

    if not raw_boxes:
        thresh = max(30.0, float(inv.max()) * 0.15)
        active_rows = np.any(inv > thresh, axis=1)
        active_cols = np.any(inv > thresh, axis=0)
        if not active_rows.any() or not active_cols.any():
            blank_t = np.zeros((1, 28, 28, 1), dtype=np.float32)
            return [{
                "box": (0, 0, 0, 0),
                "tensor": blank_t,
                "metadata": {"is_blank": True, "center_of_mass": (14.0, 14.0)},
                "crop": Image.new("L", (28, 28), 0),
                "preprocessed_image": np.zeros((28, 28), dtype=np.float32),
            }]
        rmin, rmax = np.where(active_rows)[0][[0, -1]]
        cmin, cmax = np.where(active_cols)[0][[0, -1]]
        raw_boxes = [[int(cmin), int(rmin), int(cmax), int(rmax)]]

    # Merge horizontally overlapping or vertically aligned components
    merged = True
    while merged:
        merged = False
        for i in range(len(raw_boxes)):
            for j in range(i + 1, len(raw_boxes)):
                b1, b2 = raw_boxes[i], raw_boxes[j]
                overlap_x = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                min_w = min(b1[2] - b1[0] + 1, b2[2] - b2[0] + 1)
                if (overlap_x / min_w) > 0.35:
                    raw_boxes[i] = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
                    raw_boxes.pop(j)
                    merged = True
                    break
            if merged:
                break

    # Sort left to right
    raw_boxes.sort(key=lambda b: b[0])

    # Extract, deskew, and standardize each component
    digits = []
    h_orig, w_orig = inv.shape

    for cmin, rmin, cmax, rmax in raw_boxes:
        rmin_m = max(0, rmin - 2)
        rmax_m = min(h_orig - 1, rmax + 2)
        cmin_m = max(0, cmin - 2)
        cmax_m = min(w_orig - 1, cmax + 2)

        crop = inv[rmin_m:rmax_m + 1, cmin_m:cmax_m + 1]
        crop_h, crop_w = crop.shape

        scale = 20.0 / max(crop_h, crop_w)
        new_w = max(1, min(20, int(round(crop_w * scale))))
        new_h = max(1, min(20, int(round(crop_h * scale))))

        crop_pil = Image.fromarray(crop.astype(np.uint8))
        resized = np.array(crop_pil.resize((new_w, new_h), Image.Resampling.BICUBIC), dtype=np.float32)

        if resized.max() > 0:
            resized = (resized / resized.max()) * 255.0

        stroke_frac = np.sum(resized > 50) / (new_w * new_h)
        if stroke_frac < 0.20:
            resized = scipy.ndimage.grey_dilation(resized, size=(2, 2))
            if resized.max() > 0:
                resized = (resized / resized.max()) * 255.0

        canvas28 = np.zeros((28, 28), dtype=np.float32)
        start_y = (28 - new_h) // 2
        start_x = (28 - new_w) // 2
        canvas28[start_y:start_y + new_h, start_x:start_x + new_w] = resized

        canvas28 = deskew_image(canvas28)

        total_mass = float(np.sum(canvas28))
        shift_x, shift_y = 0, 0
        cx, cy = 14.0, 14.0
        if total_mass > 0:
            cy_calc, cx_calc = scipy.ndimage.center_of_mass(canvas28)
            cx, cy = float(cx_calc), float(cy_calc)
            shift_y = max(-4, min(4, int(np.round(14.0 - cy))))
            shift_x = max(-4, min(4, int(np.round(14.0 - cx))))
            if shift_y != 0 or shift_x != 0:
                canvas28 = scipy.ndimage.shift(canvas28, (shift_y, shift_x), mode="constant", cval=0.0)

        canvas28 = np.clip(canvas28, 0.0, 255.0)
        norm_t = reshape_for_cnn((canvas28 / 255.0).astype(np.float32))

        digits.append({
            "box": (int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)),
            "tensor": norm_t,
            "metadata": {
                "is_blank": False,
                "bbox": (int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)),
                "digit_size": (int(crop_w), int(crop_h)),
                "scaled_size": (int(new_w), int(new_h)),
                "center_of_mass": (round(cx, 2), round(cy, 2)),
                "shift": (int(shift_x), int(shift_y)),
                "mean_intensity": float(norm_t.mean()),
                "max_intensity": float(norm_t.max()),
            },
            "crop": crop_pil,
            "preprocessed_image": canvas28 / 255.0,
        })

    return digits


def preprocess_external_digit(
    image_input: Union[np.ndarray, Image.Image, Path, str],
    auto_invert: bool = True
) -> np.ndarray:
    """
    Unified entry point for both upload and canvas digit preprocessing.
    """
    return preprocess_single_image(image_input, auto_invert=auto_invert, return_metadata=False)


def extract_preprocessing_stages(
    image_input: Union[np.ndarray, Image.Image, Path, str],
    auto_invert: bool = True
) -> Dict[str, Any]:
    """
    Extract intermediate representations of the preprocessing pipeline for visual debugging.
    """
    tensor, meta = preprocess_single_image(image_input, auto_invert=auto_invert, return_metadata=True)

    # 1. Original
    if isinstance(image_input, (str, Path)):
        orig_pil = Image.open(str(image_input))
    elif isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(axis=2)
        orig_pil = Image.fromarray(arr)
    elif isinstance(image_input, Image.Image):
        orig_pil = image_input.copy()
    else:
        orig_pil = Image.new("L", (28, 28), 0)

    # 2. Grayscale & RGBA Flattened
    if orig_pil.mode == "RGBA":
        r, g, b, a = orig_pil.split()
        rgb = Image.merge("RGB", (r, g, b))
        a_arr = np.array(a)
        rgb_arr = np.array(rgb.convert("L"))
        mask = a_arr > 20
        if np.any(mask):
            fg_mean = np.mean(rgb_arr[mask])
            bg_col = (0, 0, 0, 255) if fg_mean > 127 else (255, 255, 255, 255)
            gray_pil = Image.alpha_composite(Image.new("RGBA", orig_pil.size, bg_col), orig_pil).convert("L")
        else:
            gray_pil = orig_pil.convert("L")
    else:
        gray_pil = orig_pil.convert("L")

    gray_arr = np.array(gray_pil, dtype=np.float32)

    # 3. Polarity & Contrast
    if auto_invert:
        border_pixels = np.concatenate([
            gray_arr[0, :], gray_arr[-1, :],
            gray_arr[:, 0], gray_arr[:, -1]
        ])
        border_mean = float(np.mean(border_pixels))
        border_p90 = float(np.percentile(border_pixels, 90))
        if border_mean > 120:
            polarity_arr = np.clip(border_p90 - gray_arr, 0.0, 255.0)
        else:
            polarity_arr = gray_arr.copy()
    else:
        polarity_arr = gray_arr.copy()

    if polarity_arr.max() > 20:
        polarity_arr = (polarity_arr / polarity_arr.max()) * 255.0
        polarity_arr[polarity_arr < 35.0] = 0.0

    polarity_pil = Image.fromarray(np.clip(polarity_arr, 0, 255).astype(np.uint8))

    # 4. BBox Overlay
    bbox_pil = polarity_pil.convert("RGB")
    cmin, rmin, bw, bh = meta["bbox"]
    if not meta["is_blank"] and bw > 0 and bh > 0:
        draw = ImageDraw.Draw(bbox_pil)
        draw.rectangle([cmin, rmin, cmin + bw, rmin + bh], outline="#38BDF8", width=2)

    # 5. Cropped
    if not meta["is_blank"] and bw > 0 and bh > 0:
        crop_pil = polarity_pil.crop((cmin, rmin, cmin + bw, rmin + bh))
    else:
        crop_pil = Image.new("L", (20, 20), 0)

    # 6. Aspect 20x20
    scaled_w, scaled_h = meta["scaled_size"] if not meta["is_blank"] else (20, 20)
    resized_pil = crop_pil.resize((max(1, scaled_w), max(1, scaled_h)), Image.Resampling.BICUBIC)

    # 7. Centered 28x28
    centered_arr = (tensor.squeeze() * 255.0).astype(np.uint8)
    centered_pil = Image.fromarray(centered_arr)

    return {
        "original": orig_pil,
        "grayscale": gray_pil,
        "polarity_corrected": polarity_pil,
        "bbox_overlay": bbox_pil,
        "cropped_digit": crop_pil,
        "aspect_preserved_20x20": resized_pil,
        "centered_28x28": centered_pil,
        "final_tensor": tensor,
        "metadata": meta,
    }


def to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert 1D integer labels to 2D one-hot encoded matrix.

    Args:
        labels: 1D array of integer class indices.
        num_classes: Total number of classes (default 10).

    Returns:
        One-hot encoded float32 array of shape (N, num_classes).
    """
    labels = np.asarray(labels, dtype=np.int32)
    one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot
