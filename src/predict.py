"""
src/predict.py
Standalone Digit Prediction Engine with Top-K Confidence, Blank Detection, and Diagnostic Stages.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import keras
import numpy as np
from PIL import Image

from src.config import DEFAULT_PATHS
from src.preprocessing import (
    extract_preprocessing_stages,
    preprocess_single_image,
    segment_digit_components,
)
from src.utils import setup_logger

logger = setup_logger("predict")


class DigitPredictor:
    """
    Production-ready inference engine for MNIST single and multi-digit number recognition.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None, model: Optional[keras.Model] = None):
        """
        Initialize the predictor with a saved model or in-memory model.

        Args:
            model_path: Path to .keras model file.
            model: Pre-loaded Keras model.
        """
        if model is not None:
            self.model = model
        else:
            path = Path(model_path or DEFAULT_PATHS.model_save_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Trained model not found at '{path}'. Please train the model first by running `python -m src.train`."
                )
            logger.info(f"Loading trained CNN model from {path}...")
            self.model = keras.models.load_model(str(path))

    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image, Path, str],
        top_k: int = 3,
        auto_invert: bool = True,
        return_stages: bool = False
    ) -> Dict[str, Any]:
        """
        Predict single or multi-digit handwritten numbers with robust preprocessing.

        Args:
            image_input: Filepath, PIL Image, or NumPy array.
            top_k: Number of highest-probability classes to return.
            auto_invert: Automatically invert colors if image is black-on-white.
            return_stages: If True, include multi-stage visual inspection steps.

        Returns:
            Dictionary with prediction results:
            - 'predicted_digit': int or None (for first digit)
            - 'full_number': str (e.g. "42", "789", or "6")
            - 'is_multi_digit': bool
            - 'confidence': float (0.0 to 1.0)
            - 'confidence_percent': str
            - 'is_confident': bool (confidence >= 0.60)
            - 'is_blank': bool
            - 'digits': list of segmented per-digit details (for multi-digit)
            - 'top_k': list of {'digit': int, 'confidence': float}
            - 'probabilities': list of float (length 10)
            - 'preprocessed_image': 28x28 numpy array
            - 'metadata': dict with bbox, center of mass, etc.
            - 'stages': dict with intermediate PIL images (optional)
            - 'annotated_image': PIL Image with bounding box annotations
        """
        return self.predict_number(
            image_input=image_input,
            top_k=top_k,
            auto_invert=auto_invert,
            return_stages=return_stages
        )

    def predict_number(
        self,
        image_input: Union[np.ndarray, Image.Image, Path, str],
        top_k: int = 3,
        auto_invert: bool = True,
        return_stages: bool = False
    ) -> Dict[str, Any]:
        """
        Universal recognition engine supporting single and multi-digit handwritten numbers.
        """
        # 1. Segment digit components
        components = segment_digit_components(image_input, auto_invert=auto_invert)

        # 2. Handle blank image
        if len(components) == 1 and components[0]["metadata"].get("is_blank", False):
            result = {
                "predicted_digit": None,
                "full_number": "",
                "is_multi_digit": False,
                "confidence": 0.0,
                "confidence_percent": "0.00%",
                "is_confident": False,
                "is_blank": True,
                "message": "No recognizable digit detected.",
                "digits": [],
                "top_k": [{"digit": i, "confidence": 0.10} for i in range(top_k)],
                "probabilities": [0.10] * 10,
                "preprocessed_image": np.zeros((28, 28), dtype=np.float32),
                "metadata": components[0]["metadata"],
            }
            if return_stages:
                result["stages"] = extract_preprocessing_stages(image_input, auto_invert=auto_invert)
            return result

        # 3. Batch prediction across all segmented components
        batch_tensors = np.vstack([comp["tensor"] for comp in components])
        probs = self.model.predict(batch_tensors, verbose=0)

        # Prepare original image for annotation
        if isinstance(image_input, (str, Path)):
            orig_pil = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(axis=2)
            orig_pil = Image.fromarray(arr).convert("RGB")
        elif isinstance(image_input, Image.Image):
            orig_pil = image_input.convert("RGB")
        else:
            orig_pil = Image.new("RGB", (280, 280), (255, 255, 255))

        annotated_pil = orig_pil.copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(annotated_pil)

        digits_list = []
        is_multi_digit = len(components) > 1

        for idx, comp in enumerate(components):
            p = probs[idx]
            pred_d = int(np.argmax(p))
            conf = float(p[pred_d])
            top_k_indices = np.argsort(p)[::-1][:top_k]
            top_k_list = [
                {"digit": int(k), "confidence": float(p[k])}
                for k in top_k_indices
            ]

            cmin, rmin, bw, bh = comp["box"]
            if bw > 0 and bh > 0:
                draw.rectangle([cmin, rmin, cmin + bw, rmin + bh], outline="#38BDF8", width=3)
                draw.text((cmin + 2, max(0, rmin - 16)), f"{pred_d} ({conf*100:.1f}%)", fill="#38BDF8")

            digits_list.append({
                "digit_index": idx,
                "predicted_digit": pred_d,
                "confidence": conf,
                "confidence_percent": f"{conf * 100:.2f}%",
                "top_k": top_k_list,
                "probabilities": [float(val) for val in p],
                "box": comp["box"],
                "crop": comp["crop"],
                "preprocessed_image": comp["preprocessed_image"],
                "metadata": comp["metadata"],
            })

        full_number = "".join(str(d["predicted_digit"]) for d in digits_list)
        avg_confidence = float(np.mean([d["confidence"] for d in digits_list]))
        is_confident = bool(avg_confidence >= 0.60)

        result = {
            "predicted_digit": digits_list[0]["predicted_digit"],
            "full_number": full_number,
            "is_multi_digit": is_multi_digit,
            "confidence": avg_confidence,
            "confidence_percent": f"{avg_confidence * 100:.2f}%",
            "is_confident": is_confident,
            "is_blank": False,
            "digits": digits_list,
            "top_k": digits_list[0]["top_k"],
            "probabilities": digits_list[0]["probabilities"],
            "preprocessed_image": digits_list[0]["preprocessed_image"],
            "metadata": digits_list[0]["metadata"],
            "annotated_image": annotated_pil,
        }

        if return_stages:
            result["stages"] = extract_preprocessing_stages(image_input, auto_invert=auto_invert)

        logger.info(f"Number Recognized: '{full_number}' (Confidence: {result['confidence_percent']}, Multi-digit: {is_multi_digit})")
        return result

    def predict_batch(
        self,
        images: np.ndarray,
        batch_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform fast vectorized batch inference.

        Args:
            images: Preprocessed (N, 28, 28, 1) float32 array.
            batch_size: Inference batch size.

        Returns:
            (predicted_classes, confidence_scores)
        """
        probs = self.model.predict(images, batch_size=batch_size, verbose=0)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        return preds, confs


def predict_digit(
    image_input: Union[np.ndarray, Image.Image, Path, str],
    model_path: Optional[Union[str, Path]] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Convenience function for one-off digit predictions.
    """
    predictor = DigitPredictor(model_path=model_path)
    return predictor.predict(image_input, top_k=top_k)
