"""
api/app.py
Production-Grade FastAPI Service for Real-Time Digit Classification Inference.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_PATHS
from src.predict import DigitPredictor
from src.utils import get_device_info, load_json

app = FastAPI(
    title="MNIST Handwritten Digit Recognition API",
    description="High-performance deep learning REST API for classifying handwritten digits.",
    version="1.0.0",
)

# Enable CORS for client applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor cache
predictor_instance: Optional[DigitPredictor] = None


def get_api_predictor() -> DigitPredictor:
    """Retrieve or initialize global DigitPredictor instance."""
    global predictor_instance
    if predictor_instance is None:
        if not DEFAULT_PATHS.model_save_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model checkpoint not found. Please train the model before invoking inference."
            )
        predictor_instance = DigitPredictor(model_path=DEFAULT_PATHS.model_save_path)
    return predictor_instance


class PredictionCandidate(BaseModel):
    digit: int
    confidence: float


class PredictionResponse(BaseModel):
    predicted_digit: Optional[int] = None
    full_number: Optional[str] = None
    is_multi_digit: bool = False
    confidence: float
    confidence_percent: str
    is_confident: bool
    is_blank: bool
    top_k: List[PredictionCandidate]
    inference_latency_ms: float


@app.get("/health", summary="Health and Hardware Status")
def health_check() -> Dict[str, Any]:
    """Check API health and computational hardware status."""
    device_info = get_device_info()
    model_available = DEFAULT_PATHS.model_save_path.exists()
    return {
        "status": "healthy",
        "model_available": model_available,
        "device": device_info["device"],
        "python_version": device_info["python_version"],
    }


@app.get("/metrics", summary="Model Evaluation Metrics")
def get_evaluation_metrics() -> Dict[str, Any]:
    """Retrieve test set metrics (accuracy, F1-score, loss)."""
    if not DEFAULT_PATHS.evaluation_json_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation metrics not found. Run evaluation first.")
    return load_json(DEFAULT_PATHS.evaluation_json_path)


@app.post("/predict", response_model=PredictionResponse, summary="Predict Digit from Image")
async def predict_digit_endpoint(file: UploadFile = File(...), top_k: int = 3):
    """
    Classify a handwritten digit from an uploaded image file (PNG/JPEG).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (PNG, JPG, JPEG).")

    predictor = get_api_predictor()
    start_time = time.perf_counter()

    try:
        from PIL import Image
        import io
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        res = predictor.predict(image, top_k=top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if res.get("is_blank", False):
            raise HTTPException(
                status_code=400,
                detail="No recognizable digit detected. Please provide an image with a drawn or handwritten digit."
            )

        return PredictionResponse(
            predicted_digit=res.get("predicted_digit"),
            full_number=res.get("full_number", str(res.get("predicted_digit", ""))),
            is_multi_digit=res.get("is_multi_digit", False),
            confidence=res["confidence"],
            confidence_percent=res["confidence_percent"],
            is_confident=res["is_confident"],
            is_blank=False,
            top_k=[PredictionCandidate(**item) for item in res["top_k"]],
            inference_latency_ms=round(latency_ms, 2)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
