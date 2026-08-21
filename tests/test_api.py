import io
from pathlib import Path
import sys
import pytest
from PIL import Image, ImageDraw

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for FastAPI with proper lifespan handling."""
    with TestClient(app) as test_client:
        yield test_client


def test_api_health(client):
    """Verify /health endpoint returns healthy status and system info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_available" in data
    assert "device" in data
    assert "python_version" in data


def test_api_metrics(client):
    """Verify /metrics endpoint returns JSON evaluation metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_accuracy" in data
    assert "test_loss" in data
    assert data["test_accuracy"] > 0.90


def test_api_predict_valid_digit(client):
    """Verify /predict endpoint with a drawn digit image."""
    img = Image.new("RGB", (280, 280), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(140, 40), (140, 240)], fill="black", width=14)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("digit_1.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_digit"] == 1
    assert data["confidence"] > 0.50
    assert len(data["top_k"]) == 3
    assert data["inference_latency_ms"] >= 0.0


def test_api_predict_invalid_content_type(client):
    """Verify /predict returns 400 when a non-image file is provided."""
    buf = io.BytesIO(b"Hello text file")
    response = client.post(
        "/predict",
        files={"file": ("test.txt", buf, "text/plain")}
    )
    assert response.status_code == 400


def test_api_predict_blank_image(client):
    """Verify /predict returns 400 when an empty/blank image is provided."""
    img = Image.new("RGB", (280, 280), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("blank.png", buf, "image/png")}
    )
    assert response.status_code == 400
    assert "No recognizable digit" in response.json()["detail"]


def test_api_predict_multi_digit(client):
    """Verify /predict endpoint with a multi-digit number '42'."""
    img = Image.new("RGB", (600, 300), (240, 242, 245))
    draw = ImageDraw.Draw(img)
    # Digit 4
    draw.line([(150, 50), (90, 180)], fill="black", width=6)
    draw.line([(90, 180), (220, 180)], fill="black", width=6)
    draw.line([(170, 50), (170, 250)], fill="black", width=6)
    # Digit 2
    draw.arc([(340, 50), (480, 160)], 180, 0, fill="black", width=6)
    draw.line([(480, 105), (340, 250)], fill="black", width=6)
    draw.line([(340, 250), (480, 250)], fill="black", width=6)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("number_42.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_multi_digit"] is True
    assert data["full_number"] == "42"
    assert data["confidence"] > 0.50

