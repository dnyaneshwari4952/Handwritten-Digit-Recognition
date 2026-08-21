# Production Dockerfile for MNIST Handwritten Digit Recognition
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and assets
COPY src/ ./src/
COPY app/ ./app/
COPY api/ ./api/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY .streamlit/ ./.streamlit/
COPY streamlit_app.py ./

# Create runtime output directories if missing
RUN mkdir -p artifacts/models artifacts/plots artifacts/metrics artifacts/predictions

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Healthcheck for container status
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: Launch Streamlit Studio
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
