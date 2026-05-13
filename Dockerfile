FROM python:3.10-slim

# System dependencies — fixed for Debian Bookworm (libgl1-mesa-glx removed)
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
COPY requirements-hf.txt .

# Install PyTorch CPU-only (smaller image, sufficient for inference)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies from requirements
RUN pip install --no-cache-dir \
    timm>=0.9.0 \
    torchmetrics>=1.0.0 \
    mlflow>=2.5.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.3.0 \
    "Pillow>=10.0.0" \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    flask>=2.3.0 \
    flask-cors>=4.0.0 \
    "gradio>=4.0.0" \
    opencv-python-headless>=4.8.0 \
    pyyaml>=6.0 \
    tqdm>=4.65.0 \
    "huggingface_hub>=0.16.0" \
    python-dotenv>=1.0.0

# Copy project files
COPY src/ ./src/
COPY webapp/ ./webapp/
COPY configs/ ./configs/
COPY mlops/ ./mlops/
COPY scripts/ ./scripts/
COPY setup.py .
COPY verify.py .

# Create required directories
RUN mkdir -p models logs data docs/figures mlruns

# Make Python find the src package
ENV PYTHONPATH=/app

# Model and app config
ENV MODEL_PATH=/app/models/best_model.pth
ENV MODEL_ARCH=efficientnet_b3
ENV PORT=7860
ENV GRADIO_SHARE=false
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

EXPOSE 7860

# Launch Gradio demo
CMD ["python", "webapp/gradio_app.py"]