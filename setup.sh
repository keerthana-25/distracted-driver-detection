#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Distracted Driver Detection — One-Command Setup Script
# Works on Mac, Linux, Windows (WSL)
# Usage: bash setup.sh
# ─────────────────────────────────────────────────────────────

set -e  # Exit on any error

echo ""
echo "=========================================="
echo "  Distracted Driver Detection Setup"
echo "=========================================="
echo ""

# ── Check Python version ──
PYTHON=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo $PYTHON | cut -d. -f1)
MINOR=$(echo $PYTHON | cut -d. -f2)

echo "Python version: $PYTHON"

if [ "$MAJOR" -lt 3 ] || [ "$MINOR" -lt 10 ]; then
    echo "ERROR: Python 3.10+ required. Got $PYTHON"
    exit 1
fi

# ── Check pip ──
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 not found. Please install pip."
    exit 1
fi

echo ""
echo "[1/4] Installing PyTorch..."
# Detect if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU detected — installing CUDA version"
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
else
    echo "  No GPU — installing CPU version"
    pip3 install torch torchvision -q
fi

echo ""
echo "[2/4] Installing project dependencies..."
pip3 install -r requirements.txt -q

echo ""
echo "[3/4] Setting up directories..."
mkdir -p models data/raw data/processed logs docs/figures mlruns

echo ""
echo "[4/4] Verifying installation..."
python3 -c "
import torch, timm, torchmetrics, mlflow, gradio, cv2, numpy, pandas
print('  torch:', torch.__version__)
print('  timm:', timm.__version__)
print('  gradio:', gradio.__version__)
print('  mlflow:', mlflow.__version__)
print('  CUDA available:', torch.cuda.is_available())
print()
print('  All imports OK!')
"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Get data:    python3 scripts/setup_data.py --synthetic"
echo "  2. Train:       python3 scripts/train.py --synthetic --epochs 5"
echo "  3. Run demo:    python3 webapp/gradio_app.py"
echo "  4. View MLflow: mlflow ui --backend-store-uri mlruns"
echo "=========================================="
