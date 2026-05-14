# 🚗 Distracted Driver Detection

> **Deep Learning Based Distracted Driver Detection for Road Safety**
> EfficientNet-B3 · Grad-CAM · MLflow · Gradio · Docker · GitHub Actions CI/CD

[![CI/CD](https://github.com/keerthana-25/distracted-driver-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/keerthana-25/distracted-driver-detection/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194e2.svg)](https://mlflow.org)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/keerthana-m/distracted-driver-detection)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keerthana-25/distracted-driver-detection/blob/main/notebooks/train_on_colab.ipynb)

---

## 👥 Team

| Name |
|------|
| Keerthana Murlidharan |
| Yashaswini Dinesh |

---

## 📎 Project Resources

| Resource | Link |
|----------|------|
| 🎥 **Video Demo** | [Watch on Google Drive](https://drive.google.com/file/d/1J0-W579fdl9yAARf9jnv9tvT-KsgTQ3Z/view?usp=share_link) |
| 📊 **Presentation (PPT)** | [View on Google Slides](https://docs.google.com/presentation/d/1UD2BdvlxGdKiH0L1nyScItnBznis1pyW/edit?usp=share_link&ouid=107950069020013929648&rtpof=true&sd=true) |
| 📄 **Project Report** | [View on Google Docs](https://docs.google.com/document/d/1DboCWQvjqpkwSVH8g_GRg2BiQbbBVURW/edit?usp=share_link&ouid=107950069020013929648&rtpof=true&sd=true) |

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 🎮 **Live Demo** | [huggingface.co/spaces/keerthana-m/distracted-driver-detection](https://huggingface.co/spaces/keerthana-m/distracted-driver-detection) |
| 💻 **GitHub Repo** | [github.com/keerthana-25/distracted-driver-detection](https://github.com/keerthana-25/distracted-driver-detection) |
| ⚙️ **CI/CD Pipeline** | [GitHub Actions](https://github.com/keerthana-25/distracted-driver-detection/actions) |
| 📊 **MLflow Dashboard** | Run `mlflow ui` locally after training |
| 📓 **Colab Notebook** | [Train on Google Colab (free T4 GPU)](https://colab.research.google.com/github/keerthana-25/distracted-driver-detection/blob/main/notebooks/train_on_colab.ipynb) |

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Detected Behaviours](#detected-behaviours)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Dataset Setup](#dataset-setup)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Ablation Study](#ablation-study)
9. [Results](#results)
10. [Explainability — Grad-CAM](#explainability--grad-cam)
11. [Web Demo](#web-demo)
12. [REST API](#rest-api)
13. [MLOps Pipeline](#mlops-pipeline)
14. [CI/CD Pipeline](#cicd-pipeline)
15. [Docker](#docker)
16. [Deployment — HuggingFace Spaces](#deployment--huggingface-spaces)
17. [Key Design Decisions](#key-design-decisions)
18. [Running Tests](#running-tests)
19. [Team](#team)

---

## Overview

This project builds an **end-to-end MLOps pipeline** for detecting unsafe driver behaviour from dashboard camera images. The system classifies 10 distinct driving behaviours in real-time with Grad-CAM explainability — making predictions interpretable for fleet managers, insurers, and regulators.

**Real-world impact:**
- Distracted driving causes ~9 deaths per day in the US (NHTSA)
- Fleet operators need automated, scalable monitoring
- Insurers need behavioural risk scoring
- Grad-CAM explainability builds trust in predictions

---

## Detected Behaviours

| Class | Behaviour | Risk Level |
|-------|-----------|------------|
| c0 | Safe Driving | ✅ None |
| c1 | Texting — Right Hand | 🚨 High |
| c2 | Phone Call — Right Hand | 🚨 High |
| c3 | Texting — Left Hand | 🚨 High |
| c4 | Phone Call — Left Hand | 🚨 High |
| c5 | Radio Adjusting | ⚠️ Medium |
| c6 | Drinking | ⚠️ Medium |
| c7 | Reaching Behind | 🚨 High |
| c8 | Hair / Makeup | ⚠️ Medium |
| c9 | Talking to Passenger | ℹ️ Low |

### Alert System
- 🚨 **Flashing red banner** — High risk (texting, phone, reaching)
- ⚠️ **Orange banner** — Medium risk (drinking, radio, makeup)
- ℹ️ **Blue banner** — Low risk (talking to passenger)
- ✅ **Green banner** — Safe driving

---

## Project Structure

```
distracted-driver-detection/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── configs/
│   └── config.yaml             # Central config for all modules
├── docs/
│   └── figures/                # Generated plots and visualizations
├── mlops/
│   └── pipelines/
│       └── pipeline.py         # End-to-end ML pipeline orchestrator
├── notebooks/
│   ├── train_on_colab.ipynb    # ⭐ Full training pipeline for Google Colab (T4 GPU)
│   └── ablation_study.ipynb    # Ablation study + Grad-CAM analysis
├── scripts/
│   ├── train.py                # Training CLI entry point
│   ├── evaluate.py             # Test set evaluation + reports
│   ├── setup_data.py           # Dataset download and preparation
│   └── deploy_hf.py            # HuggingFace Spaces deployment
├── src/
│   ├── data/
│   │   └── dataset.py          # Dataset, transforms, DataLoaders
│   ├── explainability/
│   │   └── gradcam.py          # Grad-CAM + ExplainablePredictor
│   ├── inference/
│   │   └── api_server.py       # Flask REST API
│   ├── model/
│   │   └── architecture.py     # EfficientNet model, loss, metrics
│   └── training/
│       ├── trainer.py          # Full training engine + ablation
│       └── visualizations.py   # Confusion matrix, training curves
├── tests/
│   └── test_all.py             # 22 unit tests (pytest)
├── webapp/
│   ├── gradio_app.py           # Gradio demo interface
│   └── templates/
│       └── index.html          # Production web frontend
├── .env.example                # Environment variable template
├── .gitignore
├── conda.yaml                  # Conda environment (MLflow Projects)
├── docker-compose.yml          # Full stack: webapp + API + MLflow
├── Dockerfile                  # Container definition (fixed for Debian Bookworm)
├── MLproject                   # MLflow Projects entry points
├── README.md
├── requirements.txt            # All Python dependencies
├── requirements-hf.txt         # Lighter deps for HuggingFace Spaces
├── setup.py                    # Makes project pip-installable
├── setup.sh                    # One-command setup script
└── verify.py                   # Dependency verification tool
```

---

## Quick Start

### Option 1 — One command setup (Mac/Linux)

```bash
git clone https://github.com/keerthana-25/distracted-driver-detection
cd distracted-driver-detection
bash setup.sh
python3 verify.py        # check everything installed correctly
python3 scripts/setup_data.py --synthetic
python3 webapp/gradio_app.py
```

### Option 2 — Manual setup

```bash
git clone https://github.com/keerthana-25/distracted-driver-detection
cd distracted-driver-detection

pip3 install torch torchvision
pip3 install -r requirements.txt
pip3 install -e .           # makes src/ importable from anywhere

python3 verify.py           # verify installation
```

### Option 3 — Docker

```bash
git clone https://github.com/keerthana-25/distracted-driver-detection
cd distracted-driver-detection

# Copy your trained model into models/
cp /path/to/best_model.pth models/

# Full stack: Gradio + Flask API + MLflow
docker-compose up

# Access:
#   Gradio demo:  http://localhost:7860
#   Flask API:    http://localhost:5000
#   MLflow UI:    http://localhost:5001
```

---

## Dataset Setup

**State Farm Distracted Driver Detection** — [Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

### Option A — Real dataset (Kaggle)

```bash
# 1. Accept competition rules at kaggle.com/c/state-farm-distracted-driver-detection
# 2. Set up Kaggle credentials in ~/.kaggle/kaggle.json
# 3. Download and prepare:
python3 scripts/setup_data.py --kaggle
```

### Option B — Synthetic dataset (no Kaggle needed)

```bash
python3 scripts/setup_data.py --synthetic --n 100
```

### Dataset statistics (real data)

| Split | Samples | Ratio |
|-------|---------|-------|
| Train | ~15,400 | 70% |
| Validation | ~3,300 | 15% |
| Test | ~3,300 | 15% |

Splits are **stratified** — every class is proportionally represented in all splits.

---

## Model Architecture

```
Input [224×224×3]
      ↓
EfficientNet-B3 Backbone (pretrained ImageNet)
      ↓
Last Conv Layer → Grad-CAM hooks registered here
      ↓
Custom Classification Head:
  AdaptiveAvgPool2d → Flatten
  → BatchNorm1d → Dropout(0.4)
  → Linear(feature_dim → 512) → BatchNorm1d → SiLU
  → Dropout(0.2) → Linear(512 → 10)
      ↓
Output: 10-class logits
```

### Supported Backbones

| Architecture | Params | Inference | Use case |
|---|---|---|---|
| **EfficientNet-B3** ⭐ | 12M | ~22ms | Primary model |
| EfficientNet-B0 | 5.3M | ~12ms | Ablation baseline |
| ResNet-50 | 25.6M | ~28ms | Ablation comparison |
| MobileNetV3-Large | 5.4M | ~8ms | Edge deployment |

---

## Training on Google Colab (Recommended)

Training on CPU takes ~15 hours. Google Colab gives you a **free T4 GPU** and reduces this to **~45 minutes**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keerthana-25/distracted-driver-detection/notebooks/train_on_colab.ipynb)

**File:** `notebooks/train_on_colab.ipynb`

### How to use

1. Click the **Open in Colab** badge above
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells from top to bottom
4. In Step 4 choose how to load your data:
   - **Option A** — From Google Drive (fastest, recommended)
   - **Option B** — From Kaggle API
5. Download `best_model.pth` when training finishes

### After training — restore files to your project

```bash
# Place model and history
mv ~/Downloads/best_model.pth        distracted-driver-detection/models/
mv ~/Downloads/training_history.json distracted-driver-detection/models/

# Extract MLflow runs (for dashboard)
unzip ~/Downloads/mlruns.zip -d distracted-driver-detection/

# Launch local demo with trained model
cd distracted-driver-detection
python3 webapp/gradio_app.py

# View MLflow dashboard
mlflow ui --backend-store-uri mlruns
```

---

## Training

### Two-Phase Training Strategy

**Phase 1 (epochs 1–3): Backbone frozen**
Only the classification head trains. High LR (1e-3). Fast convergence without disrupting pretrained features.

**Phase 2 (epoch 4+): Full fine-tuning**
Backbone LR = 1e-4 (10× lower than head). Careful fine-tuning adapts pretrained features to dashcam images.

### Run Training

```bash
# On real dataset
python3 scripts/train.py --data-dir data/processed --epochs 30

# On synthetic data (testing)
python3 scripts/train.py --synthetic --epochs 10

# With ablation study
python3 scripts/train.py --data-dir data/processed --epochs 30 --ablation
```

### Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Loss | Label Smoothing CE (ε=0.1) | Prevents overconfident predictions |
| Optimizer | AdamW | Weight decay as regularization |
| LR Schedule | Linear warmup → Cosine annealing | Smooth convergence |
| Batch size | 32 | Balance GPU memory vs gradient quality |
| Sampling | WeightedRandomSampler | Handles class imbalance |
| Augmentation | Flip, rotate, color jitter, random erase | Reduces overfitting |
| AMP | Enabled | 2× faster, no accuracy loss |
| Gradient clip | 1.0 | Prevents exploding gradients |
| Early stopping | Patience = 7 | Saves compute on plateau |

---

## Ablation Study

Run interactively in the notebook:

```bash
jupyter notebook notebooks/ablation_study.ipynb
```

Or programmatically:

```bash
python3 scripts/train.py --synthetic --ablation
```

### Architecture Comparison

| Architecture | Val Accuracy | F1 | Params | Inference |
|---|---|---|---|---|
| **EfficientNet-B3** ⭐ | **94.2%** | **0.941** | 12M | 22ms |
| ResNet-50 | 91.2% | 0.908 | 25.6M | 28ms |
| EfficientNet-B0 | 88.0% | 0.875 | 5.3M | 12ms |
| MobileNetV3-L | 85.6% | 0.850 | 5.4M | 8ms |

### Learning Rate Sweep

| LR | Val Accuracy |
|---|---|
| 1e-4 | 89.1% |
| **1e-3** ⭐ | **94.2%** |
| 1e-2 | 88.7% |

### Dropout Sweep

| Dropout | Val Accuracy |
|---|---|
| 0.2 | 92.1% |
| **0.4** ⭐ | **94.2%** |
| 0.5 | 93.5% |

---

## Results

Evaluated on held-out test set (15% of State Farm dataset, ~3,300 images):

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | **94.2%** |
| Top-3 Accuracy | 98.1% |
| F1 Macro | 0.941 |
| Precision Macro | 0.938 |
| Recall Macro | 0.941 |
| AUROC | 0.997 |

### Per-Class Accuracy

| Class | Accuracy |
|-------|---------|
| Safe Driving | 97.8% |
| Texting (Right) | 95.1% |
| Phone (Right) | 94.3% |
| Texting (Left) | 93.6% |
| Phone (Left) | 93.9% |
| Radio | 92.4% |
| Drinking | 91.8% |
| Reaching Behind | 94.7% |
| Hair / Makeup | 93.2% |
| Talking | 95.6% |

---

## Explainability — Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which image regions drove each prediction — verifying the model attends to hands, phone, and face rather than spurious background correlations.

```python
from src.explainability.gradcam import ExplainablePredictor
from src.model.architecture import create_model, load_checkpoint

model = create_model("efficientnet_b3", pretrained=False)
model = load_checkpoint(model, "models/best_model.pth")

predictor = ExplainablePredictor(model)
result = predictor.predict("dashcam.jpg", generate_cam=True)

print(result["predicted_label"])   # "Texting (Right Hand)"
print(f"{result['confidence']:.1%}")  # "94.3%"
# result["cam_overlay"] → numpy RGB image with heatmap
```

---

## Web Demo

### Gradio App (recommended)

```bash
python3 webapp/gradio_app.py
# → http://localhost:7860
```

Features:
- Upload dashcam image → instant classification
- **Flashing red/orange/green alert banner** based on risk level
- Grad-CAM overlay showing model attention regions
- Probability distribution chart for all 10 classes
- Risk guide and model info tabs

### Production Web Frontend

```bash
python3 src/inference/api_server.py
# → http://localhost:5000
# Open webapp/templates/index.html in browser
```

---

## REST API

### `POST /predict` — Single image

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@dashcam.jpg"
```

**Response:**
```json
{
  "predicted_class": 1,
  "predicted_label": "Texting (Right Hand)",
  "confidence": 0.9432,
  "is_distracted": true,
  "risk_level": "high",
  "top_k_predictions": [
    {"class_idx": 1, "label": "Texting (Right Hand)", "confidence": 0.9432},
    {"class_idx": 3, "label": "Texting (Left Hand)",  "confidence": 0.0321},
    {"class_idx": 0, "label": "Safe Driving",          "confidence": 0.0121}
  ],
  "cam_overlay_base64": "...",
  "inference_time_ms": 21.4
}
```

### `POST /predict/batch` — Multiple images

```bash
curl -X POST http://localhost:5000/predict/batch \
  -F "files=@img1.jpg" -F "files=@img2.jpg"
```

### `GET /health`

```json
{"status": "healthy", "model_loaded": true, "device": "cuda"}
```

### `GET /classes`

Returns all 10 class definitions with risk levels and colours.

---

## MLOps Pipeline

Implements **MLOps Maturity Level 2** — automated training, centralized tracking, model registry with quality gates.

```
Data Ingestion → Data Processing → Model Training → Model Evaluation → Model Registry
     ↓                ↓                 ↓                 ↓                  ↓
  Log source       Log splits       Log all           Log test          Register if
  & stats          & counts         hyperparams       metrics           gates pass
```

### Run the full pipeline

```bash
# With synthetic data
python3 mlops/pipelines/pipeline.py --config configs/config.yaml --synthetic

# With real data
python3 mlops/pipelines/pipeline.py --config configs/config.yaml

# With ablation study
python3 mlops/pipelines/pipeline.py --config configs/config.yaml --run-ablation
```

### Quality Gates

A model is only registered if it meets minimum thresholds:
- Top-1 Accuracy ≥ 70%
- F1 Macro ≥ 65%

### MLflow Dashboard

```bash
mlflow ui --backend-store-uri mlruns
# → http://localhost:5000
```

Tracks: all hyperparameters, per-epoch train/val metrics, confusion matrices, best model artifacts.

---

## CI/CD Pipeline

Every push to `main` triggers the full pipeline automatically:

```
Push to GitHub
      ↓
✅ Code Quality   (black formatting + flake8 linting)
      ↓
✅ Unit Tests     (22 pytest tests + coverage report)
      ↓
✅ Integration    (synthetic pipeline smoke test)
      ↓
✅ Docker Build   (builds container image)
      ↓
✅ Deploy         (pushes to HuggingFace Spaces)
```

Manual triggers (via GitHub Actions → Run workflow):
- `run_training=true` → full training pipeline
- `run_ablation=true` → ablation study

### Secrets required

Set these in GitHub → Settings → Secrets → Actions:

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | Your HuggingFace write token |

---

## Docker

### Build and run locally

```bash
# Build
docker build -t distracted-driver-detection .

# Run demo
docker run -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  distracted-driver-detection

# Open http://localhost:7860
```

### Full stack with docker-compose

```bash
docker-compose up

# Services:
#   Gradio demo:  http://localhost:7860
#   Flask API:    http://localhost:5000
#   MLflow UI:    http://localhost:5001
```

### Docker image details

| Detail | Value |
|--------|-------|
| Base image | `python:3.10-slim` |
| PyTorch | CPU-only (smaller image) |
| OpenCV | `opencv-python-headless` (no GUI deps) |
| System libs | Fixed for Debian Bookworm (`libgl1` not `libgl1-mesa-glx`) |
| Health check | `curl http://localhost:7860/` every 30s |

---

## Deployment — HuggingFace Spaces

The demo is automatically deployed to HuggingFace Spaces on every push to `main`.

**Live URL:** [huggingface.co/spaces/keerthana-m/distracted-driver-detection](https://huggingface.co/spaces/keerthana-m/distracted-driver-detection)

### Manual deployment

```bash
# Upload model to Space files
python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(
    path_or_fileobj='models/best_model.pth',
    path_in_repo='models/best_model.pth',
    repo_id='keerthana-m/distracted-driver-detection',
    repo_type='space',
)
print('Done')
"
```

### Environment variables on HuggingFace

Set these in Space → Settings → Variables:

| Variable | Value |
|----------|-------|
| `MODEL_PATH` | `models/best_model.pth` |
| `HF_TOKEN` | Your HF token (for model download) |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backbone | EfficientNet-B3 | Best accuracy/compute tradeoff (~12M params) |
| Sampling | WeightedRandomSampler | Handles imbalance without artificial duplication |
| Loss | Label Smoothing CE (ε=0.1) | Prevents overconfident predictions, better calibration |
| Training | Two-phase freeze/unfreeze | Fast head convergence + careful backbone adaptation |
| Explainability | Grad-CAM | Verifies model attends to correct body regions |
| Augmentation | Flip, rotate, jitter, erase | Dataset ~22K — heavy augmentation prevents overfitting |
| Precision | AMP (float16) | 2× faster training without accuracy loss |
| Scheduler | Warmup + cosine annealing | Smooth convergence to flat minima |
| MLflow | SQLite backend | Avoids deprecated filesystem backend warning |

---

## Running Tests

```bash
# All 22 tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Specific test class
pytest tests/test_all.py::TestGradCAM -v

# Skip slow training test
pytest tests/ -k "not test_minimal_training"
```

### Test coverage

| Module | Tests |
|--------|-------|
| Dataset loading & transforms | 6 |
| Model forward pass & checkpoints | 5 |
| Loss function & metrics | 3 |
| Grad-CAM generation | 3 |
| Trainer & early stopping | 3 |
| Visualizations | 1 |
| **Total** | **22** |

---

## Environment Setup

### Using .env file (local development only)

```bash
cp .env.example .env
# Edit .env with your values — never commit this file
```

```env
HF_TOKEN=hf_your_token_here
MODEL_PATH=models/best_model.pth
MODEL_ARCH=efficientnet_b3
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
PORT=7860
```

### Verify your setup

```bash
python3 verify.py
```

---

## Team

| Member | Role |
|--------|------|
| Keerthana Muralidharan | Model architecture + training pipeline + Web demo + API + CI/CD deployment |
| Yashashwini Dinesh | Data pipeline + MLOps infrastructure + Grad-CAM explainability + visualizations |

---

## References

1. Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.
2. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.
3. State Farm Distracted Driver Detection. [Kaggle Competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection).
4. He et al. (2019). Bag of Tricks for Image Classification. *CVPR*.

---

## License

MIT License — see [LICENSE](LICENSE)
