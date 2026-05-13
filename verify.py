#!/usr/bin/env python3
"""
verify.py — Run this to check everything is installed correctly.
Usage: python3 verify.py
"""

import sys
import importlib

print("=" * 55)
print("  Distracted Driver Detection — Environment Check")
print("=" * 55)

# Check Python version
major, minor = sys.version_info[:2]
status = "✅" if (major == 3 and minor >= 10) else "❌"
print(f"\n{status} Python {major}.{minor} (need 3.10+)")

# Required packages with min versions
packages = [
    ("torch",          "2.0.0",  "PyTorch"),
    ("torchvision",    "0.15.0", "TorchVision"),
    ("timm",           "0.9.0",  "timm (backbones)"),
    ("torchmetrics",   "1.0.0",  "TorchMetrics"),
    ("numpy",          "1.24.0", "NumPy"),
    ("pandas",         "2.0.0",  "Pandas"),
    ("sklearn",        "1.3.0",  "Scikit-learn"),
    ("PIL",            "10.0.0", "Pillow"),
    ("cv2",            "4.8.0",  "OpenCV"),
    ("matplotlib",     "3.7.0",  "Matplotlib"),
    ("seaborn",        "0.12.0", "Seaborn"),
    ("mlflow",         "2.5.0",  "MLflow"),
    ("gradio",         "4.0.0",  "Gradio"),
    ("flask",          "2.3.0",  "Flask"),
    ("yaml",           None,     "PyYAML"),
    ("tqdm",           "4.65.0", "tqdm"),
    ("huggingface_hub","0.16.0", "HuggingFace Hub"),
    ("dotenv",         None,     "python-dotenv"),
]

print("\nPackage versions:")
all_ok = True
for pkg, min_ver, label in packages:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  ✅ {label:25s} {ver}")
    except ImportError:
        print(f"  ❌ {label:25s} NOT INSTALLED")
        all_ok = False

# Check CUDA
print("\nHardware:")
try:
    import torch
    cuda = torch.cuda.is_available()
    device = f"CUDA ({torch.cuda.get_device_name(0)})" if cuda else "CPU only"
    print(f"  {'✅' if cuda else 'ℹ️ '} Device: {device}")
except Exception:
    print("  ❌ Could not check CUDA")

# Check src imports
print("\nProject imports:")
try:
    import importlib.util, os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ".")
    from src.data.dataset import DistractedDriverDataset
    from src.model.architecture import create_model
    from src.explainability.gradcam import ExplainablePredictor
    print("  ✅ src.data.dataset")
    print("  ✅ src.model.architecture")
    print("  ✅ src.explainability.gradcam")
except Exception as e:
    print(f"  ❌ Import error: {e}")
    all_ok = False

# Check model file
print("\nModel:")
from pathlib import Path
model_path = Path("models/best_model.pth")
if model_path.exists():
    size_mb = model_path.stat().st_size / 1e6
    print(f"  ✅ best_model.pth ({size_mb:.1f} MB)")
else:
    print("  ⚠️  models/best_model.pth not found")
    print("     Run training first: python3 scripts/train.py --synthetic")

print("\n" + "=" * 55)
if all_ok:
    print("  ✅ All checks passed! Ready to run.")
    print("\n  Start demo:    python3 webapp/gradio_app.py")
    print("  Run pipeline:  python3 mlops/pipelines/pipeline.py --synthetic")
else:
    print("  ❌ Some packages missing. Run:")
    print("     bash setup.sh")
    print("  or:")
    print("     pip3 install -r requirements.txt")
print("=" * 55)
