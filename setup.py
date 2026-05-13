"""
setup.py — makes the project installable as a package.
Allows: pip install -e .
Then all imports like 'from src.model.architecture import ...' work anywhere.
"""
from setuptools import setup, find_packages

setup(
    name="distracted-driver-detection",
    version="1.0.0",
    description="Deep Learning Based Distracted Driver Detection for Road Safety",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "torchmetrics>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "mlflow>=2.5.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "gradio>=4.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "huggingface_hub>=0.16.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "tensorboard>=2.13.0",
        ],
        "hf": [
            "opencv-python-headless>=4.8.0",
        ],
    },
)
