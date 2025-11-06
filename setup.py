from setuptools import setup, find_packages

setup(
    name="toys",
    version="0.1.0",
    description="Lightweight decision oracles using PCA + Predictor models",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "requests>=2.25.0",
    ],
    python_requires=">=3.8",
)
