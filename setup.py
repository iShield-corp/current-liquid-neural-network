"""
Setup script for the Hybrid Liquid-Spiking Neural Network System.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hybrid-liquid-spiking-nn",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A hybrid neural network combining Liquid Neural Networks with Spiking Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hybrid-liquid-spiking-nn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
            "jupyter>=1.0",
            "tensorboard>=2.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
        "neuromorphic": [
            "lava-loihi>=0.3",
            "intel-nrc-lib",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybrid-nn=scripts.cli:main",
            "hybrid-train=train:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.py"],
        "scripts": ["*.py"],
        "tests": ["*.py"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hybrid-liquid-spiking-nn/issues",
        "Source": "https://github.com/yourusername/hybrid-liquid-spiking-nn",
        "Documentation": "https://github.com/yourusername/hybrid-liquid-spiking-nn/wiki",
    },
)
