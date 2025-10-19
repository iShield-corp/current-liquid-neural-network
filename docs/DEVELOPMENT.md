# Development Guide

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd hybrid-liquid-spiking-nn

# Create virtual environment
python -m venv nn
source nn/bin/activate  # On Windows: nn\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Directory Structure
```
project_root/
├── src/                    # Core source code
│   ├── core/              # Main neural network implementations
│   ├── datasets/          # Dataset loading and processing
│   └── training/          # Training scripts and utilities
├── scripts/               # Command-line tools
├── tests/                 # Test suite
├── models/                # Saved model files
├── cache/                 # Cached data and temporary files
├── experiments/           # Experiment results and logs
└── docs/                  # Additional documentation
```

## Development Workflow

### Running Training
```bash
# Quick start - interactive mode
python train.py

# Basic language model training
python train.py llm

# Optimized training
python train.py llm_optimized

# Advanced training options
python train.py cli train --help
```

### Using the CLI
```bash
# Train a model
python scripts/cli.py train --task vision --epochs 20

# Run inference
python scripts/cli.py inference --model-path models/my_model.pt

# Benchmark performance
python scripts/cli.py benchmark --model-path models/my_model.pt
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python tests/test_dataset_integration.py

# Run test suite script
python tests/run_tests.py
```

## Code Organization

### Import Structure
```python
# Import from core modules
from src.core.main import LiquidSpikingNetwork
from src.datasets.advanced_programming_datasets import ProgrammingDatasetConfig
from src.training.train_llm_optimized import main as train_optimized
```

### Adding New Features

1. **Core functionality**: Add to `src/core/`
2. **Dataset handling**: Add to `src/datasets/`
3. **Training scripts**: Add to `src/training/`
4. **CLI commands**: Extend `scripts/cli.py`
5. **Tests**: Add to `tests/`

### Development Tools

#### Code Quality
```bash
# Install development dependencies
pip install -e .[dev]

# Format code
black src/ scripts/ tests/

# Lint code
flake8 src/ scripts/ tests/

# Type checking
mypy src/
```

#### Jupyter Notebooks
```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook
```

## Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export WANDB_PROJECT=hybrid-nn  # Weights & Biases project
export HF_DATASETS_CACHE=./cache/huggingface  # HuggingFace cache
```

### Model Configuration
Models are configured through Python dictionaries or YAML files. See examples in the CLI tool:
```bash
python scripts/cli.py config --help
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the project root directory
2. **CUDA errors**: Check GPU availability and PyTorch installation
3. **Memory issues**: Reduce batch size or model size
4. **Dataset loading**: Check internet connection and cache directory

### Debug Mode
```bash
# Enable verbose logging
python scripts/cli.py train --debug --task vision

# Run with profiling
python -m cProfile scripts/cli.py train --task vision
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run the test suite
5. Submit a pull request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Write tests for new features
- Keep functions small and focused

### Commit Messages
```
feat: add new dataset loader
fix: resolve memory leak in training
docs: update API documentation
test: add integration tests
refactor: simplify model architecture
```

## Building and Distribution

### Creating a Package
```bash
# Build the package
python -m build

# Install from local build
pip install dist/hybrid_liquid_spiking_nn-*.whl
```

### Running from Package
Once installed, you can use console commands:
```bash
hybrid-nn train --task vision
hybrid-train llm_optimized
```

## Advanced Usage

### Custom Models
Extend the base `LiquidSpikingNetwork` class:
```python
from src.core.main import LiquidSpikingNetwork

class CustomNetwork(LiquidSpikingNetwork):
    def __init__(self, config):
        super().__init__(config)
        # Add custom layers
```

### Custom Datasets
Implement dataset loaders in `src/datasets/`:
```python
from src.datasets.advanced_programming_datasets import BaseDatasetLoader

class CustomDataset(BaseDatasetLoader):
    def load_data(self):
        # Implement custom loading logic
        pass
```

### Experiment Tracking
The system integrates with:
- Weights & Biases (wandb)
- TensorBoard
- Custom logging

Configure in your training script or use CLI options.
