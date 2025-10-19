# Tests for Liquid-Spiking Neural Networks

This directory contains all test files for the liquid-spiking neural network project.

## Test Files

### Core Tests
- **`test_setup.py`** - Basic setup verification and component testing
- **`run_tests.py`** - Main test runner script

### Generation Tests  
- **`test_generation.py`** - Basic text generation testing
- **`demo_generation.py`** - Focused demonstration of model capabilities
- **`comprehensive_generation_test.py`** - Comprehensive generation analysis

### Evaluation Tests
- **`evaluate_model.py`** - Model performance evaluation and metrics

## Running Tests

### Run All Tests
```bash
cd /home/sovr610/ssn-cfc
source nn/bin/activate
python tests/run_tests.py
```

### Run Specific Tests
```bash
# Setup test only
python tests/run_tests.py --test setup

# Text generation test only  
python tests/run_tests.py --test generation

# Model evaluation only
python tests/run_tests.py --test evaluation

# Demo only
python tests/run_tests.py --test demo
```

### Run Individual Test Files
```bash
# Basic setup verification
python tests/test_setup.py

# Simple text generation test
python tests/test_generation.py

# Model evaluation
python tests/evaluate_model.py

# Demonstration
python tests/demo_generation.py

# Comprehensive generation testing
python tests/comprehensive_generation_test.py
```

## Test Requirements

All tests require:
- Trained model checkpoints (`best_model_epoch_*.pt`) in the parent directory
- Active virtual environment with all dependencies
- CUDA-capable GPU (recommended)

## Test Coverage

✅ **Model Architecture**: Verifies liquid-spiking hybrid network creation  
✅ **Text Processing**: Tests real text tokenization and dataset handling  
✅ **Training Pipeline**: Validates training loop functionality  
✅ **Text Generation**: Tests inference and text generation capabilities  
✅ **Model Evaluation**: Measures perplexity and performance metrics  
✅ **Integration**: End-to-end pipeline testing  

## Expected Results

- **Setup Test**: All components load and basic forward pass works
- **Generation Test**: Model produces coherent text from prompts  
- **Evaluation Test**: Perplexity decreases across training epochs
- **Demo**: Showcases real liquid-spiking neural network LLM capabilities

All tests use **real neural network computation** with **no shortcuts, mock data, or fallbacks**.
