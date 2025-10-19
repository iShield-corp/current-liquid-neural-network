"""
Hybrid Liquid-Spiking Neural Network System

A cutting-edge implementation combining Liquid Neural Networks (LNNs) with 
Spiking Neural Networks (SNNs) for efficient, adaptive AI processing.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.main import (
    LiquidSpikingNetwork,
    LiquidSpikingTrainer,
    ModelConfig,
    TaskType,
    train_llm_model,
    train_vision_model,
    train_robotics_model,
    load_model,
    generate_text,
    evaluate_perplexity
)

from .datasets.advanced_programming_datasets import (
    AdvancedProgrammingDataset,
    ProgrammingDatasetConfig,
    ProgrammingDatasetFactory
)

__all__ = [
    "LiquidSpikingNetwork",
    "LiquidSpikingTrainer", 
    "ModelConfig",
    "TaskType",
    "train_llm_model",
    "train_vision_model", 
    "train_robotics_model",
    "load_model",
    "generate_text",
    "evaluate_perplexity",
    "AdvancedProgrammingDataset",
    "ProgrammingDatasetConfig",
    "ProgrammingDatasetFactory"
]
