"""
Real Datasets Module for Liquid-Spiking Neural Networks

This module provides comprehensive real-world datasets for vision, robotics,
and LLM tasks without shortcuts, fallback logic, or mock data.
"""

from .advanced_programming_datasets import (
    AdvancedProgrammingDataset,
    ProgrammingDatasetConfig,
    ProgrammingDatasetFactory
)

from .vision_datasets import (
    VisionDatasetConfig,
    RealVisionDataset, 
    VisionDatasetFactory,
    create_real_vision_dataset
)

from .robotics_datasets import (
    RoboticsDatasetConfig,
    RealRoboticsDataset,
    RoboticsDatasetFactory, 
    create_real_robotics_dataset
)

__all__ = [
    # LLM datasets (existing)
    "AdvancedProgrammingDataset",
    "ProgrammingDatasetConfig", 
    "ProgrammingDatasetFactory",
    
    # Vision datasets (new)
    'VisionDatasetConfig',
    'RealVisionDataset',
    'VisionDatasetFactory', 
    'create_real_vision_dataset',
    
    # Robotics datasets (new)
    'RoboticsDatasetConfig',
    'RealRoboticsDataset',
    'RoboticsDatasetFactory',
    'create_real_robotics_dataset'
]
