"""
Curriculum Learning Module for Hybrid Liquid-Spiking Neural Networks
Implements progressive difficulty scheduling and adaptive pacing
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CurriculumStrategy(Enum):
    """Different curriculum learning strategies."""
    LINEAR = "linear"                    # Linear progression
    EXPONENTIAL = "exponential"          # Exponential growth
    STEP = "step"                        # Step-wise increases
    ADAPTIVE = "adaptive"                # Based on performance
    SELF_PACED = "self_paced"           # Model determines pace


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    strategy: CurriculumStrategy = CurriculumStrategy.LINEAR
    
    # Sequence length curriculum
    initial_seq_length: int = 16
    final_seq_length: int = 256
    seq_growth_rate: float = 1.2
    
    # Spike steps curriculum
    initial_spike_steps: int = 4
    final_spike_steps: int = 32
    spike_growth_rate: float = 1.15
    
    # Complexity curriculum (task-specific)
    initial_difficulty: float = 0.1
    final_difficulty: float = 1.0
    difficulty_growth_rate: float = 0.05
    
    # Adaptive parameters
    performance_threshold: float = 0.7  # Accuracy threshold to advance
    patience: int = 3                    # Epochs to wait before advancing
    warmup_epochs: int = 5               # Initial epochs at easiest level
    
    # Self-paced parameters
    learning_rate: float = 0.1
    age_parameter: float = 0.5


class CurriculumScheduler:
    """
    Main curriculum learning scheduler that manages difficulty progression.
    """
    
    def __init__(self, config: CurriculumConfig, total_epochs: int):
        """
        Args:
            config: Curriculum configuration
            total_epochs: Total number of training epochs
        """
        self.config = config
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Track current curriculum state
        self.current_seq_length = config.initial_seq_length
        self.current_spike_steps = config.initial_spike_steps
        self.current_difficulty = config.initial_difficulty
        
        # Performance tracking for adaptive strategies
        self.performance_history = []
        self.epochs_at_current_level = 0
        self.consecutive_success = 0
        
        logger.info(f"📚 Curriculum learning initialized: {config.strategy.value}")
        logger.info(f"   Sequence: {config.initial_seq_length} → {config.final_seq_length}")
        logger.info(f"   Spikes: {config.initial_spike_steps} → {config.final_spike_steps}")
    
    def step(self, epoch: int, performance_metric: Optional[float] = None) -> Dict[str, any]:
        """
        Update curriculum based on current epoch and performance.
        
        Args:
            epoch: Current training epoch
            performance_metric: Current performance (e.g., accuracy, loss)
        
        Returns:
            Dictionary with current curriculum parameters
        """
        self.current_epoch = epoch
        
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
        
        # Apply strategy-specific updates
        if self.config.strategy == CurriculumStrategy.LINEAR:
            self._update_linear()
        elif self.config.strategy == CurriculumStrategy.EXPONENTIAL:
            self._update_exponential()
        elif self.config.strategy == CurriculumStrategy.STEP:
            self._update_step()
        elif self.config.strategy == CurriculumStrategy.ADAPTIVE:
            self._update_adaptive(performance_metric)
        elif self.config.strategy == CurriculumStrategy.SELF_PACED:
            self._update_self_paced(performance_metric)
        
        self.epochs_at_current_level += 1
        
        return self.get_current_params()
    
    def _update_linear(self):
        """Linear progression through curriculum."""
        if self.current_epoch < self.config.warmup_epochs:
            return
        
        progress = (self.current_epoch - self.config.warmup_epochs) / (
            self.total_epochs - self.config.warmup_epochs
        )
        progress = min(1.0, progress)
        
        # Update sequence length
        self.current_seq_length = int(
            self.config.initial_seq_length +
            progress * (self.config.final_seq_length - self.config.initial_seq_length)
        )
        
        # Update spike steps
        self.current_spike_steps = int(
            self.config.initial_spike_steps +
            progress * (self.config.final_spike_steps - self.config.initial_spike_steps)
        )
        
        # Update difficulty
        self.current_difficulty = (
            self.config.initial_difficulty +
            progress * (self.config.final_difficulty - self.config.initial_difficulty)
        )
    
    def _update_exponential(self):
        """Exponential progression - faster initial growth."""
        if self.current_epoch < self.config.warmup_epochs:
            return
        
        epochs_since_warmup = self.current_epoch - self.config.warmup_epochs
        
        # Sequence length
        if self.current_seq_length < self.config.final_seq_length:
            self.current_seq_length = min(
                int(self.current_seq_length * self.config.seq_growth_rate),
                self.config.final_seq_length
            )
        
        # Spike steps
        if self.current_spike_steps < self.config.final_spike_steps:
            self.current_spike_steps = min(
                int(self.current_spike_steps * self.config.spike_growth_rate),
                self.config.final_spike_steps
            )
        
        # Difficulty
        self.current_difficulty = min(
            self.current_difficulty + self.config.difficulty_growth_rate,
            self.config.final_difficulty
        )
    
    def _update_step(self):
        """Step-wise increases at specific epochs."""
        # Define steps (customize these based on your needs)
        step_epochs = [
            self.config.warmup_epochs,
            self.total_epochs // 4,
            self.total_epochs // 2,
            3 * self.total_epochs // 4
        ]
        
        step_levels = [
            (self.config.initial_seq_length, self.config.initial_spike_steps, 0.25),
            (self.config.initial_seq_length * 2, self.config.initial_spike_steps * 2, 0.50),
            (self.config.initial_seq_length * 3, self.config.initial_spike_steps * 3, 0.75),
            (self.config.final_seq_length, self.config.final_spike_steps, 1.0)
        ]
        
        for i, epoch_threshold in enumerate(step_epochs):
            if self.current_epoch >= epoch_threshold:
                seq_len, spike_steps, difficulty = step_levels[min(i, len(step_levels) - 1)]
                self.current_seq_length = seq_len
                self.current_spike_steps = spike_steps
                self.current_difficulty = difficulty
    
    def _update_adaptive(self, performance_metric: Optional[float]):
        """Adaptive progression based on model performance."""
        if self.current_epoch < self.config.warmup_epochs:
            return
        
        if performance_metric is None:
            return
        
        # Check if performance meets threshold
        if performance_metric >= self.config.performance_threshold:
            self.consecutive_success += 1
        else:
            self.consecutive_success = 0
        
        # Advance curriculum if consistent good performance
        if self.consecutive_success >= self.config.patience:
            self._advance_curriculum()
            self.consecutive_success = 0
            self.epochs_at_current_level = 0
            logger.info(f"📈 Curriculum advanced at epoch {self.current_epoch}")
    
    def _update_self_paced(self, performance_metric: Optional[float]):
        """Self-paced learning where model controls progression."""
        if performance_metric is None or len(self.performance_history) < 2:
            return
        
        # Calculate learning progress (improvement rate)
        recent_improvement = (
            self.performance_history[-1] - self.performance_history[-2]
        )
        
        # Adjust curriculum based on learning progress
        if recent_improvement > 0:
            # Model is learning well, can increase difficulty
            pacing_factor = 1.0 + self.config.learning_rate * recent_improvement
        else:
            # Model is struggling, slow down or maintain
            pacing_factor = 1.0 - self.config.learning_rate * abs(recent_improvement)
        
        pacing_factor = np.clip(pacing_factor, 0.95, 1.15)
        
        # Apply pacing to curriculum progression
        self.current_difficulty = min(
            self.current_difficulty * pacing_factor,
            self.config.final_difficulty
        )
        
        # Update complexity parameters accordingly
        progress = (self.current_difficulty - self.config.initial_difficulty) / (
            self.config.final_difficulty - self.config.initial_difficulty
        )
        
        self.current_seq_length = int(
            self.config.initial_seq_length +
            progress * (self.config.final_seq_length - self.config.initial_seq_length)
        )
        
        self.current_spike_steps = int(
            self.config.initial_spike_steps +
            progress * (self.config.final_spike_steps - self.config.initial_spike_steps)
        )
    
    def _advance_curriculum(self):
        """Manually advance to next difficulty level."""
        # Increase sequence length
        self.current_seq_length = min(
            int(self.current_seq_length * 1.5),
            self.config.final_seq_length
        )
        
        # Increase spike steps
        self.current_spike_steps = min(
            int(self.current_spike_steps * 1.3),
            self.config.final_spike_steps
        )
        
        # Increase difficulty
        self.current_difficulty = min(
            self.current_difficulty + 0.2,
            self.config.final_difficulty
        )
    
    def get_current_params(self) -> Dict[str, any]:
        """Get current curriculum parameters."""
        return {
            'epoch': self.current_epoch,
            'sequence_length': self.current_seq_length,
            'num_spike_steps': self.current_spike_steps,
            'difficulty': self.current_difficulty,
            'epochs_at_level': self.epochs_at_current_level
        }
    
    def should_filter_sample(self, sample_difficulty: float) -> bool:
        """
        Determine if a training sample should be used based on current difficulty.
        
        Args:
            sample_difficulty: Difficulty rating of the sample (0.0 to 1.0)
        
        Returns:
            True if sample should be included in training
        """
        # Include samples up to current difficulty level
        return sample_difficulty <= self.current_difficulty
    
    def get_sample_weight(self, sample_difficulty: float) -> float:
        """
        Get training weight for a sample based on curriculum.
        
        Args:
            sample_difficulty: Difficulty rating of the sample (0.0 to 1.0)
        
        Returns:
            Weight multiplier for the sample's loss
        """
        if sample_difficulty > self.current_difficulty:
            return 0.0
        
        # Weight samples closer to current difficulty higher
        difficulty_distance = abs(sample_difficulty - self.current_difficulty)
        weight = np.exp(-difficulty_distance * 2.0)
        
        return weight


class CurriculumDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that applies curriculum learning filtering and weighting.
    """
    
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        difficulty_fn: Callable[[any], float],
        curriculum_scheduler: CurriculumScheduler
    ):
        """
        Args:
            base_dataset: Original dataset
            difficulty_fn: Function that computes difficulty for each sample
            curriculum_scheduler: Curriculum scheduler
        """
        self.base_dataset = base_dataset
        self.difficulty_fn = difficulty_fn
        self.scheduler = curriculum_scheduler
        
        # Pre-compute difficulties for all samples
        self.difficulties = self._compute_difficulties()
        
        # Track which samples are currently included
        self.update_curriculum()
    
    def _compute_difficulties(self) -> List[float]:
        """Pre-compute difficulty for all samples."""
        difficulties = []
        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]
            difficulty = self.difficulty_fn(sample)
            difficulties.append(difficulty)
        return difficulties
    
    def update_curriculum(self):
        """Update which samples are included based on current curriculum."""
        current_difficulty = self.scheduler.current_difficulty
        
        # Filter samples by difficulty
        self.valid_indices = [
            idx for idx, diff in enumerate(self.difficulties)
            if diff <= current_difficulty
        ]
        
        logger.info(f"📊 Curriculum update: {len(self.valid_indices)}/{len(self.difficulties)} samples available")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map to valid index
        actual_idx = self.valid_indices[idx]
        sample = self.base_dataset[actual_idx]
        
        # Add curriculum weight
        difficulty = self.difficulties[actual_idx]
        weight = self.scheduler.get_sample_weight(difficulty)
        
        # Return sample with curriculum weight
        if isinstance(sample, dict):
            sample['curriculum_weight'] = weight
        else:
            # Assume tuple format (data, label)
            sample = (*sample, weight)
        
        return sample


def integrate_curriculum_with_trainer(
    trainer,
    strategy: str = "adaptive",
    enable_curriculum: bool = True
):
    """
    Integrate curriculum learning into existing Trainer class.
    
    Args:
        trainer: Trainer instance
        strategy: Curriculum strategy ('linear', 'exponential', 'step', 'adaptive', 'self_paced')
        enable_curriculum: Whether to enable curriculum learning
    """
    if not enable_curriculum:
        return
    
    # Create curriculum config
    curriculum_config = CurriculumConfig(
        strategy=CurriculumStrategy(strategy),
        initial_seq_length=trainer.config.sequence_length // 4,
        final_seq_length=trainer.config.sequence_length,
        initial_spike_steps=trainer.config.num_spike_steps // 4,
        final_spike_steps=trainer.config.num_spike_steps,
        warmup_epochs=5,
        performance_threshold=0.7
    )
    
    # Create curriculum scheduler
    trainer.curriculum_scheduler = CurriculumScheduler(
        config=curriculum_config,
        total_epochs=trainer.total_epochs if hasattr(trainer, 'total_epochs') else 50
    )
    
    # Wrap dataset with curriculum filtering (if needed)
    # This would require implementing difficulty functions for your specific tasks
    
    logger.info(f"📚 Curriculum learning enabled: {strategy}")
    
    # Add hook to update curriculum each epoch
    original_train_epoch = trainer.train_epoch
    
    def curriculum_train_epoch(train_loader):
        """Wrapped training epoch with curriculum updates."""
        # Get current curriculum parameters
        curriculum_params = trainer.curriculum_scheduler.get_current_params()
        
        # Update model configuration if needed
        if hasattr(trainer.config, 'sequence_length'):
            trainer.config.sequence_length = curriculum_params['sequence_length']
        if hasattr(trainer.config, 'num_spike_steps'):
            trainer.config.num_spike_steps = curriculum_params['num_spike_steps']
        
        # Update model's spike steps dynamically
        if hasattr(trainer.model, 'num_spike_steps'):
            trainer.model.num_spike_steps = curriculum_params['num_spike_steps']
        
        # Call original training
        loss, grad_norm = original_train_epoch(train_loader)
        
        # Update curriculum based on performance
        # Use validation accuracy or inverse of loss as performance metric
        performance = 1.0 / (1.0 + loss)  # Simple performance metric
        trainer.curriculum_scheduler.step(
            epoch=trainer.current_epoch if hasattr(trainer, 'current_epoch') else 0,
            performance_metric=performance
        )
        
        # Log curriculum progress
        logger.info(f"📚 Curriculum: seq_len={curriculum_params['sequence_length']}, "
                   f"spike_steps={curriculum_params['num_spike_steps']}, "
                   f"difficulty={curriculum_params['difficulty']:.2f}")
        
        return loss, grad_norm
    
    trainer.train_epoch = curriculum_train_epoch


# Example difficulty functions for different tasks
def llm_difficulty(sample: Dict) -> float:
    """Compute difficulty for LLM samples based on sequence complexity."""
    if 'input_ids' not in sample:
        return 0.5
    
    input_ids = sample['input_ids']
    
    # Factors: length, vocabulary diversity, rare tokens
    length_factor = len(input_ids) / 512.0
    unique_tokens = len(set(input_ids.tolist()))
    diversity_factor = unique_tokens / len(input_ids)
    
    difficulty = (length_factor * 0.6 + diversity_factor * 0.4)
    return np.clip(difficulty, 0.0, 1.0)


def vision_difficulty(sample: Tuple) -> float:
    """Compute difficulty for vision samples based on image complexity."""
    image, label = sample[0], sample[1]
    
    # Factors: edge density, color variance, spatial frequency
    if isinstance(image, torch.Tensor):
        image_np = image.numpy()
    else:
        image_np = np.array(image)
    
    # Simple difficulty based on standard deviation (more complex = higher std)
    complexity = np.std(image_np)
    difficulty = complexity / 255.0  # Normalize
    
    return np.clip(difficulty, 0.0, 1.0)


def robotics_difficulty(sample: Dict) -> float:
    """Compute difficulty for robotics samples based on trajectory complexity."""
    if 'trajectory' not in sample:
        return 0.5
    
    trajectory = sample['trajectory']
    
    # Factors: trajectory length, curvature, velocity changes
    length_factor = len(trajectory) / 100.0
    velocity_changes = np.diff(trajectory, axis=0).std()
    
    difficulty = (length_factor * 0.5 + velocity_changes * 0.5)
    return np.clip(difficulty, 0.0, 1.0)