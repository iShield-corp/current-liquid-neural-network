"""
Complete Integration Guide for Meta-Learning and Curriculum Learning
Shows how to integrate both features into your existing training pipeline
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

# Import the modules (assuming they're in the same directory)
from meta_learning import LiquidMAML, TaskSampler, integrate_maml_with_trainer
from curriculum_learning import (
    CurriculumScheduler, CurriculumConfig, CurriculumStrategy,
    integrate_curriculum_with_trainer, llm_difficulty, vision_difficulty
)

logger = logging.getLogger(__name__)


class EnhancedTrainer:
    """
    Enhanced Trainer class that integrates both meta-learning and curriculum learning.
    This wraps your existing Trainer from main.py
    """
    
    def __init__(
        self,
        base_trainer,
        enable_meta_learning: bool = False,
        enable_curriculum: bool = False,
        curriculum_strategy: str = "adaptive",
        meta_learning_config: Optional[Dict] = None,
        curriculum_config: Optional[Dict] = None
    ):
        """
        Args:
            base_trainer: Your existing Trainer instance
            enable_meta_learning: Enable MAML meta-learning
            enable_curriculum: Enable curriculum learning
            curriculum_strategy: Strategy for curriculum ('linear', 'adaptive', etc.)
            meta_learning_config: Custom MAML configuration
            curriculum_config: Custom curriculum configuration
        """
        self.trainer = base_trainer
        self.enable_meta = enable_meta_learning
        self.enable_curriculum = enable_curriculum
        
        # Initialize meta-learning if enabled
        if self.enable_meta:
            self._setup_meta_learning(meta_learning_config)
        
        # Initialize curriculum learning if enabled
        if self.enable_curriculum:
            self._setup_curriculum_learning(curriculum_strategy, curriculum_config)
        
        logger.info("✨ Enhanced trainer initialized")
        logger.info(f"   Meta-learning: {'✓' if enable_meta_learning else '✗'}")
        logger.info(f"   Curriculum: {'✓' if enable_curriculum else '✗'}")
    
    def _setup_meta_learning(self, config: Optional[Dict]):
        """Setup meta-learning components."""
        meta_config = config or {}
        
        self.maml = LiquidMAML(
            model=self.trainer.model,
            meta_lr=meta_config.get('meta_lr', self.trainer.config.learning_rate),
            inner_lr=meta_config.get('inner_lr', self.trainer.config.learning_rate * 10),
            num_inner_steps=meta_config.get('num_inner_steps', 5),
            first_order=meta_config.get('first_order', False),
            adapt_liquid_only=meta_config.get('adapt_liquid_only', True),
            device=self.trainer.device
        )
        
        # Setup task sampler based on task type
        if hasattr(self.trainer, 'train_dataset'):
            self.task_sampler = TaskSampler(
                dataset=self.trainer.train_dataset,
                n_way=meta_config.get('n_way', 5),
                k_shot=meta_config.get('k_shot', 5),
                query_size=meta_config.get('query_size', 15),
                task_batch_size=meta_config.get('task_batch_size', 4)
            )
        
        logger.info("🧠 Meta-learning (MAML) configured")
    
    def _setup_curriculum_learning(self, strategy: str, config: Optional[Dict]):
        """Setup curriculum learning components."""
        curr_config = config or {}
        
        curriculum_cfg = CurriculumConfig(
            strategy=CurriculumStrategy(strategy),
            initial_seq_length=curr_config.get(
                'initial_seq_length',
                self.trainer.config.sequence_length // 4
            ),
            final_seq_length=curr_config.get(
                'final_seq_length',
                self.trainer.config.sequence_length
            ),
            initial_spike_steps=curr_config.get(
                'initial_spike_steps',
                getattr(self.trainer.config, 'num_spike_steps', 32) // 4
            ),
            final_spike_steps=curr_config.get(
                'final_spike_steps',
                getattr(self.trainer.config, 'num_spike_steps', 32)
            ),
            warmup_epochs=curr_config.get('warmup_epochs', 5),
            performance_threshold=curr_config.get('performance_threshold', 0.7),
            patience=curr_config.get('patience', 3)
        )
        
        # Get total epochs from trainer
        total_epochs = curr_config.get('total_epochs', 50)
        
        self.curriculum = CurriculumScheduler(
            config=curriculum_cfg,
            total_epochs=total_epochs
        )
        
        logger.info(f"📚 Curriculum learning configured: {strategy}")
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        meta_learning_frequency: int = 5
    ):
        """
        Enhanced training loop with meta-learning and curriculum learning.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            meta_learning_frequency: How often to perform meta-learning (every N epochs)
        """
        logger.info(f"🚀 Starting enhanced training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # 1. Curriculum Learning Update
            if self.enable_curriculum:
                self._update_curriculum(epoch)
            
            # 2. Regular Training Epoch
            train_loss, grad_norm = self._train_epoch_with_curriculum(
                train_loader, epoch
            )
            
            # 3. Meta-Learning Step (periodic)
            if self.enable_meta and (epoch + 1) % meta_learning_frequency == 0:
                self._meta_learning_step(epoch)
            
            # 4. Validation
            val_loss, val_accuracy = self._validate(val_loader)
            
            # 5. Update Curriculum Based on Performance
            if self.enable_curriculum:
                self.curriculum.step(epoch, performance_metric=val_accuracy)
            
            # 6. Learning Rate Scheduling
            if hasattr(self.trainer, 'scheduler'):
                self.trainer.scheduler.step()
                if hasattr(self.trainer, 'plateau_scheduler'):
                    self.trainer.plateau_scheduler.step(val_loss)
            
            # 7. Logging
            self._log_epoch_results(epoch, train_loss, val_loss, val_accuracy, grad_norm)
            
            # 8. Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)
        
        logger.info(f"\n🎉 Enhanced training completed!")
        logger.info(f"📈 Best validation loss: {best_val_loss:.4f}")
    
    def _update_curriculum(self, epoch: int):
        """Update curriculum parameters for the current epoch."""
        params = self.curriculum.get_current_params()
        
        # Update model configuration
        if hasattr(self.trainer.config, 'sequence_length'):
            old_seq = self.trainer.config.sequence_length
            self.trainer.config.sequence_length = params['sequence_length']
            
            if old_seq != params['sequence_length']:
                logger.info(f"📏 Sequence length: {old_seq} → {params['sequence_length']}")
        
        if hasattr(self.trainer.config, 'num_spike_steps'):
            old_spike = self.trainer.config.num_spike_steps
            self.trainer.config.num_spike_steps = params['num_spike_steps']
            
            if old_spike != params['num_spike_steps']:
                logger.info(f"⚡ Spike steps: {old_spike} → {params['num_spike_steps']}")
        
        # Update model directly if it has these attributes
        if hasattr(self.trainer.model, 'num_spike_steps'):
            self.trainer.model.num_spike_steps = params['num_spike_steps']
        
        logger.info(f"📊 Curriculum difficulty: {params['difficulty']:.2f}")
    
    def _train_epoch_with_curriculum(self, train_loader, epoch: int):
        """Train one epoch with curriculum-weighted samples."""
        self.trainer.model.train()
        
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle curriculum weights if present
            if isinstance(batch, tuple) and len(batch) == 3:
                data, targets, weights = batch
            elif isinstance(batch, dict):
                data = batch.get('input_ids', batch.get('data'))
                targets = batch.get('labels', batch.get('targets'))
                weights = batch.get('curriculum_weight', torch.ones(len(data)))
            else:
                data, targets = batch
                weights = torch.ones(len(data))
            
            # Move to device
            data = data.to(self.trainer.device)
            targets = targets.to(self.trainer.device)
            weights = weights.to(self.trainer.device)
            
            # Forward pass
            self.trainer.optimizer.zero_grad()
            
            if self.trainer.config.mixed_precision and self.trainer.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.trainer.model(data)
                    loss = self.trainer._compute_loss(outputs, targets)
                    
                    # Apply curriculum weights
                    if weights is not None:
                        loss = (loss * weights.unsqueeze(-1)).mean()
                
                self.trainer.scaler.scale(loss).backward()
                
                if self.trainer.config.gradient_clip > 0:
                    self.trainer.scaler.unscale_(self.trainer.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.trainer.model.parameters(),
                        self.trainer.config.gradient_clip
                    )
                else:
                    grad_norm = 0.0
                
                self.trainer.scaler.step(self.trainer.optimizer)
                self.trainer.scaler.update()
            else:
                outputs = self.trainer.model(data)
                loss = self.trainer._compute_loss(outputs, targets)
                
                # Apply curriculum weights
                if weights is not None:
                    loss = (loss * weights.unsqueeze(-1)).mean()
                
                loss.backward()
                
                if self.trainer.config.gradient_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.trainer.model.parameters(),
                        self.trainer.config.gradient_clip
                    )
                else:
                    grad_norm = 0.0
                
                self.trainer.optimizer.step()
            
            total_loss += loss.item()
            total_grad_norm += grad_norm if isinstance(grad_norm, float) else grad_norm.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_grad_norm = total_grad_norm / num_batches
        
        return avg_loss, avg_grad_norm
    
    def _meta_learning_step(self, epoch: int):
        """Perform a meta-learning step."""
        logger.info("🧠 Performing meta-learning step...")
        
        # Sample batch of tasks
        task_batch = self.task_sampler.sample_task_batch()
        
        # Meta-training step
        metrics = self.maml.meta_train_step(task_batch, self.trainer.criterion)
        
        logger.info(f"   Meta-loss: {metrics['meta_loss']:.4f}")
        logger.info(f"   Avg task loss: {metrics['avg_task_loss']:.4f}")
        logger.info(f"   Avg adaptation loss: {metrics['avg_adaptation_loss']:.4f}")
    
    def _validate(self, val_loader):
        """Validation with the enhanced trainer."""
        self.trainer.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    data = batch.get('input_ids', batch.get('data'))
                    targets = batch.get('labels', batch.get('targets'))
                else:
                    data, targets = batch[0], batch[1]
                
                data = data.to(self.trainer.device)
                targets = targets.to(self.trainer.device)
                
                outputs = self.trainer.model(data)
                loss = self.trainer._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                
                # Calculate accuracy for classification tasks
                if len(outputs.shape) > 1 and outputs.shape[-1] > 1:
                    _, predicted = torch.max(outputs, -1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _log_epoch_results(self, epoch, train_loss, val_loss, val_accuracy, grad_norm):
        """Log comprehensive results for the epoch."""
        logger.info(f"\n📊 Epoch {epoch + 1} Results:")
        logger.info(f"   🔥 Train Loss: {train_loss:.4f}")
        logger.info(f"   ✅ Val Loss: {val_loss:.4f}")
        logger.info(f"   🎯 Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"   📈 Grad Norm: {grad_norm:.3f}")
        
        if self.enable_curriculum:
            params = self.curriculum.get_current_params()
            logger.info(f"   📚 Curriculum:")
            logger.info(f"      Seq Length: {params['sequence_length']}")
            logger.info(f"      Spike Steps: {params['num_spike_steps']}")
            logger.info(f"      Difficulty: {params['difficulty']:.2f}")
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save enhanced checkpoint with meta-learning and curriculum state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        # Add meta-learning state
        if self.enable_meta:
            checkpoint['maml_state'] = {
                'meta_optimizer': self.maml.meta_optimizer.state_dict(),
                'inner_lr': self.maml.inner_lr,
                'num_inner_steps': self.maml.num_inner_steps
            }
        
        # Add curriculum state
        if self.enable_curriculum:
            checkpoint['curriculum_state'] = {
                'current_params': self.curriculum.get_current_params(),
                'performance_history': self.curriculum.performance_history,
                'config': self.curriculum.config.__dict__
            }
        
        filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch+1}.pt'
        filepath = f"./models/{filename}"
        
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Checkpoint saved: {filepath}")


# =============================================================================
# Usage Examples
# =============================================================================

def example_1_basic_curriculum():
    """Example 1: Basic curriculum learning only."""
    from src.core.main import ModelConfig, TaskType, LiquidSpikingNetwork, Trainer
    
    # Create your model and trainer as usual
    config = ModelConfig(
        task_type=TaskType.LLM,
        input_dim=768,
        hidden_dim=512,
        output_dim=50257,
        liquid_units=256,
        spiking_units=128,
        num_layers=6,
        sequence_length=256,
        batch_size=8,
        learning_rate=1e-4,
        # ... other params
    )
    
    model = LiquidSpikingNetwork(config)
    trainer = Trainer(model, config)
    
    # Wrap with enhanced trainer - curriculum only
    enhanced_trainer = EnhancedTrainer(
        base_trainer=trainer,
        enable_meta_learning=False,
        enable_curriculum=True,
        curriculum_strategy="adaptive"  # or 'linear', 'exponential', 'step'
    )
    
    # Train with curriculum
    enhanced_trainer.train(train_loader, val_loader, num_epochs=50)


def example_2_both_features():
    """Example 2: Both meta-learning and curriculum learning."""
    from src.core.main import ModelConfig, TaskType, LiquidSpikingNetwork, Trainer
    
    config = ModelConfig(
        task_type=TaskType.VISION,
        input_dim=3072,
        hidden_dim=256,
        output_dim=10,
        liquid_units=128,
        spiking_units=64,
        num_layers=6,
        # ... other params
    )
    
    model = LiquidSpikingNetwork(config)
    trainer = Trainer(model, config)
    
    # Configure both features
    enhanced_trainer = EnhancedTrainer(
        base_trainer=trainer,
        enable_meta_learning=True,
        enable_curriculum=True,
        curriculum_strategy="self_paced",
        meta_learning_config={
            'meta_lr': 1e-3,
            'inner_lr': 1e-2,
            'num_inner_steps': 5,
            'adapt_liquid_only': True
        },
        curriculum_config={
            'initial_seq_length': 64,
            'final_seq_length': 256,
            'warmup_epochs': 10,
            'performance_threshold': 0.75
        }
    )
    
    # Train with both features
    # Meta-learning will be performed every 5 epochs
    enhanced_trainer.train(
        train_loader,
        val_loader,
        num_epochs=100,
        meta_learning_frequency=5
    )


def example_3_cli_integration():
    """Example 3: How to integrate with CLI (modify cli.py)."""
    
    # In your cli.py, add arguments:
    """
    parser.add_argument('--enable-meta-learning', action='store_true',
                       help='Enable MAML meta-learning')
    parser.add_argument('--enable-curriculum', action='store_true',
                       help='Enable curriculum learning')
    parser.add_argument('--curriculum-strategy', type=str, default='adaptive',
                       choices=['linear', 'exponential', 'step', 'adaptive', 'self_paced'],
                       help='Curriculum learning strategy')
    parser.add_argument('--meta-frequency', type=int, default=5,
                       help='Perform meta-learning every N epochs')
    """
    
    # Then in the train command handler:
    """
    if args.enable_curriculum or args.enable_meta_learning:
        enhanced_trainer = EnhancedTrainer(
            base_trainer=trainer,
            enable_meta_learning=args.enable_meta_learning,
            enable_curriculum=args.enable_curriculum,
            curriculum_strategy=args.curriculum_strategy
        )
        enhanced_trainer.train(train_loader, val_loader, args.epochs, args.meta_frequency)
    else:
        # Regular training
        trainer.train(train_loader, val_loader, args.epochs)
    """


# CLI Usage Examples:
"""
# Curriculum learning only
python scripts/cli.py train --task llm --epochs 50 --enable-curriculum --curriculum-strategy adaptive

# Meta-learning only  
python scripts/cli.py train --task vision --epochs 100 --enable-meta-learning --meta-frequency 5

# Both features combined
python scripts/cli.py train --task robotics --epochs 80 \
  --enable-curriculum --curriculum-strategy self_paced \
  --enable-meta-learning --meta-frequency 10

# Advanced configuration
python scripts/cli.py train --task llm --epochs 100 \
  --liquid-units 384 --spiking-units 192 \
  --enable-curriculum --curriculum-strategy exponential \
  --enable-meta-learning --meta-frequency 5 \
  --batch-size 16 --learning-rate 2e-4
"""


if __name__ == "__main__":
    print("📚 Integration guide loaded successfully!")
    print("See example functions above for usage patterns.")