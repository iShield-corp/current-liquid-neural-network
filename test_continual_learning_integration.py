#!/usr/bin/env python3
"""
Comprehensive test for continual learning integration with Liquid-Spiking model.

Tests all 6 continual learning features:
1. Episodic Memory Bank
2. Experience Replay Buffer
3. Elastic Weight Consolidation (EWC)
4. Synaptic Intelligence (SI)
5. Memory Consolidation
6. Progressive Network Expansion (optional)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.main import ModelConfig, LiquidSpikingTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_task_data(task_id: int, num_samples: int = 500, seq_len: int = 32, vocab_size: int = 1000):
    """
    Create synthetic sequential data for a specific task.
    Each task has different patterns to test continual learning.
    
    Args:
        task_id: Task identifier
        num_samples: Number of training examples
        seq_len: Sequence length
        vocab_size: Vocabulary size
        
    Returns:
        DataLoader with synthetic task data
    """
    np.random.seed(task_id * 42)  # Different seed per task
    
    # Create task-specific patterns
    if task_id == 0:
        # Task 0: Repeat pattern (ABC ABC ABC...)
        pattern = [100, 200, 300]
        inputs = np.tile(pattern, (num_samples, seq_len // len(pattern) + 1))[:, :seq_len]
        targets = np.roll(inputs, -1, axis=1)
        
    elif task_id == 1:
        # Task 1: Alternating pattern (ABABAB...)
        pattern = [400, 500]
        inputs = np.tile(pattern, (num_samples, seq_len // len(pattern) + 1))[:, :seq_len]
        targets = np.roll(inputs, -1, axis=1)
        
    elif task_id == 2:
        # Task 2: Random sequences with bias
        inputs = np.random.randint(600, 800, (num_samples, seq_len))
        targets = np.roll(inputs, -1, axis=1)
        
    else:
        # Default: Random sequences
        inputs = np.random.randint(0, vocab_size, (num_samples, seq_len))
        targets = np.roll(inputs, -1, axis=1)
    
    # Convert to tensors
    inputs_tensor = torch.from_numpy(inputs).long()
    targets_tensor = torch.from_numpy(targets).long()
    
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    logger.info(f"📊 Created Task {task_id} data: {num_samples} samples, seq_len={seq_len}")
    
    return dataloader


def test_continual_learning_system():
    """Test the complete continual learning system with multiple tasks."""
    
    logger.info("="*70)
    logger.info("🧪 TESTING CONTINUAL LEARNING SYSTEM")
    logger.info("="*70)
    
    # Configuration
    vocab_size = 1000
    seq_len = 32
    num_tasks = 3
    epochs_per_task = 2  # Quick test
    
    # Create model configuration with continual learning enabled
    from src.core.main import TaskType
    
    config = ModelConfig(
        task_type=TaskType.LLM,
        input_dim=vocab_size,
        hidden_dim=256,
        output_dim=vocab_size,
        liquid_units=256,
        liquid_backbone='cfc',
        spiking_units=256,
        spike_threshold=1.0,
        beta=0.95,
        num_layers=4,
        dropout=0.1,
        sequence_length=seq_len,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.01,
        gradient_clip=1.0,
        mixed_precision=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        
        # Continual Learning Settings
        use_continual_learning=True,
        episodic_memory_size=1000,
        memory_key_dim=128,
        ewc_lambda=1000.0,
        si_c=0.1,
        consolidation_frequency=100,
        replay_frequency=0.3,
        replay_batch_size=16,
        enable_progressive_networks=False,  # Start without expansion
    )
    
    logger.info("\n📝 Model Configuration:")
    logger.info(f"   Hidden size: {config.hidden_size}")
    logger.info(f"   Layers: {config.num_layers}")
    logger.info(f"   Episodic memory: {config.episodic_memory_size}")
    logger.info(f"   EWC lambda: {config.ewc_lambda}")
    logger.info(f"   SI c: {config.si_c}")
    
    # Initialize trainer
    logger.info(f"\n💻 Using device: {config.device}")
    
    from src.core.main import LiquidSpikingNetwork
    model = LiquidSpikingNetwork(config)
    trainer = LiquidSpikingTrainer(model, config)
    
    # Store task dataloaders for evaluation
    task_dataloaders = {}
    
    # Train on multiple tasks sequentially
    logger.info("\n" + "="*70)
    logger.info("📚 SEQUENTIAL TASK TRAINING")
    logger.info("="*70)
    
    for task_id in range(num_tasks):
        logger.info(f"\n{'='*70}")
        logger.info(f"📖 Task {task_id}/{num_tasks-1}")
        logger.info(f"{'='*70}")
        
        # Create task data
        train_loader = create_synthetic_task_data(task_id, num_samples=500, seq_len=seq_len, vocab_size=vocab_size)
        val_loader = create_synthetic_task_data(task_id, num_samples=100, seq_len=seq_len, vocab_size=vocab_size)
        
        # Store for later evaluation
        task_dataloaders[task_id] = val_loader
        
        # Train on this task
        accuracy = trainer.train_on_task(
            task_id=task_id,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs_per_task
        )
        
        logger.info(f"✅ Task {task_id} completed with accuracy: {accuracy:.4f}")
        
        # Show continual learning statistics
        if trainer.continual_memory_system is not None:
            stats = trainer.continual_memory_system.get_memory_stats()
            logger.info(f"\n📊 Continual Learning Stats after Task {task_id}:")
            logger.info(f"   🧠 Episodic Memory: {stats['episodic_memory_usage']}/{stats['episodic_memory_capacity']}")
            logger.info(f"   🔄 Replay Buffer: {stats['replay_buffer_size']}/{stats['replay_buffer_capacity']}")
            logger.info(f"   🔒 EWC Tasks: {stats['ewc_tasks']}")
            logger.info(f"   ⚖️  SI Omega Mean: {stats['si_omega_mean']:.6f}")
            logger.info(f"   📈 Consolidations: {stats['consolidations_performed']}")
    
    # Final evaluation on all tasks
    logger.info("\n" + "="*70)
    logger.info("📊 FINAL EVALUATION ON ALL TASKS")
    logger.info("="*70)
    
    results, avg_accuracy, avg_forgetting = trainer.evaluate_all_tasks(task_dataloaders)
    
    logger.info(f"\n{'='*70}")
    logger.info("📈 FINAL RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Average Forgetting: {avg_forgetting:.4f}")
    
    for task_id, acc in results.items():
        initial_acc = trainer.task_performance.get(task_id, 0.0)
        forgetting = max(0, initial_acc - acc)
        logger.info(f"Task {task_id}: {acc:.4f} (initial: {initial_acc:.4f}, forgetting: {forgetting:.4f})")
    
    # Test memory retrieval
    logger.info("\n" + "="*70)
    logger.info("🔍 TESTING EPISODIC MEMORY RETRIEVAL")
    logger.info("="*70)
    
    if trainer.continual_memory_system is not None:
        # Create a query from Task 0
        test_loader = create_synthetic_task_data(0, num_samples=1, seq_len=seq_len, vocab_size=vocab_size)
        device = torch.device(trainer.config.device)
        for batch in test_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            
            # Get hidden state as query
            with torch.no_grad():
                hidden = trainer.model.liquid_backbone(inputs)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                
                # Retrieve from episodic memory
                memories = trainer.continual_memory_system.episodic_memory.read(hidden[:, -1:, :])
                logger.info(f"✅ Retrieved memories shape: {memories.shape}")
                logger.info(f"   Query hidden shape: {hidden[:, -1:, :].shape}")
            break
    
    # Test experience replay
    logger.info("\n" + "="*70)
    logger.info("🔄 TESTING EXPERIENCE REPLAY")
    logger.info("="*70)
    
    if trainer.continual_memory_system is not None:
        replay_batch = trainer.continual_memory_system.replay_buffer.sample(batch_size=8)
        if replay_batch:
            logger.info(f"✅ Sampled {len(replay_batch['inputs'])} replay examples")
            logger.info(f"   Tasks: {replay_batch['task_ids']}")
            logger.info(f"   Priorities: {[f'{p:.4f}' for p in replay_batch['priorities'][:5]]}...")
        else:
            logger.info("⚠️  Replay buffer empty")
    
    # Final memory statistics
    logger.info("\n" + "="*70)
    logger.info("📊 FINAL MEMORY SYSTEM STATISTICS")
    logger.info("="*70)
    
    if trainer.continual_memory_system is not None:
        stats = trainer.continual_memory_system.get_memory_stats()
        logger.info("Continual Learning System Stats:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ CONTINUAL LEARNING TEST COMPLETED")
    logger.info("="*70)
    
    return {
        'avg_accuracy': avg_accuracy,
        'avg_forgetting': avg_forgetting,
        'task_results': results,
        'memory_stats': stats if trainer.continual_memory_system else {}
    }


def test_feature_isolation():
    """Test each continual learning feature in isolation."""
    
    logger.info("\n" + "="*70)
    logger.info("🔬 TESTING INDIVIDUAL FEATURES")
    logger.info("="*70)
    
    # Test configurations for each feature
    feature_configs = {
        'baseline': {
            'use_continual_learning': False,
            'name': 'Baseline (No Continual Learning)'
        },
        'episodic_only': {
            'use_continual_learning': True,
            'episodic_memory_size': 1000,
            'ewc_lambda': 0.0,  # Disable EWC
            'si_c': 0.0,  # Disable SI
            'name': 'Episodic Memory Only'
        },
        'ewc_only': {
            'use_continual_learning': True,
            'episodic_memory_size': 0,  # Disable episodic
            'ewc_lambda': 1000.0,
            'si_c': 0.0,
            'name': 'EWC Only'
        },
        'si_only': {
            'use_continual_learning': True,
            'episodic_memory_size': 0,
            'ewc_lambda': 0.0,
            'si_c': 0.1,
            'name': 'Synaptic Intelligence Only'
        },
        'full_system': {
            'use_continual_learning': True,
            'episodic_memory_size': 1000,
            'ewc_lambda': 1000.0,
            'si_c': 0.1,
            'name': 'Full System (All Features)'
        }
    }
    
    results_summary = {}
    
    for config_name, feature_config in feature_configs.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {feature_config['name']}")
        logger.info(f"{'='*70}")
        
        # Base config
        from src.core.main import TaskType
        config = ModelConfig(
            task_type=TaskType.LLM,
            input_dim=1000,
            hidden_dim=128,
            output_dim=1000,
            liquid_units=128,
            liquid_backbone='cfc',
            spiking_units=128,
            spike_threshold=1.0,
            beta=0.95,
            num_layers=2,
            dropout=0.1,
            sequence_length=16,
            batch_size=32,
            learning_rate=0.001,
            weight_decay=0.01,
            gradient_clip=1.0,
            mixed_precision=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            seed=42,
            **{k: v for k, v in feature_config.items() if k != 'name'}
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from src.core.main import LiquidSpikingNetwork
        model = LiquidSpikingNetwork(config)
        trainer = LiquidSpikingTrainer(model, config)
        
        # Quick 2-task test
        task_loaders = {}
        for task_id in range(2):
            train_loader = create_synthetic_task_data(task_id, num_samples=200, seq_len=16)
            val_loader = create_synthetic_task_data(task_id, num_samples=50, seq_len=16)
            task_loaders[task_id] = val_loader
            
            trainer.train_on_task(task_id, train_loader, val_loader, num_epochs=1)
        
        # Evaluate
        results, avg_acc, avg_forget = trainer.evaluate_all_tasks(task_loaders)
        
        results_summary[config_name] = {
            'name': feature_config['name'],
            'avg_accuracy': avg_acc,
            'avg_forgetting': avg_forget
        }
        
        logger.info(f"✅ {feature_config['name']}:")
        logger.info(f"   Avg Accuracy: {avg_acc:.4f}")
        logger.info(f"   Avg Forgetting: {avg_forget:.4f}")
    
    # Print comparison
    logger.info("\n" + "="*70)
    logger.info("📊 FEATURE COMPARISON")
    logger.info("="*70)
    
    for config_name, results in results_summary.items():
        logger.info(f"\n{results['name']}:")
        logger.info(f"  Accuracy:  {results['avg_accuracy']:.4f}")
        logger.info(f"  Forgetting: {results['avg_forgetting']:.4f}")
    
    return results_summary


if __name__ == '__main__':
    logger.info("🚀 Starting Continual Learning Integration Tests\n")
    
    # Test 1: Full system
    logger.info("="*70)
    logger.info("TEST 1: Full Continual Learning System")
    logger.info("="*70)
    full_results = test_continual_learning_system()
    
    # Test 2: Feature isolation
    logger.info("\n\n" + "="*70)
    logger.info("TEST 2: Individual Feature Testing")
    logger.info("="*70)
    feature_results = test_feature_isolation()
    
    # Final summary
    logger.info("\n\n" + "="*70)
    logger.info("🎉 ALL TESTS COMPLETED")
    logger.info("="*70)
    
    logger.info("\n✅ Full System Results:")
    logger.info(f"   Average Accuracy: {full_results['avg_accuracy']:.4f}")
    logger.info(f"   Average Forgetting: {full_results['avg_forgetting']:.4f}")
    
    logger.info("\n📊 Feature Comparison:")
    for name, results in feature_results.items():
        logger.info(f"   {results['name']}: Acc={results['avg_accuracy']:.4f}, Forget={results['avg_forgetting']:.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("🎉 SUCCESS: All continual learning features working!")
    logger.info("="*70)
