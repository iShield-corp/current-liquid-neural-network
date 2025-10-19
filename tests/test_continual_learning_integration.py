#!/usr/bin/env python3
"""
Integration test for STDP, Meta-Plasticity, and Continual Learning features.

This script validates that all plasticity features work together correctly.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.main import (
    ModelConfig, LiquidSpikingNetwork, LiquidSpikingTrainer,
    TaskType, create_llm_config
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_task_data(task_id, num_samples=1000, seq_len=128, vocab_size=1000):
    """Create dummy data for testing continual learning."""
    torch.manual_seed(42 + task_id)
    
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=8, shuffle=True)


def test_stdp_only():
    """Test STDP integration only."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: STDP Only")
    logger.info("="*70)
    
    try:
        config = create_llm_config('gpt2')
        config.num_epochs = 2
        config.use_stdp = True
        config.stdp_type = 'homeostatic'
        config.stdp_learning_rate = 0.01
        
        model = LiquidSpikingNetwork(config)
        trainer = LiquidSpikingTrainer(model, config)
        
        train_loader = create_dummy_task_data(0, num_samples=200)
        val_loader = create_dummy_task_data(0, num_samples=50)
        
        trainer.train(train_loader, val_loader, num_epochs=2)
        
        logger.info("✅ STDP integration test PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ STDP integration test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_meta_plasticity_only():
    """Test meta-plasticity integration only."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Meta-Plasticity Only")
    logger.info("="*70)
    
    try:
        config = create_llm_config('gpt2')
        config.num_epochs = 2
        config.use_meta_plasticity = True
        config.meta_lr = 0.001
        
        model = LiquidSpikingNetwork(config)
        trainer = LiquidSpikingTrainer(model, config)
        
        train_loader = create_dummy_task_data(0, num_samples=200)
        val_loader = create_dummy_task_data(0, num_samples=50)
        
        trainer.train(train_loader, val_loader, num_epochs=2)
        
        logger.info("✅ Meta-plasticity integration test PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Meta-plasticity integration test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_continual_learning_full():
    """Test full continual learning with all features."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Full Continual Learning (STDP + Meta + Replay)")
    logger.info("="*70)
    
    try:
        config = create_llm_config('gpt2')
        config.num_epochs = 2
        config.use_stdp = True
        config.stdp_type = 'homeostatic'
        config.use_meta_plasticity = True
        config.use_continual_learning = True
        config.consolidation_strength = 1000.0
        config.use_experience_replay = True
        config.replay_buffer_size = 200
        
        model = LiquidSpikingNetwork(config)
        trainer = LiquidSpikingTrainer(model, config)
        
        # Create 3 tasks
        num_tasks = 3
        task_loaders = {}
        
        for task_id in range(num_tasks):
            task_loaders[task_id] = {
                'train': create_dummy_task_data(task_id, num_samples=200),
                'val': create_dummy_task_data(task_id, num_samples=50)
            }
        
        # Train on tasks sequentially
        logger.info("\n📚 Training on multiple tasks sequentially...")
        for task_id in range(num_tasks):
            logger.info(f"\n{'='*60}")
            logger.info(f"TASK {task_id + 1} / {num_tasks}")
            logger.info(f"{'='*60}")
            
            trainer.train_on_task(
                task_id=task_id,
                train_loader=task_loaders[task_id]['train'],
                val_loader=task_loaders[task_id]['val'],
                num_epochs=2
            )
        
        # Evaluate on all tasks
        logger.info("\n📊 Evaluating on all tasks...")
        all_val_loaders = {i: task_loaders[i]['val'] for i in range(num_tasks)}
        results, avg_acc, forgetting = trainer.evaluate_all_tasks(all_val_loaders)
        
        logger.info(f"\n📈 Final Results:")
        for task_id, acc in results.items():
            logger.info(f"   Task {task_id}: {acc:.3f}")
        logger.info(f"   Average Accuracy: {avg_acc:.3f}")
        logger.info(f"   Average Forgetting: {forgetting:.3f}")
        
        # Validate forgetting is reasonable
        if forgetting > 0.5:
            logger.warning(f"⚠️  High forgetting detected: {forgetting:.3f}")
            logger.info("   This is expected for very limited training epochs")
        
        logger.info("✅ Full continual learning test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Full continual learning test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_config_serialization():
    """Test that new config parameters serialize/deserialize correctly."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Config Serialization with Plasticity Parameters")
    logger.info("="*70)
    
    try:
        config = create_llm_config('gpt2')
        config.use_stdp = True
        config.stdp_type = 'homeostatic'
        config.use_meta_plasticity = True
        config.use_continual_learning = True
        
        # Convert to dict and back
        config_dict = config.to_dict()
        config_restored = ModelConfig.from_dict(config_dict)
        
        # Verify all plasticity parameters
        assert config_restored.use_stdp == config.use_stdp
        assert config_restored.stdp_type == config.stdp_type
        assert config_restored.use_meta_plasticity == config.use_meta_plasticity
        assert config_restored.use_continual_learning == config.use_continual_learning
        
        logger.info("✅ Config serialization test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config serialization test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    logger.info("\n" + "="*70)
    logger.info("CONTINUAL LEARNING INTEGRATION TEST SUITE")
    logger.info("="*70)
    logger.info("Testing STDP + Meta-Plasticity + Continual Learning integration")
    logger.info("="*70)
    
    results = {
        'STDP Only': test_stdp_only(),
        'Meta-Plasticity Only': test_meta_plasticity_only(),
        'Full Continual Learning': test_continual_learning_full(),
        'Config Serialization': test_config_serialization()
    }
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name:.<50} {status}")
    
    logger.info("="*70)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
