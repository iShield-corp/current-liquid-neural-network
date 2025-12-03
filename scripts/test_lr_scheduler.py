#!/usr/bin/env python3
"""
Test that learning rate scheduler doesn't drop to zero.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.main import LiquidSpikingTrainer, create_llm_config, LiquidSpikingNetwork

def test_lr_scheduler():
    """Test that LR doesn't go to zero."""
    print("=" * 70)
    print("🧪 TESTING LEARNING RATE SCHEDULER")
    print("=" * 70)
    
    # Create minimal config
    config = create_llm_config("gpt2")
    config.num_epochs = 10
    
    # Create model and trainer
    model = LiquidSpikingNetwork(config)
    trainer = LiquidSpikingTrainer(model, config)
    
    print(f"\nInitial LR: {trainer.optimizer.param_groups[0]['lr']:.6e}")
    print(f"Expected: {config.learning_rate:.6e}")
    
    # Simulate training epochs
    print("\nSimulating epoch progression:")
    print("-" * 70)
    
    all_good = True
    
    for epoch in range(10):
        lr = trainer.optimizer.param_groups[0]['lr']
        
        # Check LR is not zero
        if lr == 0.0:
            print(f"Epoch {epoch}: LR = {lr:.6e}  ❌ ZERO!")
            all_good = False
        elif lr < 1e-8:
            print(f"Epoch {epoch}: LR = {lr:.6e}  ⚠️  Very small")
            all_good = False
        else:
            print(f"Epoch {epoch}: LR = {lr:.6e}  ✅")
        
        # Step scheduler (simulates end of epoch)
        trainer.scheduler.step()
    
    print("-" * 70)
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_good:
        print("✅ SCHEDULER WORKING CORRECTLY")
        print("   LR never dropped to zero")
        return 0
    else:
        print("❌ SCHEDULER HAS ISSUES")
        print("   LR dropped to zero or too small")
        return 1


if __name__ == "__main__":
    exit_code = test_lr_scheduler()
    sys.exit(exit_code)
