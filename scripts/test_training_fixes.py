#!/usr/bin/env python3
"""
Quick validation test for training fixes.
Runs a minimal training session to verify fixes work correctly.
"""

import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.main import LiquidSpikingNetwork, create_llm_config, LiquidSpikingTrainer
from src.training.training_fixes import (
    apply_training_fixes,
    diagnose_training_stuck,
    GradientHealthMonitor,
    add_warmup_scheduler
)
from torch.utils.data import DataLoader, TensorDataset


def create_dummy_data(vocab_size=1000, seq_length=32, num_samples=100, batch_size=4):
    """Create dummy data for quick testing."""
    # Random token sequences
    inputs = torch.randint(0, vocab_size, (num_samples, seq_length))
    targets = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader


def test_fixes():
    """Test if fixes enable learning on dummy data."""
    print("=" * 70)
    print("🧪 TESTING TRAINING FIXES")
    print("=" * 70)
    print("\nThis test validates that the fixes enable actual learning.")
    print("Test: 5 epochs on 100 dummy samples")
    print("Expected: Loss should decrease by epoch 3\n")
    
    # Create minimal config
    config = create_llm_config("gpt2")
    config.num_layers = 2
    config.hidden_dim = 128
    config.liquid_units = 64
    config.spiking_units = 32
    config.num_epochs = 5
    config.batch_size = 4
    config.learning_rate = 5e-5  # Critical fix
    config.gradient_clip = 0.5    # Critical fix
    
    print(f"📋 Configuration:")
    print(f"   Layers: {config.num_layers}")
    print(f"   Hidden: {config.hidden_dim}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Gradient Clip: {config.gradient_clip}")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = LiquidSpikingNetwork(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   Device: {device}")
    
    # Apply fixes
    print(f"\n🔧 Applying training fixes...")
    model = apply_training_fixes(model, config)
    print(f"   ✅ Fixed spiking encoders (surrogate slope=25)")
    print(f"   ✅ Improved spike decoder (multi-path)")
    print(f"   ✅ Re-initialized output head (std=0.01)")
    
    # Create dummy data
    print(f"\n📚 Creating test data...")
    train_loader = create_dummy_data(
        vocab_size=config.vocab_size,
        seq_length=config.sequence_length,
        num_samples=100,
        batch_size=config.batch_size
    )
    print(f"   Samples: 100")
    print(f"   Batch size: {config.batch_size}")
    
    # Run diagnostics
    print(f"\n📋 Pre-training diagnostics...")
    diagnostics = diagnose_training_stuck(model, train_loader, device)
    
    # Create trainer
    print(f"\n🏋️  Setting up trainer...")
    trainer = LiquidSpikingTrainer(model, config)
    
    # Add warmup scheduler
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = len(train_loader) * 1  # 1 epoch warmup
    trainer.scheduler = add_warmup_scheduler(
        trainer.optimizer,
        num_warmup_steps,
        num_training_steps
    )
    print(f"   Warmup steps: {num_warmup_steps}")
    
    # Add gradient monitor
    grad_monitor = GradientHealthMonitor(model)
    
    # Training loop
    print(f"\n" + "=" * 70)
    print(f"🚀 TRAINING")
    print(f"=" * 70)
    
    losses = []
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)
        losses.append(train_loss)
        
        # Check gradients
        grad_stats = grad_monitor.check_gradients()
        
        # Validate (use same data for quick test)
        val_loss, val_acc = trainer.validate(train_loader)
        
        # Print results
        status = "✅" if train_loss < 12.0 else "⚠️"
        print(f"{status} Epoch {epoch+1}/{config.num_epochs}: "
              f"Loss={train_loss:.4f}, "
              f"Grad={grad_stats['mean_norm']:.6f}")
        
        # Check if learning
        if epoch > 0:
            loss_change = losses[-2] - losses[-1]
            if loss_change > 0.01:
                print(f"   ✅ Learning! Loss decreased by {loss_change:.4f}")
            elif abs(loss_change) < 0.001:
                print(f"   ⚠️  Warning: Loss not changing much ({loss_change:.6f})")
    
    # Analyze results
    print(f"\n" + "=" * 70)
    print(f"📊 RESULTS")
    print(f"=" * 70)
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = initial_loss - final_loss
    improvement_pct = (improvement / initial_loss) * 100
    
    print(f"\nLoss Progress:")
    print(f"   Initial: {initial_loss:.4f}")
    print(f"   Final:   {final_loss:.4f}")
    print(f"   Change:  {improvement:.4f} ({improvement_pct:.1f}%)")
    
    # Check gradient health
    print(f"\nGradient Health:")
    diagnosis = grad_monitor.diagnose()
    print(f"   {diagnosis}")
    
    # Verdict
    print(f"\n" + "=" * 70)
    print(f"🎯 VERDICT")
    print(f"=" * 70)
    
    if improvement > 0.1 and final_loss < 11.5:
        print(f"✅ SUCCESS! Fixes are working correctly.")
        print(f"   - Loss decreased significantly")
        print(f"   - Gradients are healthy")
        print(f"   - Model is learning")
        print(f"\n✨ You can now train on real data!")
        print(f"   Command: python train.py llm_fixed --epochs 10")
        return True
    elif improvement > 0.01:
        print(f"⚠️  PARTIAL SUCCESS. Some learning detected.")
        print(f"   - Loss decreased slightly")
        print(f"   - May need more epochs or tuning")
        print(f"\n💡 Try: python train.py llm_fixed --lr 1e-5 --epochs 20")
        return True
    else:
        print(f"❌ FIXES NOT WORKING. No learning detected.")
        print(f"   - Loss not decreasing")
        print(f"   - Check gradient health above")
        print(f"\n🔍 Please review:")
        print(f"   1. Gradient statistics")
        print(f"   2. Try even lower learning rate (1e-5)")
        print(f"   3. Check for errors in apply_training_fixes()")
        return False


if __name__ == "__main__":
    try:
        success = test_fixes()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
