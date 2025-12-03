#!/usr/bin/env python3
"""
Train model with critical fixes for zero-learning issue.

This script applies all gradient flow and initialization fixes to enable proper learning.

Usage:
    python scripts/train_with_fixes.py [--epochs 10] [--batch-size 4] [--lr 5e-5]
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.main import (
    LiquidSpikingNetwork, LiquidSpikingTrainer,
    DatasetFactory, create_llm_config
)
from src.training.training_fixes import (
    apply_training_fixes, 
    add_warmup_scheduler,
    diagnose_training_stuck,
    GradientHealthMonitor
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train with fixes')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dataset', type=str, default='wikitext2', 
                       choices=['wikitext2', 'wikitext103'], help='Dataset to use')
    parser.add_argument('--subset-size', type=int, default=1000, 
                       help='Dataset subset size for testing (0 for full dataset)')
    parser.add_argument('--diagnose', action='store_true', 
                       help='Run diagnostics before training')
    parser.add_argument('--output-dir', type=str, default='./models', 
                       help='Output directory for saved models')
    return parser.parse_args()


def train_with_fixes():
    """Train model with all critical fixes applied."""
    args = parse_args()
    
    print("=" * 70)
    print("🔧 Training Liquid-Spiking Network with Critical Fixes")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Layers: {args.layers}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Subset Size: {args.subset_size if args.subset_size > 0 else 'Full'}")
    print()
    
    # Create config with better hyperparameters
    config = create_llm_config("gpt2")
    
    # Apply optimized hyperparameters
    config.learning_rate = args.lr
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.num_layers = args.layers
    config.hidden_dim = args.hidden_dim
    config.gradient_clip = 0.5  # Tighter clipping for stability
    
    # Create model
    print("🏗️  Creating model...")
    model = LiquidSpikingNetwork(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Apply fixes
    print("\n🔧 Applying critical training fixes...")
    model = apply_training_fixes(model, config)
    
    # Create dataset
    print(f"\n📚 Loading {args.dataset} dataset...")
    dataset, tokenizer = DatasetFactory.create_llm_dataset(
        vocab_size=config.vocab_size,
        seq_length=config.sequence_length,
        tokenizer_type='gpt2',
        dataset_type=args.dataset,
        cache_dir='./data'
    )
    
    # Use subset if specified
    if args.subset_size > 0:
        subset_size = min(args.subset_size, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(subset_size))
        print(f"  Using subset: {subset_size} samples")
    else:
        print(f"  Using full dataset: {len(dataset)} samples")
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"  Train samples: {train_size}")
    print(f"  Val samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Run diagnostics if requested
    if args.diagnose:
        print("\n📋 Running pre-training diagnostics...")
        diagnose_training_stuck(model, train_loader, device)
    
    # Create trainer
    print("\n🏋️  Initializing trainer...")
    trainer = LiquidSpikingTrainer(model, config)
    
    # Replace scheduler with warmup
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = len(train_loader) * 2  # 2 epochs warmup
    trainer.scheduler = add_warmup_scheduler(
        trainer.optimizer, 
        num_warmup_steps, 
        num_training_steps
    )
    print(f"  Optimizer: {type(trainer.optimizer).__name__}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Total training steps: {num_training_steps}")
    
    # Add gradient monitoring
    grad_monitor = GradientHealthMonitor(model)
    
    # Training loop
    print("\n" + "=" * 70)
    print("🚀 Starting Training")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 3
    
    for epoch in range(config.num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{config.num_epochs}")
        print("-" * 70)
        
        # Train epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # Check gradients
        grad_stats = grad_monitor.check_gradients()
        grad_health = "✓" if grad_stats['mean_norm'] > 1e-6 else "⚠"
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  {grad_health} Gradient Health: mean={grad_stats['mean_norm']:.6f}, "
              f"max={grad_stats['max_norm']:.6f}, "
              f"zeros={grad_stats['zero_grads']}/{grad_stats['total_params']}")
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader)
        
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # Learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check if learning is happening
        if epoch > 2:
            prev_train_loss = trainer.train_losses[-2]
            loss_change = abs(train_loss - prev_train_loss)
            
            if loss_change < 1e-6:
                print(f"  ⚠️  WARNING: Loss not changing (Δ={loss_change:.8f})")
                print("  Running diagnostics...")
                diagnose_training_stuck(model, train_loader, device)
                print(f"  {grad_monitor.diagnose()}")
            else:
                improvement = "↓" if train_loss < prev_train_loss else "↑"
                print(f"  {improvement} Loss change: {loss_change:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_path = os.path.join(args.output_dir, 'llm_best_fixed_model.pt')
            trainer.save_checkpoint(best_model_path)
            print(f"  ⭐ New best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience_limit}")
            
            if patience_counter >= patience_limit:
                print("\n⏸️  Early stopping triggered!")
                break
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'llm_final_fixed_model.pt')
    trainer.save_checkpoint(final_model_path)
    
    # Training summary
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Final Train Loss: {train_loss:.4f}")
    print(f"  Final Val Loss: {val_loss:.4f}")
    print(f"  Final Val Acc: {val_acc:.4f}")
    print(f"\nModels saved to:")
    print(f"  Best: {best_model_path}")
    print(f"  Final: {final_model_path}")
    
    # Final gradient diagnosis
    print("\n📊 Final Gradient Health Check:")
    print(grad_monitor.diagnose())
    
    return trainer


if __name__ == "__main__":
    train_with_fixes()
