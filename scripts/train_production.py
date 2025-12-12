#!/usr/bin/env python3
"""
Production-ready training script for LLM training.
Works with both CLI and GUI, includes checkpointing, validation, early stopping.
NOW WITH CONTINUAL LEARNING SUPPORT (All 6 features)
"""
import argparse
import torch
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from src.core.main import (
    create_llm_config,
    LiquidSpikingNetwork,
    LiquidSpikingTrainer,
    WikiTextDataset,
    TextDataset,
    TaskType,
    ModelConfig
)

# Import memory manager utilities
from src.utils.memory_manager import print_gpu_memory_info

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Production Training for Liquid-Spiking Neural Network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['tiny', 'small', 'medium', 'large'],
                        help='Model size preset')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of layers (overrides preset)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help='Hidden dimension (overrides preset)')
    parser.add_argument('--liquid-units', type=int, default=None,
                        help='Liquid network units (overrides preset)')
    parser.add_argument('--spiking-units', type=int, default=None,
                        help='Spiking network units (overrides preset)')
    parser.add_argument('--num-attention-heads', type=int, default=None,
                        help='Number of attention heads (overrides preset)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (overrides preset)')
    parser.add_argument('--gradient-clip', type=float, default=None,
                        help='Gradient clipping value (overrides preset)')
    parser.add_argument('--seq-length', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'wikitext103', 'custom', 'combined'],
                        help='Dataset to use')
    parser.add_argument('--combined-datasets', type=str, default=None,
                        help='Comma-separated list of datasets for combined mode (e.g., programming,wikitext103,bookcorpus,openwebtext)')
    parser.add_argument('--custom-dataset-path', type=str, default=None,
                        help='Path to custom dataset (if dataset=custom)')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='Tokenizer to use')
    parser.add_argument('--vocab-size', type=int, default=50257,
                        help='Vocabulary size')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./models',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from checkpoint path')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=0.001,
                        help='Minimum improvement for early stopping')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # ============================================================
    # CONTINUAL LEARNING PARAMETERS (NEW - All 6 Features)
    # ============================================================
    parser.add_argument('--use-continual-learning', action='store_true',
                        help='Enable continual learning system')
    
    # Feature 1: Episodic Memory Bank
    parser.add_argument('--episodic-memory-size', type=int, default=10000,
                        help='Episodic memory bank size')
    parser.add_argument('--episodic-key-dim', type=int, default=128,
                        help='Episodic memory key dimension')
    parser.add_argument('--episodic-value-dim', type=int, default=512,
                        help='Episodic memory value dimension')
    
    # Feature 2: Experience Replay Buffer
    parser.add_argument('--replay-buffer-size', type=int, default=5000,
                        help='Experience replay buffer size')
    parser.add_argument('--replay-frequency', type=float, default=0.25,
                        help='Replay frequency (0.0-1.0)')
    parser.add_argument('--replay-priority-alpha', type=float, default=0.6,
                        help='Priority exponent for replay sampling')
    
    # Feature 3: Elastic Weight Consolidation (EWC)
    parser.add_argument('--ewc-lambda', type=float, default=5000.0,
                        help='EWC regularization strength')
    parser.add_argument('--ewc-fisher-samples', type=int, default=200,
                        help='Number of samples for Fisher matrix')
    parser.add_argument('--use-online-ewc', action='store_true',
                        help='Use online EWC variant')
    parser.add_argument('--ewc-gamma', type=float, default=0.9,
                        help='Online EWC decay factor')
    
    # Feature 4: Synaptic Intelligence (SI)
    parser.add_argument('--si-c', type=float, default=0.1,
                        help='Synaptic Intelligence coefficient')
    parser.add_argument('--si-epsilon', type=float, default=0.001,
                        help='SI damping parameter')
    
    # Feature 5: Memory Consolidation
    parser.add_argument('--consolidation-frequency', type=int, default=1000,
                        help='Steps between consolidation')
    parser.add_argument('--consolidation-strength', type=float, default=0.5,
                        help='Consolidation transfer strength')
    parser.add_argument('--consolidation-threshold', type=float, default=0.7,
                        help='Similarity threshold for merging')
    
    # Feature 6: Progressive Neural Networks
    parser.add_argument('--enable-progressive-networks', action='store_true',
                        help='Enable progressive network expansion')
    parser.add_argument('--expansion-threshold', type=float, default=0.95,
                        help='Performance threshold for expansion')
    parser.add_argument('--lateral-connections', action='store_true',
                        help='Use lateral connections in progressive nets')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='liquid-spiking-llm',
                        help='W&B project name')
    
    # ============================================================
    # MAMBA INTEGRATION PARAMETERS (NEW)
    # ============================================================
    mamba_group = parser.add_argument_group('Mamba SSM Integration', 
                                            'Long-range sequence modeling with Mamba')
    mamba_group.add_argument('--use-mamba', action='store_true',
                            help='Enable Mamba SSM integration')
    mamba_group.add_argument('--integration-mode', 
                            choices=['sequential', 'parallel', 'bidirectional'],
                            default='bidirectional',
                            help='Mamba-Liquid-Spiking integration mode (default: bidirectional)')
    mamba_group.add_argument('--mamba-d-state', type=int, default=16,
                            help='Mamba state space dimension (default: 16)')
    mamba_group.add_argument('--mamba-d-conv', type=int, default=4,
                            help='Mamba convolution kernel size (default: 4)')
    mamba_group.add_argument('--mamba-expand', type=int, default=2,
                            help='Mamba expansion factor (default: 2)')
    mamba_group.add_argument('--use-cross-attention', action='store_true',
                            help='Enable cross-attention between Liquid and Mamba (bidirectional mode only)')
    mamba_group.add_argument('--use-adaptive-gating', action='store_true',
                            help='Enable adaptive gating for integration')
    
    # ============================================================
    # STDP AND META-PLASTICITY PARAMETERS (NEW)
    # ============================================================
    plasticity_group = parser.add_argument_group('Plasticity Mechanisms',
                                                  'STDP and meta-plasticity features')
    plasticity_group.add_argument('--use-stdp', action='store_true',
                                  help='Enable Spike-Timing-Dependent Plasticity')
    plasticity_group.add_argument('--use-meta-plasticity', action='store_true',
                                  help='Enable meta-plasticity mechanisms')
    
    return parser.parse_args()


def get_model_preset(preset_name: str) -> Dict[str, Any]:
    """Get model size preset configurations."""
    presets = {
        'tiny': {
            'num_layers': 2,
            'hidden_dim': 128,
            'liquid_units': 128,
            'spiking_units': 128,
            'num_attention_heads': 4,
            'learning_rate': 0.001,
            'gradient_clip': 1.0
        },
        'small': {
            'num_layers': 4,
            'hidden_dim': 256,
            'liquid_units': 256,
            'spiking_units': 256,
            'num_attention_heads': 8,
            'learning_rate': 0.0005,
            'gradient_clip': 1.0
        },
        'medium': {
            'num_layers': 6,
            'hidden_dim': 512,
            'liquid_units': 512,
            'spiking_units': 512,
            'num_attention_heads': 8,
            'learning_rate': 0.0003,
            'gradient_clip': 1.0
        },
        'large': {
            'num_layers': 12,
            'hidden_dim': 768,
            'liquid_units': 768,
            'spiking_units': 768,
            'num_attention_heads': 12,
            'learning_rate': 0.0001,
            'gradient_clip': 1.0
        }
    }
    return presets.get(preset_name, presets['small'])


def setup_model_config(args) -> ModelConfig:
    """Create model configuration from arguments."""
    # Get preset
    preset = get_model_preset(args.model_size)
    
    # Override with CLI args if provided
    num_layers = args.num_layers if args.num_layers else preset['num_layers']
    hidden_dim = args.hidden_dim if args.hidden_dim else preset['hidden_dim']
    liquid_units = args.liquid_units if args.liquid_units else preset['liquid_units']
    spiking_units = args.spiking_units if args.spiking_units else preset['spiking_units']
    learning_rate = args.learning_rate if args.learning_rate else preset['learning_rate']
    gradient_clip = args.gradient_clip if args.gradient_clip else preset['gradient_clip']
    
    # Create base config using the factory function
    config = create_llm_config(tokenizer_type=args.tokenizer)
    
    # Override with preset and CLI arguments
    config.vocab_size = args.vocab_size
    config.output_dim = args.vocab_size
    config.hidden_dim = hidden_dim
    config.num_layers = num_layers
    config.sequence_length = args.seq_length
    config.batch_size = args.batch_size
    config.learning_rate = learning_rate
    config.num_epochs = args.epochs
    config.gradient_clip = gradient_clip
    config.device = args.device
    config.mixed_precision = args.mixed_precision
    config.liquid_units = liquid_units
    config.spiking_units = spiking_units
    config.weight_decay = 0.01
    config.accumulation_steps = args.accumulation_steps
    
    # ============================================================
    # ADD CONTINUAL LEARNING PARAMETERS (NEW)
    # ============================================================
    if args.use_continual_learning:
        logger.info("🧠 Enabling Continual Learning System with all 6 features...")
        
        config.use_continual_learning = True
        
        # Feature 1: Episodic Memory
        config.episodic_memory_size = args.episodic_memory_size
        config.episodic_key_dim = args.episodic_key_dim
        config.episodic_value_dim = args.episodic_value_dim
        
        # Feature 2: Experience Replay
        config.replay_buffer_size = args.replay_buffer_size
        config.replay_frequency = args.replay_frequency
        config.replay_priority_alpha = args.replay_priority_alpha
        
        # Feature 3: EWC
        config.ewc_lambda = args.ewc_lambda
        config.ewc_fisher_samples = args.ewc_fisher_samples
        config.use_online_ewc = args.use_online_ewc
        config.ewc_gamma = args.ewc_gamma
        
        # Feature 4: Synaptic Intelligence
        config.si_c = args.si_c
        config.si_epsilon = args.si_epsilon
        
        # Feature 5: Memory Consolidation
        config.consolidation_frequency = args.consolidation_frequency
        config.consolidation_strength = args.consolidation_strength
        config.consolidation_threshold = args.consolidation_threshold
        
        # Feature 6: Progressive Networks
        config.enable_progressive_networks = args.enable_progressive_networks
        config.expansion_threshold = args.expansion_threshold
        config.lateral_connections = args.lateral_connections
        
        logger.info(f"  ✅ Episodic Memory: {args.episodic_memory_size} slots")
        logger.info(f"  ✅ Replay Buffer: {args.replay_buffer_size} examples")
        logger.info(f"  ✅ EWC Lambda: {args.ewc_lambda}")
        logger.info(f"  ✅ SI Coefficient: {args.si_c}")
        logger.info(f"  ✅ Consolidation: Every {args.consolidation_frequency} steps")
        logger.info(f"  ✅ Progressive Networks: {'Enabled' if args.enable_progressive_networks else 'Disabled'}")
    
    # ============================================================
    # ADD MAMBA INTEGRATION PARAMETERS (NEW)
    # ============================================================
    if args.use_mamba:
        logger.info("🔗 Enabling Mamba SSM Integration...")
        
        config.use_mamba = True
        config.integration_mode = args.integration_mode
        config.mamba_d_state = args.mamba_d_state
        config.mamba_d_conv = args.mamba_d_conv
        config.mamba_expand = args.mamba_expand
        config.use_cross_attention = args.use_cross_attention
        config.use_adaptive_gating = args.use_adaptive_gating
        
        logger.info(f"  ✅ Integration Mode: {args.integration_mode}")
        logger.info(f"  ✅ Mamba State Dim: {args.mamba_d_state}")
        logger.info(f"  ✅ Mamba Conv Kernel: {args.mamba_d_conv}")
        logger.info(f"  ✅ Mamba Expansion Factor: {args.mamba_expand}")
        logger.info(f"  ✅ Cross-Attention: {'Enabled' if args.use_cross_attention else 'Disabled'}")
        logger.info(f"  ✅ Adaptive Gating: {'Enabled' if args.use_adaptive_gating else 'Disabled'}")
    
    # ============================================================
    # ADD STDP AND META-PLASTICITY PARAMETERS (NEW)
    # ============================================================
    if args.use_stdp or args.use_meta_plasticity:
        logger.info("⚡ Enabling Plasticity Mechanisms...")
        
        if args.use_stdp:
            config.use_stdp = True
            logger.info(f"  ✅ STDP: Enabled")
        
        if args.use_meta_plasticity:
            config.use_meta_plasticity = True
            logger.info(f"  ✅ Meta-plasticity: Enabled")
    
    return config


def load_dataset(args, tokenizer, split='train'):
    """Load dataset based on arguments."""
    if args.dataset == 'custom':
        if not args.custom_dataset_path:
            raise ValueError("--custom-dataset-path required for custom dataset")
        return TextDataset(
            args.custom_dataset_path,
            tokenizer,
            max_length=args.seq_length
        )
    elif args.dataset == 'combined':
        # Combined datasets mode
        if not args.combined_datasets:
            raise ValueError("--combined-datasets required when using --dataset combined")
        
        logger.info(f"Loading combined datasets: {args.combined_datasets}")
        datasets = []
        dataset_names = [name.strip() for name in args.combined_datasets.split(',')]
        
        for name in dataset_names:
            if name == 'wikitext2':
                ds = WikiTextDataset.create_dataset(
                    version='wikitext-2-v1',
                    split=split,
                    tokenizer=tokenizer,
                    max_length=args.seq_length
                )
                datasets.append(ds)
                logger.info(f"  ✅ Added wikitext2 ({len(ds)} samples)")
            elif name == 'wikitext103':
                ds = WikiTextDataset.create_dataset(
                    version='wikitext-103-v1',
                    split=split,
                    tokenizer=tokenizer,
                    max_length=args.seq_length
                )
                datasets.append(ds)
                logger.info(f"  ✅ Added wikitext103 ({len(ds)} samples)")
            elif name in ['programming', 'bookcorpus', 'openwebtext']:
                # These would need specific loaders - for now, log warning
                logger.warning(f"  ⚠️  Dataset '{name}' not yet implemented, skipping")
            else:
                logger.warning(f"  ⚠️  Unknown dataset '{name}', skipping")
        
        if not datasets:
            raise ValueError("No valid datasets loaded from combined list")
        
        # Concatenate all datasets
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset(datasets)
        logger.info(f"Combined dataset total: {len(combined)} samples")
        return combined
        
    elif args.dataset == 'wikitext2':
        return WikiTextDataset.create_dataset(
            version='wikitext-2-v1',
            split=split,
            tokenizer=tokenizer,
            max_length=args.seq_length
        )
    elif args.dataset == 'wikitext103':
        return WikiTextDataset.create_dataset(
            version='wikitext-103-v1',
            split=split,
            tokenizer=tokenizer,
            max_length=args.seq_length
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def create_dataloaders(args, tokenizer):
    """Create train and validation dataloaders."""
    logger.info(f"Loading {args.dataset} dataset...")
    
    train_dataset = load_dataset(args, tokenizer, split='train')
    val_dataset = load_dataset(args, tokenizer, split='validation')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def save_training_config(args, config, checkpoint_dir):
    """Save training configuration to JSON."""
    config_dict = {
        'args': vars(args),
        'model_config': {
            'task_type': str(config.task_type),
            'input_dim': config.input_dim,
            'hidden_dim': config.hidden_dim,
            'output_dim': config.output_dim,
            'num_layers': config.num_layers,
            'liquid_units': config.liquid_units,
            'spiking_units': config.spiking_units,
            'sequence_length': config.sequence_length,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'use_continual_learning': getattr(config, 'use_continual_learning', False),
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Add continual learning config if enabled
    if getattr(config, 'use_continual_learning', False):
        config_dict['continual_learning'] = {
            'episodic_memory_size': getattr(config, 'episodic_memory_size', 0),
            'replay_buffer_size': getattr(config, 'replay_buffer_size', 0),
            'ewc_lambda': getattr(config, 'ewc_lambda', 0),
            'si_c': getattr(config, 'si_c', 0),
            'consolidation_frequency': getattr(config, 'consolidation_frequency', 0),
            'enable_progressive_networks': getattr(config, 'enable_progressive_networks', False)
        }
    
    # Add mamba config if enabled
    if getattr(config, 'use_mamba', False):
        config_dict['mamba_integration'] = {
            'integration_mode': getattr(config, 'integration_mode', ''),
            'mamba_d_state': getattr(config, 'mamba_d_state', 0),
            'mamba_d_conv': getattr(config, 'mamba_d_conv', 0),
            'mamba_expand': getattr(config, 'mamba_expand', 0),
            'use_cross_attention': getattr(config, 'use_cross_attention', False),
            'use_adaptive_gating': getattr(config, 'use_adaptive_gating', False)
        }
    
    config_path = Path(checkpoint_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Training config saved to {config_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"{args.model_size}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            args.wandb = False
    
    # Print GPU info
    if args.device == 'cuda':
        print_gpu_memory_info()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create config
    config = setup_model_config(args)
    save_training_config(args, config, checkpoint_dir)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args, tokenizer)
    
    # Create model
    logger.info(f"Creating {args.model_size} model...")
    model = LiquidSpikingNetwork(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = LiquidSpikingTrainer(model, config)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Continual Learning: {'✅ ENABLED' if args.use_continual_learning else '❌ DISABLED'}")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = datetime.now()
        
        # Train epoch - use train_on_task for continual learning support
        if args.use_continual_learning:
            train_acc = trainer.train_on_task(
                task_id=epoch,  # Treat each epoch as a "task" for continual learning
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=1
            )
            train_loss = trainer.train_losses[-1] if trainer.train_losses else 0
            val_loss = trainer.val_losses[-1] if trainer.val_losses else 0
        else:
            # Standard training loop
            train_loss = 0
            model.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)
                
                loss = trainer.training_step(inputs, targets)
                train_loss += loss
                
                if batch_idx % args.log_interval == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{args.epochs} "
                        f"[{batch_idx}/{len(train_loader)}] "
                        f"Loss: {loss:.4f}"
                    )
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss, train_acc = trainer.evaluate(val_loader)
        
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        
        # Log to W&B
        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': train_acc,
                'epoch_time': epoch_time,
                'learning_rate': trainer.optimizer.param_groups[0]['lr']
            })
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {train_acc:.4f}, "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Continual learning stats
        if args.use_continual_learning and hasattr(trainer, 'continual_memory_system'):
            stats = trainer.continual_memory_system.get_memory_stats()
            logger.info(
                f"  🧠 Memory: {stats['episodic_memory_usage']}/{stats['episodic_memory_capacity']} | "
                f"Replay: {stats['replay_buffer_size']}/{stats['replay_buffer_capacity']} | "
                f"EWC Tasks: {stats['ewc_tasks']}"
            )
        
        # Early stopping check
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': train_acc,
                'config': config
            }, best_path)
            logger.info(f"✅ Saved best model to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': train_acc,
                'config': config
            }, checkpoint_path)
            logger.info(f"📁 Saved checkpoint to {checkpoint_path}")
    
    # Final save
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    logger.info(f"🎉 Training complete! Final model saved to {final_path}")
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
