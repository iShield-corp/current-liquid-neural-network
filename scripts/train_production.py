#!/usr/bin/env python3
"""
Production-ready training script for LLM training.
Works with both CLI and GUI, includes checkpointing, validation, early stopping.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import json
from pathlib import Path
from datetime import datetime

from src.core.main import (
    create_llm_config,
    LiquidSpikingNetwork,
    LiquidSpikingTrainer,
    WikiTextDataset,
    TextDataset,
    TaskType
)

# Import memory manager utilities
from src.utils.memory_manager import print_gpu_memory_info


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
    parser.add_argument('--liquid-units', type=int, default=256,
                        help='Number of units in liquid layers')
    parser.add_argument('--spiking-units', type=int, default=128,
                        help='Number of units in spiking layers')
    parser.add_argument('--num-attention-heads', type=int, default=8,
                        help='Number of attention heads')
    
    # Advanced features
    parser.add_argument('--use-mamba', action='store_true',
                        help='Enable Mamba SSM layers')
    parser.add_argument('--integration-mode', type=str, default='parallel',
                        choices=['parallel', 'sequential', 'bidirectional'],
                        help='Mamba-Liquid integration mode')
    parser.add_argument('--mamba-d-state', type=int, default=16,
                        help='Mamba SSM state dimension')
    parser.add_argument('--use-cross-attention', action='store_true',
                        help='Enable cross-attention between components')
    parser.add_argument('--use-adaptive-gating', action='store_true',
                        help='Enable adaptive gating mechanism')
    parser.add_argument('--use-stdp', action='store_true',
                        help='Enable STDP plasticity')
    parser.add_argument('--use-meta-plasticity', action='store_true',
                        help='Enable meta-plasticity')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size (default: 4 for large models)')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--gradient-clip', type=float, default=0.5,
                        help='Gradient clipping threshold')
    parser.add_argument('--seq-length', type=int, default=128,
                        help='Sequence length for training (default: 128 for large models)')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'wikitext103', 'combined', 'programming'],
                        help='Dataset to use for training')
    parser.add_argument('--combined-datasets', type=str,
                        default='wikitext103,bookcorpus,openwebtext',
                        help='Comma-separated list of datasets for combined mode')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        choices=['gpt2', 'gpt3', 'gpt4', 'o200k', 'codellama', 'llama2'],
                        help='Tokenizer to use')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpoint configuration
    parser.add_argument('--checkpoint-dir', type=str, default='./models',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Optimization
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--no-mixed-precision', dest='mixed_precision',
                        action='store_false',
                        help='Disable mixed precision training')
    parser.add_argument('--accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps (default: 4, effective batch=batch_size*4)')
    parser.add_argument('--checkpoint-activations', action='store_true', default=True,
                        help='Use gradient checkpointing to save memory (slower but uses less VRAM)')
    parser.add_argument('--no-checkpoint-activations', dest='checkpoint_activations',
                        action='store_false',
                        help='Disable gradient checkpointing')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Log every N batches')
    parser.add_argument('--save-best-only', action='store_true',
                        help='Only save best model checkpoint')
    
    return parser.parse_args()


def get_model_preset(preset_name):
    """Get model configuration preset."""
    presets = {
        'tiny': {'num_layers': 2, 'hidden_dim': 256, 'input_dim': 384},
        'small': {'num_layers': 4, 'hidden_dim': 512, 'input_dim': 768},
        'medium': {'num_layers': 8, 'hidden_dim': 768, 'input_dim': 1024},
        'large': {'num_layers': 12, 'hidden_dim': 1024, 'input_dim': 1536},
    }
    return presets.get(preset_name, presets['small'])


def setup_model_config(args):
    """Create model configuration from arguments."""
    # Start with base config using specified tokenizer
    config = create_llm_config(args.tokenizer, 'llm')
    
    # Apply preset
    preset = get_model_preset(args.model_size)
    config.num_layers = preset['num_layers']
    config.hidden_dim = preset['hidden_dim']
    config.input_dim = preset['input_dim']
    
    # Override with explicit arguments
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    
    # Liquid-Spiking specific parameters
    config.liquid_units = args.liquid_units
    config.spiking_units = args.spiking_units
    config.num_attention_heads = args.num_attention_heads
    config.use_mamba = args.use_mamba
    config.integration_mode = args.integration_mode
    config.mamba_d_state = args.mamba_d_state
    config.use_cross_attention = args.use_cross_attention
    config.use_adaptive_gating = args.use_adaptive_gating
    config.use_stdp = args.use_stdp
    config.use_meta_plasticity = args.use_meta_plasticity
    
    # Training settings
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.gradient_clip = args.gradient_clip
    config.sequence_length = args.seq_length
    config.mixed_precision = args.mixed_precision
    config.num_epochs = args.epochs
    
    return config


def load_dataset(args, tokenizer, split='train'):
    """Load and prepare dataset."""
    print(f"\n📚 Loading {args.dataset} dataset ({split} split)...")
    
    if args.dataset == 'wikitext2':
        texts = WikiTextDataset.load_wikitext2(split=split)
    elif args.dataset == 'wikitext103':
        texts = WikiTextDataset.load_wikitext103(split=split)
    elif args.dataset == 'programming':
        # Use DatasetFactory for programming dataset
        from src.core.main import DatasetFactory
        print("Loading programming dataset...")
        full_dataset, _ = DatasetFactory.create_llm_dataset(
            vocab_size=tokenizer.vocab_size,
            seq_length=args.seq_length,
            tokenizer_type='gpt2',
            dataset_type='programming',
            cache_dir='./data'
        )
        # Return the full dataset - caller will split it
        return full_dataset
    else:  # combined
        # Parse combined datasets list
        dataset_names = [name.strip() for name in args.combined_datasets.split(',')]
        print(f"Loading combined datasets: {', '.join(dataset_names)}")
        
        all_texts = []
        for dataset_name in dataset_names:
            if dataset_name == 'wikitext2':
                texts = WikiTextDataset.load_wikitext2(split=split)
                all_texts.extend(texts)
            elif dataset_name == 'wikitext103':
                texts = WikiTextDataset.load_wikitext103(split=split)
                all_texts.extend(texts)
            elif dataset_name == 'programming':
                texts = WikiTextDataset.load_programming_dataset(split=split)
                all_texts.extend(texts)
            elif dataset_name == 'bookcorpus':
                texts = WikiTextDataset.load_bookcorpus(split=split)
                all_texts.extend(texts)
            elif dataset_name == 'ccnews':
                texts = WikiTextDataset.load_ccnews(split=split)
                all_texts.extend(texts)
            elif dataset_name == 'openwebtext':
                texts = WikiTextDataset.load_openwebtext(split=split)
                all_texts.extend(texts)
            else:
                print(f"⚠️  Unknown dataset: {dataset_name}, skipping...")
        
        texts = all_texts
        print(f"✅ Combined: {len(texts):,} texts from {len(dataset_names)} datasets")
    
    # Create dataset from texts
    dataset = TextDataset(texts, tokenizer, seq_length=args.seq_length)
    
    print(f"✅ Dataset ready: {len(dataset):,} examples")
    print(f"   Total tokens: {len(dataset) * args.seq_length:,}")
    
    return dataset


def create_dataloaders(args, tokenizer):
    """Create train and validation dataloaders."""
    # Load datasets
    if args.dataset == 'programming':
        # Programming dataset is already split by DatasetFactory
        full_dataset = load_dataset(args, tokenizer, split='train')
        
        # Split into train/val
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        print(f"\n📊 Split dataset:")
        print(f"   Train: {train_size:,} examples")
        print(f"   Val: {val_size:,} examples")
    else:
        # WikiText datasets have separate splits
        train_dataset = load_dataset(args, tokenizer, split='train')
        val_dataset = load_dataset(args, tokenizer, split='validation')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"\n📦 DataLoaders created:")
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Val batches: {len(val_loader):,}")
    
    return train_loader, val_loader


def save_training_config(args, config, checkpoint_dir):
    """Save training configuration to JSON."""
    config_dict = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'model_config': {
            'num_layers': config.num_layers,
            'hidden_dim': config.hidden_dim,
            'input_dim': config.input_dim,
            'vocab_size': config.vocab_size,
            'sequence_length': config.sequence_length,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
        }
    }
    
    config_path = Path(checkpoint_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"💾 Training config saved to {config_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Print banner
    print("=" * 80)
    print("🚀 PRODUCTION TRAINING - Liquid-Spiking Neural Network")
    print("=" * 80)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Display detailed GPU memory configuration
        print_gpu_memory_info()
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Model Configuration - APPLY ALL ARGUMENTS
    print("\n⚙️  Model Configuration:")
    config = create_llm_config(args.tokenizer)
    
    # Apply preset defaults if arguments not explicitly provided
    if args.model_size:
        size_configs = {
            'tiny': {'layers': 4, 'hidden': 256, 'liquid': 128, 'spiking': 64, 'heads': 4},
            'small': {'layers': 6, 'hidden': 512, 'liquid': 256, 'spiking': 128, 'heads': 8},
            'medium': {'layers': 8, 'hidden': 768, 'liquid': 384, 'spiking': 192, 'heads': 12},
            'large': {'layers': 12, 'hidden': 1024, 'liquid': 512, 'spiking': 256, 'heads': 16},
        }
        preset = size_configs.get(args.model_size, size_configs['small'])
        # Apply preset as defaults (will be overridden if explicit args provided)
        if args.num_layers is None:
            config.num_layers = preset['layers']
        if args.hidden_dim is None:
            config.hidden_dim = preset['hidden']
    
    # Override with explicit arguments (these take precedence)
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    config.liquid_units = args.liquid_units
    config.spiking_units = args.spiking_units
    config.num_attention_heads = args.num_attention_heads
    
    # Advanced features
    config.use_mamba = args.use_mamba
    config.integration_mode = args.integration_mode
    config.mamba_d_state = args.mamba_d_state
    config.use_cross_attention = args.use_cross_attention
    config.use_adaptive_gating = args.use_adaptive_gating
    config.use_stdp = args.use_stdp
    config.use_meta_plasticity = args.use_meta_plasticity
    
    # Training parameters
    config.sequence_length = args.seq_length
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.epochs
    config.gradient_clip = args.gradient_clip
    config.use_gradient_checkpointing = args.checkpoint_activations  # Enable gradient checkpointing
    
    print(f"   Size preset: {args.model_size}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Liquid units: {config.liquid_units}")
    print(f"   Spiking units: {config.spiking_units}")
    print(f"   Attention heads: {config.num_attention_heads}")
    print(f"   Mamba SSM: {'✅ Enabled' if config.use_mamba else '❌ Disabled'}")
    if config.use_mamba:
        print(f"      Mode: {config.integration_mode}")
        print(f"      State dim: {config.mamba_d_state}")
    print(f"   Cross-Attention: {'✅' if config.use_cross_attention else '❌'}")
    print(f"   Adaptive Gating: {'✅' if config.use_adaptive_gating else '❌'}")
    print(f"   STDP Plasticity: {'✅' if config.use_stdp else '❌'}")
    print(f"   Meta-Plasticity: {'✅' if config.use_meta_plasticity else '❌'}")
    print(f"   Input dim: {config.input_dim}")
    print(f"   Vocab size: {config.vocab_size:,}")
    print(f"   Sequence length: {config.sequence_length}")
    
    # Setup tokenizer
    print("\n🔤 Loading tokenizer...")
    
    # For o200k and other tiktoken-based tokenizers, we need special handling
    if args.tokenizer in ['o200k', 'gpt4', 'gpt3']:
        try:
            import tiktoken
            # Map to tiktoken encoding names
            tiktoken_map = {
                'o200k': 'o200k_base',
                'gpt4': 'cl100k_base',
                'gpt3': 'p50k_base'
            }
            encoding_name = tiktoken_map[args.tokenizer]
            encoding = tiktoken.get_encoding(encoding_name)
            
            # Create a wrapper class to make tiktoken compatible with HuggingFace interface
            class TiktokenWrapper:
                def __init__(self, encoding, vocab_size):
                    self.encoding = encoding
                    self.vocab_size = vocab_size
                    self.eos_token = '<|endoftext|>'
                    self.pad_token = '<|endoftext|>'
                    # Encode special tokens with allowed_special parameter
                    self.eos_token_id = encoding.encode(self.eos_token, allowed_special={self.eos_token})[0]
                    self.pad_token_id = self.eos_token_id
                    
                def encode(self, text, **kwargs):
                    # Allow special tokens by default
                    return self.encoding.encode(text, allowed_special='all')
                    
                def decode(self, ids, **kwargs):
                    return self.encoding.decode(ids)
                    
                def __len__(self):
                    return self.vocab_size
            
            # Get vocab size from config
            vocab_size_map = {
                'o200k': 200019,
                'gpt4': 100277,
                'gpt3': 50281
            }
            tokenizer = TiktokenWrapper(encoding, vocab_size_map[args.tokenizer])
            print(f"   Tokenizer: {args.tokenizer} via tiktoken (vocab: {tokenizer.vocab_size:,})")
            
        except ImportError:
            print(f"   ⚠️  tiktoken not available, falling back to GPT-2")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            print(f"   Tokenizer: GPT-2 fallback (vocab: {len(tokenizer):,})")
    else:
        # Use HuggingFace tokenizers for others
        tokenizer_map = {
            'gpt2': 'gpt2',
            'codellama': 'codellama/CodeLlama-7b-hf',
            'llama2': 'meta-llama/Llama-2-7b-hf'
        }
        tokenizer_name = tokenizer_map.get(args.tokenizer, 'gpt2')
        
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
            tokenizer.pad_token = tokenizer.eos_token
            print(f"   Tokenizer: {args.tokenizer} (vocab: {len(tokenizer):,})")
        except Exception as e:
            print(f"   ⚠️  Failed to load {tokenizer_name}, falling back to GPT-2")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            print(f"   Tokenizer: GPT-2 fallback (vocab: {len(tokenizer):,})")
    
    # Load datasets
    train_loader, val_loader = create_dataloaders(args, tokenizer)
    
    # Create model
    print("\n🏗️  Creating model...")
    model = LiquidSpikingNetwork(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1e6:.1f} MB (FP32)")
    
    # Create trainer
    print("\n🎓 Setting up trainer...")
    trainer = LiquidSpikingTrainer(
        model=model,
        config=config
    )
    
    # Set early stopping patience and accumulation steps
    trainer.max_patience = args.patience
    if hasattr(trainer, 'gradient_accumulation_steps'):
        trainer.gradient_accumulation_steps = args.accumulation_steps
    
    print(f"   Learning rate: {config.learning_rate:.2e}")
    print(f"   Gradient clip: {config.gradient_clip}")
    print(f"   Mixed precision: {config.mixed_precision}")
    print(f"   Accumulation steps: {args.accumulation_steps}")
    print(f"   Early stopping patience: {args.patience}")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"   Resuming from epoch {start_epoch}")
    
    # Save training configuration
    save_training_config(args, config, checkpoint_dir)
    
    # Training
    print("\n" + "=" * 80)
    print("🎯 STARTING TRAINING")
    print("=" * 80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("=" * 80)
    
    try:
        # Train with OOM error recovery
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs
        )
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Save final model
        final_path = checkpoint_dir / 'final_model.pt'
        trainer.save_checkpoint(str(final_path))
        print(f"\n💾 Final model saved to {final_path}")
        
        # Print summary
        print("\n📊 Training Summary:")
        print(f"   Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"   Final learning rate: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        print(f"   Total epochs: {len(trainer.train_losses)}")
        
        return 0
    
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n\n💥 CUDA Out of Memory Error!")
        print("Attempting emergency cleanup and checkpoint save...")
        
        # Emergency cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        # Try to save checkpoint
        try:
            oom_path = checkpoint_dir / 'oom_checkpoint.pt'
            trainer.save_checkpoint(str(oom_path))
            print(f"💾 Emergency checkpoint saved to {oom_path}")
            print("\n📋 Recovery suggestions:")
            print("   1. Reduce --batch-size")
            print("   2. Reduce --seq-length")
            print("   3. Increase --accumulation-steps")
            print("   4. Reduce model size (--num-layers, --hidden-dim)")
            print(f"   5. Resume from checkpoint: --resume {oom_path}")
        except Exception as save_error:
            print(f"⚠️  Could not save emergency checkpoint: {save_error}")
        
        return 2
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Saving checkpoint...")
        interrupt_path = checkpoint_dir / 'interrupted_model.pt'
        trainer.save_checkpoint(str(interrupt_path))
        print(f"💾 Checkpoint saved to {interrupt_path}")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save checkpoint on any error
        try:
            error_path = checkpoint_dir / 'error_checkpoint.pt'
            trainer.save_checkpoint(str(error_path))
            print(f"\n💾 Error checkpoint saved to {error_path}")
        except:
            pass
        
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
