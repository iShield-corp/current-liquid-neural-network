#!/usr/bin/env python3
"""
Liquid-Spiking Neural Network CLI Application

This CLI tool provides a complete interface for training, saving, loading, 
and running the hybrid liquid-spiking neural networks defined in main.py.

Usage:
    python cli.py train --task vision --epochs 20
    python cli.py load --model-path vision_model.pt --input-file test_image.npy
    python cli.py benchmark --model-path vision_model.pt
    python cli.py export --model-path vision_model.pt --format onnx
"""

import sys
import os
import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import time
import json

# Rich imports for beautiful CLI
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
)
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.tree import Tree
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.markdown import Markdown
from rich import print as rprint
from rich.status import Status
from rich.logging import RichHandler
from contextlib import contextmanager

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.main import (
    TaskType, ModelConfig, LiquidSpikingNetwork, LiquidSpikingTrainer,
    DatasetFactory, create_llm_config, create_vision_config, create_robotics_config,
    create_custom_config, save_config, load_config, print_config_summary,
    get_model_parameter_count, load_model, benchmark_model, export_onnx, 
    generate_text, evaluate_perplexity, train_llm_model, train_vision_model, 
    train_robotics_model, inference_example
)

# Import training fixes
try:
    from src.training.training_fixes import (
        apply_training_fixes,
        add_warmup_scheduler,
        GradientHealthMonitor,
        diagnose_training_stuck
    )
    FIXES_AVAILABLE = True
except ImportError:
    FIXES_AVAILABLE = False

# Initialize Rich console
console = Console()

def setup_logging():
    """Configure logging to work properly with Rich console."""
    # Create Rich handler for logging
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        show_level=False,
        markup=True,
        rich_tracebacks=True
    )
    
    # Configure root logger
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors by default
        format="%(message)s",
        handlers=[rich_handler]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("src.core.main").setLevel(logging.WARNING)
    logging.getLogger("src.utils.gpu_utils").setLevel(logging.WARNING)
    logging.getLogger("src.utils.memory_manager").setLevel(logging.WARNING)
    logging.getLogger("src.datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

class RichLogger:
    """Enhanced logger using Rich for beautiful output."""
    
    def __init__(self):
        self.console = console
    
    def info(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [green]ℹ[/green] {message}")
    
    def warning(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [yellow]⚠[/yellow] {message}")
    
    def error(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [red]✗[/red] {message}")
    
    def success(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [green]✓[/green] {message}")
    
    def header(self, title: str, subtitle: str = ""):
        """Create a beautiful header panel."""
        content = f"[bold white]{title}[/bold white]"
        if subtitle:
            content += f"\n[dim]{subtitle}[/dim]"
        
        panel = Panel(
            content,
            border_style="blue",
            padding=(1, 2),
            title="🧠 Liquid-Spiking Neural Network",
            title_align="left"
        )
        self.console.print(panel)

class LiquidSpikingCLI:
    """Enhanced CLI application with Rich graphics."""
    
    def __init__(self):
        # Setup logging first
        setup_logging()
        
        self.parser = self._create_parser()
        self.logger = RichLogger()
        self.console = console
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with all subcommands."""
        parser = argparse.ArgumentParser(
            description="🧠 Liquid-Spiking Neural Network CLI Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
[bold blue]Examples:[/bold blue]
  [green]# Train with GPT-4o competitive preset[/green]
  python cli.py train --task llm --config-preset gpt-4o --epochs 100
  
  [green]# Train with WikiText-103 (Recommended)[/green]
  python cli.py train --task llm --dataset wikitext103 --epochs 30
  
  [green]# Train with Combined Datasets[/green]
  python cli.py train --task llm --dataset combined --epochs 50
  
  [green]# Train a vision model for 20 epochs[/green]
  python cli.py train --task vision --epochs 20 --batch-size 64
  
  [green]# Load and run inference on an image[/green]
  python cli.py inference --model-path vision_model.pt --input-file test.npy
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Train command
        self._add_train_parser(subparsers)
        
        # Inference command
        self._add_inference_parser(subparsers)
        
        # Benchmark command
        self._add_benchmark_parser(subparsers)
        
        # Export command
        self._add_export_parser(subparsers)
        
        return parser
    
    def _add_train_parser(self, subparsers):
        """Add training subcommand parser."""
        train_parser = subparsers.add_parser('train', help='🚀 Train a neural network model')
        
        # Basic training parameters
        train_parser.add_argument('--task', choices=['llm', 'vision', 'robotics'], 
                                required=True, help='Type of task to train for')
        train_parser.add_argument('--epochs', type=int, default=10, 
                                help='Number of training epochs (default: 10)')
        train_parser.add_argument('--batch-size', type=int, 
                                help='Batch size for training (overrides config default)')
        train_parser.add_argument('--learning-rate', type=float,
                                help='Learning rate (overrides config default)')
        
        # Configuration preset options
        train_parser.add_argument('--config-preset', 
                                choices=['gpt-4o', 'claude-sonnet-4', 'ultra-advanced'], 
                                help='Use predefined configuration presets')
        
        # Neural network architecture parameters
        arch_group = train_parser.add_argument_group('Neural Network Architecture')
        arch_group.add_argument('--liquid-units', type=int,
                               help='Number of liquid neural network units')
        arch_group.add_argument('--spiking-units', type=int,
                               help='Number of spiking neural network units')
        arch_group.add_argument('--num-layers', type=int,
                               help='Number of hybrid liquid-spiking layers')
        arch_group.add_argument('--hidden-dim', type=int,
                               help='Hidden dimension size')
        arch_group.add_argument('--num-attention-heads', type=int,
                               help='Number of attention heads')
        arch_group.add_argument('--sequence-length', type=int,
                               help='Sequence length for training')
        
        # Dataset selection parameters (NEW)
        dataset_group = train_parser.add_argument_group('Dataset Selection', 
                                                        'Choose training dataset for LLM tasks')
        dataset_group.add_argument('--dataset', 
                                  choices=['programming', 'wikitext103', 'wikitext2', 'bookcorpus', 
                                          'ccnews', 'openwebtext', 'combined'],
                                  default='wikitext103',
                                  help='Dataset to use for LLM training (default: wikitext103). '
                                       'wikitext103 is RECOMMENDED for 100M+ param models (100M tokens). '
                                       'wikitext2 is TOO SMALL for large models (only 2M tokens). '
                                       'combined uses multiple datasets for best quality (200M+ tokens).')
        dataset_group.add_argument('--combined-datasets', type=str,
                                  default='wikitext103,bookcorpus,openwebtext',
                                  help='Comma-separated list of datasets for combined mode (default: wikitext103,bookcorpus,openwebtext)')
        dataset_group.add_argument('--dataset-cache-dir', type=str, default='./data',
                                  help='Directory to cache downloaded datasets (default: ./data)')
        
        # Tokenizer parameters
        tokenizer_group = train_parser.add_argument_group('Tokenizer Parameters')
        tokenizer_group.add_argument('--tokenizer', choices=['gpt2', 'gpt3', 'gpt4', 'o200k', 'codellama', 'llama2'],
                                   default='gpt2',
                                   help='Tokenizer type to use (default: gpt2)')
        
        # Advanced options
        adv_group = train_parser.add_argument_group('Advanced Options')
        adv_group.add_argument('--use-stdp', action='store_true', help='Enable STDP plasticity')
        adv_group.add_argument('--use-meta-plasticity', action='store_true', help='Enable meta-plasticity')
        adv_group.add_argument('--mixed-precision', action='store_true', default=True, help='Enable mixed precision')
        adv_group.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false',
                              help='Disable mixed precision training')
        adv_group.add_argument('--gradient-clip', type=float, help='Gradient clipping value')
        adv_group.add_argument('--use-fixes', action='store_true', 
                              help='🔧 Apply gradient flow fixes for zero-learning issues (RECOMMENDED)')
        adv_group.add_argument('--diagnose', action='store_true',
                              help='Run gradient diagnostics before training (requires --use-fixes)')
        adv_group.add_argument('--production-mode', action='store_true',
                              help='🚀 Use production training script (RECOMMENDED for LLM tasks)')
        adv_group.add_argument('--model-size', choices=['tiny', 'small', 'medium', 'large'],
                              default='small', help='Model size preset (for production mode)')
        adv_group.add_argument('--resume', type=str, help='Resume training from checkpoint path')
        adv_group.add_argument('--patience', type=int, default=10, help='Early stopping patience')
        adv_group.add_argument('--accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
        
        # Mamba Integration Options
        mamba_group = train_parser.add_argument_group('Mamba Integration')
        mamba_group.add_argument('--use-mamba', action='store_true', help='Enable Mamba integration')
        mamba_group.add_argument('--integration-mode', choices=['sequential', 'parallel', 'bidirectional'],
                                help='Mamba integration mode')
        mamba_group.add_argument('--mamba-d-state', type=int, help='Mamba state dimension')
        mamba_group.add_argument('--use-cross-attention', action='store_true', help='Enable cross-attention (for bidirectional mode)')
        mamba_group.add_argument('--use-adaptive-gating', action='store_true', help='Enable adaptive gating')
        
        # Training options
        train_parser.add_argument('--output-dir', type=str, default='./models',
                                help='Directory to save trained models (default: ./models)')
        train_parser.add_argument('--save-interval', type=int, default=5,
                                help='Save checkpoint every N epochs (default: 5)')
        train_parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                                help='Device to use for training (default: auto)')
        
        # Multi-GPU training options
        gpu_group = train_parser.add_argument_group('Multi-GPU Training')
        gpu_group.add_argument('--multi-gpu', action='store_true',
                              help='Enable multi-GPU training (auto-detect available GPUs)')
        gpu_group.add_argument('--gpu-strategy', choices=['auto', 'dp', 'ddp', 'none'], 
                              default='auto',
                              help='Multi-GPU strategy')
        
        # Continual Learning options
        continual_group = train_parser.add_argument_group('Continual Learning', 
                                                          'Advanced memory features for post-training learning')
        continual_group.add_argument('--use-continual-learning', action='store_true',
                                    help='Enable continual learning system (all 6 features)')
        continual_group.add_argument('--episodic-memory-size', type=int, default=10000,
                                    help='Episodic memory bank size (default: 10000)')
        continual_group.add_argument('--memory-key-dim', type=int, default=256,
                                    help='Memory key dimension (default: 256)')
        continual_group.add_argument('--ewc-lambda', type=float, default=5000.0,
                                    help='EWC regularization strength (default: 5000.0)')
        continual_group.add_argument('--si-c', type=float, default=0.1,
                                    help='Synaptic Intelligence coefficient (default: 0.1)')
        continual_group.add_argument('--consolidation-frequency', type=int, default=1000,
                                    help='Memory consolidation frequency in steps (default: 1000)')
        continual_group.add_argument('--replay-frequency', type=float, default=0.3,
                                    help='Experience replay frequency 0-1 (default: 0.3)')
        continual_group.add_argument('--replay-batch-size', type=int, default=16,
                                    help='Replay batch size (default: 16)')
        continual_group.add_argument('--replay-buffer-size', type=int, default=5000,
                                    help='Experience replay buffer size (default: 5000)')
        continual_group.add_argument('--enable-progressive-networks', action='store_true',
                                    help='Enable progressive network expansion (optional feature)')

    def _add_inference_parser(self, subparsers):
        parser = subparsers.add_parser('inference', help='💡 Run inference')
        parser.add_argument('--model-path', required=True, help='Path to trained model')
        parser.add_argument('--prompt', type=str, help='Text prompt for LLM')
        parser.add_argument('--max-length', type=int, default=100, help='Max generation length')
        parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
        parser.add_argument('--input-file', type=str, help='Input file for vision/robotics')

    def _add_benchmark_parser(self, subparsers):
        parser = subparsers.add_parser('benchmark', help='⚡ Benchmark model')
        parser.add_argument('--model-path', required=True)
        parser.add_argument('--iterations', type=int, default=100)

    def _add_export_parser(self, subparsers):
        parser = subparsers.add_parser('export', help='📦 Export model')
        parser.add_argument('--model-path', required=True)
        parser.add_argument('--format', choices=['onnx', 'torchscript'], default='onnx')
        parser.add_argument('--output-path', required=True)

    def run(self):
        args = self.parser.parse_args()
        if not args.command:
            self.parser.print_help()
            return
            
        self.logger.header("Liquid-Spiking Neural Network CLI")
        
        if args.command == 'train':
            self._handle_train(args)
        elif args.command == 'inference':
            self._handle_inference(args)
        elif args.command == 'benchmark':
            self._handle_benchmark(args)
        elif args.command == 'export':
            self._handle_export(args)

    def _handle_train(self, args):
        """Handle training command - route to production script if requested."""
        # Check if production mode requested for LLM tasks
        if getattr(args, 'production_mode', False) and args.task == 'llm':
            self._handle_production_training(args)
            return
        
        # Original training flow
        self.logger.info(f"🚀 Starting training for task: [bold cyan]{args.task}[/bold cyan]")
        
        # Create configuration
        if args.task == 'llm':
            tokenizer_type = getattr(args, 'tokenizer', 'gpt2')
            config = create_llm_config(tokenizer_type)
        elif args.task == 'vision':
            config = create_vision_config()
        else:
            config = create_robotics_config()
            
        # Apply overrides
        if args.num_layers: config.num_layers = args.num_layers
        if args.hidden_dim: config.hidden_dim = args.hidden_dim
        if args.liquid_units: config.liquid_units = args.liquid_units
        if args.spiking_units: config.spiking_units = args.spiking_units
        if args.batch_size: config.batch_size = args.batch_size
        if args.learning_rate: config.learning_rate = args.learning_rate
        if args.sequence_length:
            config.sequence_length = args.sequence_length
            config.max_position_embeddings = args.sequence_length
        if args.num_attention_heads: config.num_attention_heads = args.num_attention_heads
        if args.gradient_clip: config.gradient_clip = args.gradient_clip
        
        if args.use_stdp: config.use_stdp = True
        if args.use_meta_plasticity: config.use_meta_plasticity = True
        if args.use_mamba: config.use_mamba = True
        if args.integration_mode: config.integration_mode = args.integration_mode
        if args.mamba_d_state: config.mamba_d_state = args.mamba_d_state
        if args.use_cross_attention: config.use_cross_attention = True
        if args.use_adaptive_gating: config.use_adaptive_gating = True
        
        # Apply continual learning settings
        if getattr(args, 'use_continual_learning', False):
            config.use_continual_learning = True
            if args.episodic_memory_size: config.episodic_memory_size = args.episodic_memory_size
            if args.memory_key_dim: config.memory_key_dim = args.memory_key_dim
            if args.ewc_lambda: config.ewc_lambda = args.ewc_lambda
            if args.si_c: config.si_c = args.si_c
            if args.consolidation_frequency: 
                config.consolidation_frequency = args.consolidation_frequency
                config.consolidation_interval = args.consolidation_frequency
            if args.replay_frequency: config.replay_frequency = args.replay_frequency
            if args.replay_batch_size: config.replay_batch_size = args.replay_batch_size
            if hasattr(args, 'replay_buffer_size') and args.replay_buffer_size:
                config.replay_buffer_size = args.replay_buffer_size
            if args.enable_progressive_networks: config.enable_progressive_networks = True
        
        # Create model
        self.logger.info("🧠 Initializing model...")
        model = LiquidSpikingNetwork(config)
        
        # Continue with training
        self._continue_legacy_training(args, config, model)
    
    def _handle_production_training(self, args):
        """Handle production mode training."""
        import subprocess
        
        self.logger.header("Production Training Mode")
        self.logger.info("🚀 Using optimized production training script")
        self.logger.info("   ✅ Fixed Trainer class with all gradient flow fixes")
        self.logger.info("   ✅ Proper LR scheduling from epoch 0")
        self.logger.info("   ✅ Checkpointing and early stopping")
        self.logger.info("")
        
        # Build command with ALL arguments
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'train_production.py'
        )
        
        cmd = [
            sys.executable,
            str(script_path),
            '--model-size', args.model_size,
            '--num-layers', str(args.num_layers),
            '--hidden-dim', str(args.hidden_dim),
            '--liquid-units', str(args.liquid_units),           # NEW
            '--spiking-units', str(args.spiking_units),         # NEW
            '--num-attention-heads', str(args.num_attention_heads),  # NEW
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--learning-rate', str(args.learning_rate),
            '--gradient-clip', str(args.gradient_clip),
            '--seq-length', str(args.sequence_length),
            '--dataset', args.dataset,
            '--tokenizer', args.tokenizer,
            '--checkpoint-dir', args.output_dir,
            '--checkpoint-freq', str(args.save_interval),
            '--accumulation-steps', str(getattr(args, 'accumulation_steps', 1)),
            '--patience', str(getattr(args, 'patience', 10)),
        ]
        
        # Add boolean flags
        if args.use_mamba:                                      # NEW
            cmd.append('--use-mamba')
        if args.integration_mode:                               # NEW
            cmd.extend(['--integration-mode', args.integration_mode])
        if args.mamba_d_state:                                  # NEW
            cmd.extend(['--mamba-d-state', str(args.mamba_d_state)])
        if args.use_cross_attention:                            # NEW
            cmd.append('--use-cross-attention')
        if args.use_adaptive_gating:                            # NEW
            cmd.append('--use-adaptive-gating')
        if args.use_stdp:                                       # NEW
            cmd.append('--use-stdp')
        if args.use_meta_plasticity:                            # NEW
            cmd.append('--use-meta-plasticity')
        if args.mixed_precision:
            cmd.append('--mixed-precision')
        
        # Device
        if args.device != 'auto':
            cmd.extend(['--device', args.device])
        
        self.logger.info(f"Executing: {' '.join(cmd)}")
        self.logger.info("")
        
        # Execute production training script
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Production training failed with exit code {e.returncode}")
            return e.returncode
        except KeyboardInterrupt:
            self.logger.warning("⚠️  Training interrupted by user")
            return 1
        
    def _continue_legacy_training(self, args):
        """Continue using legacy training with production script."""
        self.console.print(Panel.fit(
            "Production Training Mode",
            style="bold cyan"
        ))
        
        self.console.print("🚀 Using optimized production training script", style="info")
        self.console.print("   ✅ Fixed Trainer class with all gradient flow fixes", style="info")
        self.console.print("   ✅ Proper LR scheduling from epoch 0", style="info")
        self.console.print("   ✅ Checkpointing and early stopping", style="info")
        self.console.print("")
        
        # Build command for production script
        script_path = Path(__file__).parent / "train_production.py"
        
        # Build base command
        cmd = [
            sys.executable,
            str(script_path),
            "--model-size", args.model_size,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seq-length", str(args.sequence_length),
            "--dataset", args.dataset,
            "--tokenizer", args.tokenizer,
            "--checkpoint-dir", args.output_dir,
            "--checkpoint-freq", str(args.save_interval),
            "--accumulation-steps", str(args.accumulation_steps),
            "--patience", str(args.patience),
        ]
        
        # Only add optional args if they have actual values (not None)
        if args.num_layers is not None:
            cmd.extend(["--num-layers", str(args.num_layers)])
        if args.hidden_dim is not None:
            cmd.extend(["--hidden-dim", str(args.hidden_dim)])
        if args.liquid_units is not None:
            cmd.extend(["--liquid-units", str(args.liquid_units)])
        if args.spiking_units is not None:
            cmd.extend(["--spiking-units", str(args.spiking_units)])
        if args.num_attention_heads is not None:
            cmd.extend(["--num-attention-heads", str(args.num_attention_heads)])
        if args.learning_rate is not None:
            cmd.extend(["--learning-rate", str(args.learning_rate)])
        if args.gradient_clip is not None:
            cmd.extend(["--gradient-clip", str(args.gradient_clip)])
        
        if args.mixed_precision:
            cmd.append("--mixed-precision")
        
        # Add continual learning args if enabled
        if args.use_continual_learning:
            cmd.append("--use-continual-learning")
            cmd.extend([
                "--episodic-memory-size", str(args.episodic_memory_size),
                "--replay-buffer-size", str(args.replay_buffer_size),
                "--ewc-lambda", str(args.ewc_lambda),
                "--si-c", str(args.si_c),
                "--consolidation-frequency", str(args.consolidation_frequency),
                "--replay-frequency", str(args.replay_frequency)
            ])
            
            if args.enable_progressive_networks:
                cmd.append("--enable-progressive-networks")
        
        # Print command for debugging
        self.console.print(f"Executing: {' '.join(cmd)}", style="info")
        self.console.print("")
        
        # Execute
        import subprocess
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            self.console.print(f"❌ Training failed with error code {e.returncode}", style="error")
            return e.returncode

    def _handle_inference(self, args):
        self.logger.header("Inference")
        self.logger.info(f"📂 Loading model from {args.model_path}")
        
        # Detect task type from path or default to LLM
        task_type = TaskType.LLM
        if 'vision' in args.model_path.lower():
            task_type = TaskType.VISION
        elif 'robotics' in args.model_path.lower():
            task_type = TaskType.ROBOTICS
        
        model, config = load_model(args.model_path, task_type)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        if task_type == TaskType.LLM and args.prompt:
            self.logger.info(f"📝 Prompt: {args.prompt}")
            self.logger.info(f"🔧 Settings: max_length={args.max_length}, temperature={args.temperature}")
            self.console.print("\n[yellow]🤖 Generating...[/yellow]\n")
            
            # Detect tokenizer type from config
            tokenizer_type = getattr(config, 'tokenizer_type', None)
            vocab_size = getattr(config, 'vocab_size', 50257)
            
            # Determine tokenizer based on vocab size if not specified
            if tokenizer_type is None:
                if vocab_size >= 200000:
                    tokenizer_type = 'o200k'
                elif vocab_size >= 100000:
                    tokenizer_type = 'gpt4'
                else:
                    tokenizer_type = 'gpt2'
            
            self.logger.info(f"🔤 Using tokenizer: {tokenizer_type} (vocab: {vocab_size:,})")
            
            try:
                generated_text = self._generate_text_with_model(
                    model, config, args.prompt, 
                    max_length=args.max_length,
                    temperature=args.temperature,
                    tokenizer_type=tokenizer_type,
                    device=device
                )
                
                # Display prompt and generated text separately
                self.console.print("[green]" + "="*70 + "[/green]")
                self.console.print(f"[bold cyan]📝 Prompt:[/bold cyan] {args.prompt}")
                self.console.print("[green]" + "-"*70 + "[/green]")
                rprint(Panel(generated_text, title="✨ Generated Text (New Tokens Only)", border_style="green"))
                self.console.print("[green]" + "-"*70 + "[/green]")
                self.console.print(f"[bold cyan]📄 Full Output:[/bold cyan] {args.prompt}{generated_text}")
                self.console.print("[green]" + "="*70 + "[/green]")
                
            except Exception as e:
                self.logger.error(f"Generation failed: {e}")
                import traceback
                traceback.print_exc()
        
        elif args.input_file:
            # Handle vision/robotics inference
            import numpy as np
            self.logger.info(f"📁 Loading input from {args.input_file}")
            input_data = np.load(args.input_file)
            input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device)
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            self.logger.success(f"Output shape: {output.shape}")
            self.console.print(f"[cyan]Output:[/cyan] {output}")
        
        else:
            self.logger.warning("⚠️ No prompt or input file provided")
            self.console.print("Usage:")
            self.console.print("  --prompt 'Your text here' for LLM generation")
            self.console.print("  --input-file data.npy for vision/robotics inference")
    
    def _generate_text_with_model(self, model, config, prompt, max_length=100, 
                                   temperature=0.8, tokenizer_type='o200k', device='cuda'):
        """Generate text using the model with proper tokenizer handling."""
        import torch.nn.functional as F
        
        # Load tokenizer based on type
        if tokenizer_type in ['o200k', 'gpt4', 'gpt3']:
            import tiktoken
            encoding_map = {
                'o200k': 'o200k_base',
                'gpt4': 'cl100k_base',
                'gpt3': 'p50k_base'
            }
            encoding = tiktoken.get_encoding(encoding_map.get(tokenizer_type, 'o200k_base'))
            
            # Encode prompt
            input_ids = encoding.encode(prompt, allowed_special='all')
            eos_token_id = encoding.eot_token
            
            def decode_fn(ids):
                return encoding.decode(ids)
        else:
            # Use transformers tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            input_ids = tokenizer.encode(prompt)
            eos_token_id = tokenizer.eos_token_id
            
            def decode_fn(ids):
                return tokenizer.decode(ids, skip_special_tokens=True)
        
        # Convert to tensor
        generated = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Generation loop
        model.eval()
        with torch.no_grad():
            for step in range(max_length):
                # Get model output
                outputs = model(generated)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Get logits for last position
                if logits.dim() == 3:
                    next_token_logits = logits[:, -1, :]
                elif logits.dim() == 2:
                    next_token_logits = logits
                else:
                    next_token_logits = logits.view(1, -1)
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Sample from distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                
                # Progress indicator every 10 tokens
                if (step + 1) % 10 == 0:
                    self.console.print(f"  Generated {step + 1}/{max_length} tokens...", end='\r')
        
        self.console.print(f"  ✅ Generated {generated.shape[1] - len(input_ids)} new tokens")
        
        # Decode only the NEW tokens (exclude prompt)
        generated_ids = generated[0].tolist()
        new_token_ids = generated_ids[len(input_ids):]  # Only new tokens
        
        # Return only the newly generated text (without prompt)
        generated_text = decode_fn(new_token_ids)
        
        return generated_text
            
    def _handle_benchmark(self, args):
        self.logger.header("Benchmark")
        # Simplified benchmark implementation
        self.logger.info(f"Benchmarking model: {args.model_path}")
        # Call actual benchmark function if needed
        
    def _handle_export(self, args):
        self.logger.header("Export")
        self.logger.info(f"Exporting model to {args.format}...")
        # Call actual export function if needed

    def _display_training_summary(self, epoch_history, total_epochs):
        """Display a comprehensive training summary with epoch comparisons."""
        if not epoch_history:
            return
        
        self.console.print("\n")
        self.logger.header("Training Summary", "Epoch-by-Epoch Performance Analysis")
        
        # Create comparison table
        table = Table(title="📊 Training Progress Overview", show_header=True, header_style="bold magenta")
        table.add_column("Epoch", style="cyan", justify="right")
        table.add_column("Train Loss", justify="right")
        table.add_column("Val Loss", justify="right")
        table.add_column("Val Acc", justify="right")
        table.add_column("Grad Norm", justify="right")
        table.add_column("Δ Train", justify="right")
        table.add_column("Δ Val", justify="right")
        table.add_column("Status", justify="center")
        
        # Add rows with delta calculations
        for i, metrics in enumerate(epoch_history):
            epoch_num = str(metrics['epoch'])
            train_loss = f"{metrics['train_loss']:.4f}"
            val_loss = f"{metrics['val_loss']:.4f}"
            val_acc = f"{metrics['val_acc']:.4f}"
            grad_norm = f"{metrics['grad_norm']:.3f}"
            
            # Calculate deltas from previous epoch
            if i > 0:
                prev = epoch_history[i-1]
                delta_train = metrics['train_loss'] - prev['train_loss']
                delta_val = metrics['val_loss'] - prev['val_loss']
                
                # Color code improvements (green) vs regressions (red)
                delta_train_str = f"[green]{delta_train:+.4f}[/green]" if delta_train < 0 else f"[red]{delta_train:+.4f}[/red]"
                delta_val_str = f"[green]{delta_val:+.4f}[/green]" if delta_val < 0 else f"[red]{delta_val:+.4f}[/red]"
            else:
                delta_train_str = "-"
                delta_val_str = "-"
            
            # Status indicator
            status = "⭐ BEST" if metrics['is_best'] else "✓"
            
            table.add_row(
                epoch_num, train_loss, val_loss, val_acc, grad_norm,
                delta_train_str, delta_val_str, status
            )
        
        self.console.print(table)
        
        # Summary statistics
        best_epoch = min(epoch_history, key=lambda x: x['val_loss'])
        final_epoch = epoch_history[-1]
        
        self.console.print("\n[bold cyan]📈 Key Statistics:[/bold cyan]")
        self.console.print(f"   • Best Epoch: {best_epoch['epoch']} (Val Loss: {best_epoch['val_loss']:.4f})")
        self.console.print(f"   • Final Train Loss: {final_epoch['train_loss']:.4f}")
        self.console.print(f"   • Final Val Loss: {final_epoch['val_loss']:.4f}")
        self.console.print(f"   • Final Val Accuracy: {final_epoch['val_acc']:.4f}")
        
        # Calculate total improvement
        if len(epoch_history) > 1:
            first_epoch = epoch_history[0]
            total_train_improvement = first_epoch['train_loss'] - final_epoch['train_loss']
            total_val_improvement = first_epoch['val_loss'] - final_epoch['val_loss']
            
            train_color = "green" if total_train_improvement > 0 else "red"
            val_color = "green" if total_val_improvement > 0 else "red"
            
            self.console.print(f"\n[bold cyan]📉 Total Improvement:[/bold cyan]")
            self.console.print(f"   • Train Loss: [{train_color}]{total_train_improvement:+.4f}[/{train_color}]")
            self.console.print(f"   • Val Loss: [{val_color}]{total_val_improvement:+.4f}[/{val_color}]")
        
        self.console.print("\n")


def main():
    """Main entry point for the CLI."""
    cli = LiquidSpikingCLI()
    cli.run()

if __name__ == "__main__":
    main()
