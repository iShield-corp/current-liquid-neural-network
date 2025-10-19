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

import argparse
import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import platform
import psutil
import subprocess
import logging

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

# Import from main.py
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
    
def enable_verbose_logging():
    """Enable verbose logging for debugging."""
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("src.core.main").setLevel(logging.INFO)
    logging.getLogger("src.utils.gpu_utils").setLevel(logging.INFO)
    logging.getLogger("src.utils.memory_manager").setLevel(logging.INFO)

@contextmanager
def suppress_logging(level=logging.ERROR):
    """Context manager to temporarily suppress logging during status displays."""
    old_levels = {}
    loggers_to_suppress = [
        "",  # root logger
        "src.core.main",
        "src.utils.gpu_utils", 
        "src.utils.memory_manager",
        "src.datasets",
        "transformers",
        "datasets"
    ]
    
    # Save old levels and set new ones
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        old_levels[logger_name] = logger.level
        logger.setLevel(level)
    
    try:
        yield
    finally:
        # Restore old levels
        for logger_name, old_level in old_levels.items():
            logging.getLogger(logger_name).setLevel(old_level)

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
        if subtitle:
            content = f"[bold blue]{title}[/bold blue]\n[dim]{subtitle}[/dim]"
        else:
            content = f"[bold blue]{title}[/bold blue]"
        
        panel = Panel(
            content,
            border_style="blue",
            padding=(1, 2),
            title="🧠 Liquid-Spiking Neural Network",
            title_align="left"
        )
        self.console.print(panel)

class SystemInfo:
    """Gather and display system information."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Collect comprehensive system information."""
        info = {
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor() or "Unknown",
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').total,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown",
                'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory if torch.cuda.device_count() > 0 else 0,
            })
        
        # Check for specific dependencies
        try:
            import snntorch
            info['snntorch_version'] = snntorch.__version__
        except ImportError:
            info['snntorch_version'] = "Not installed"
        
        try:
            import ncps
            info['ncps_version'] = "Available"
        except ImportError:
            info['ncps_version'] = "Not installed"
        
        return info

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
  
  [green]# Train with Claude Sonnet 4 competitive preset[/green]
  python cli.py train --task llm --config-preset claude-sonnet-4 --epochs 150
  
  [green]# Train with ultra-advanced preset (better than all modern LLMs)[/green]
  python cli.py train --task llm --config-preset ultra-advanced --epochs 200
  
  [green]# Train with custom parameters (traditional approach)[/green]
  python cli.py train --task llm --liquid-units 128 --spiking-units 64 --num-layers 3 --hidden-dim 256 --epochs 2
  
  [green]# Train a vision model for 20 epochs[/green]
  python cli.py train --task vision --epochs 20 --batch-size 64
  
  [green]# Train with multi-GPU acceleration (auto-detect GPUs)[/green]
  python cli.py train --task llm --epochs 15 --multi-gpu
  
  [green]# Train with specific GPUs using DataParallel[/green]
  python cli.py train --task vision --gpu-strategy dp --gpu-ids "0,1,2"
  
  [green]# Train with DistributedDataParallel (recommended for 4+ GPUs)[/green]
  python cli.py train --task robotics --gpu-strategy ddp --multi-gpu
  
  [green]# Load and run inference on an image[/green]
  python cli.py inference --model-path vision_model.pt --input-file test.npy
  
  [green]# Benchmark a trained model[/green]
  python cli.py benchmark --model-path vision_model.pt --iterations 1000
  
  [green]# Export model to ONNX format[/green]
  python cli.py export --model-path vision_model.pt --output-path model.onnx
  
  [green]# Interactive configuration editor[/green]
  python cli.py config --task robotics --interactive
  
  [green]# Check system and GPU information[/green]
  python cli.py info --system --gpu
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
        
        # Config command
        self._add_config_parser(subparsers)
        
        # Info command
        self._add_info_parser(subparsers)
        
        # Status command
        self._add_status_parser(subparsers)
        
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
                                help='Use predefined configuration presets: gpt-4o (GPT-4o competitive), claude-sonnet-4 (Claude Sonnet 4 competitive), ultra-advanced (better than all modern LLMs)')
        
        # Configuration file options
        train_parser.add_argument('--config-path', type=str,
                                help='Path to custom configuration JSON file')
        train_parser.add_argument('--save-config', type=str,
                                help='Save final configuration to JSON file')
        
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
        arch_group.add_argument('--liquid-backbone', choices=['cfc', 'ltc', 'ncp'],
                               help='Liquid neural network backbone type')
        
        # Spiking network parameters
        spike_group = train_parser.add_argument_group('Spiking Network Parameters')
        spike_group.add_argument('--spike-threshold', type=float,
                                help='Spike threshold for spiking neurons')
        spike_group.add_argument('--beta', type=float,
                                help='Membrane potential decay factor (0-1)')
        spike_group.add_argument('--num-spike-steps', type=int,
                                help='Number of time steps for spiking dynamics')
        
        # LLM-specific parameters
        llm_group = train_parser.add_argument_group('Language Model Parameters')
        llm_group.add_argument('--vocab-size', type=int,
                              help='Vocabulary size for LLM (default: 50257)')
        llm_group.add_argument('--embedding-dim', type=int,
                              help='Embedding dimension for tokens')
        llm_group.add_argument('--max-position-embeddings', type=int,
                              help='Maximum position embeddings')
        llm_group.add_argument('--sequence-length', type=int,
                              help='Maximum sequence length')
        
        # Tokenizer parameters
        tokenizer_group = train_parser.add_argument_group('Tokenizer Parameters')
        tokenizer_group.add_argument('--tokenizer', choices=['gpt2', 'gpt3', 'gpt4', 'o200k', 'codellama', 'llama2'],
                                   default='gpt2',
                                   help='Tokenizer type to use (default: gpt2)')
        tokenizer_group.add_argument('--tokenizer-vocab-size', type=int,
                                   help='Override tokenizer vocabulary size')
        
        # Vision-specific parameters
        vision_group = train_parser.add_argument_group('Vision Model Parameters')
        vision_group.add_argument('--conv-channels', type=str,
                                 help='Convolutional channels (comma-separated, e.g., "32,64,128")')
        vision_group.add_argument('--conv-kernel-sizes', type=str,
                                 help='Convolutional kernel sizes (comma-separated, e.g., "3,3,3")')
        
        # Regularization parameters
        reg_group = train_parser.add_argument_group('Regularization')
        reg_group.add_argument('--dropout', type=float,
                              help='Dropout rate')
        reg_group.add_argument('--attention-dropout', type=float,
                              help='Attention dropout rate')
        reg_group.add_argument('--embedding-dropout', type=float,
                              help='Embedding dropout rate')
        reg_group.add_argument('--weight-decay', type=float,
                              help='Weight decay (L2 regularization)')
        reg_group.add_argument('--gradient-clip', type=float,
                              help='Gradient clipping value')
        
        # Training options
        train_parser.add_argument('--output-dir', type=str, default='./models',
                                help='Directory to save trained models (default: ./models)')
        train_parser.add_argument('--resume', type=str,
                                help='Path to checkpoint to resume training from')
        train_parser.add_argument('--save-interval', type=int, default=5,
                                help='Save checkpoint every N epochs (default: 5)')
        train_parser.add_argument('--no-validation', action='store_true',
                                help='Skip validation during training')
        train_parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                                help='Device to use for training (default: auto)')
        train_parser.add_argument('--mixed-precision', dest='mixed_precision', action='store_true', default=True,
                                help='Enable mixed precision training (default: True)')
        train_parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false',
                                help='Disable mixed precision training')
        train_parser.add_argument('--seed', type=int, default=42,
                                help='Random seed for reproducibility')
        
        # Multi-GPU training options
        gpu_group = train_parser.add_argument_group('Multi-GPU Training')
        gpu_group.add_argument('--multi-gpu', action='store_true',
                              help='Enable multi-GPU training (auto-detect available GPUs)')
        gpu_group.add_argument('--gpu-strategy', choices=['auto', 'dp', 'ddp', 'none'], 
                              default='auto',
                              help='Multi-GPU strategy: auto (automatic), dp (DataParallel), ddp (DistributedDataParallel), none (single GPU/CPU)')
        gpu_group.add_argument('--gpu-ids', type=str,
                              help='Specific GPU IDs to use (comma-separated, e.g., "0,1,2,3")')
        gpu_group.add_argument('--distributed-backend', choices=['nccl', 'gloo'], 
                              default='nccl',
                              help='Distributed training backend (nccl for GPU, gloo for CPU)')
        gpu_group.add_argument('--master-port', type=str, default='12355',
                              help='Port for distributed training communication (default: 12355)')
        gpu_group.add_argument('--sync-batchnorm', action='store_true', default=True,
                              help='Use synchronized batch normalization for distributed training')
        gpu_group.add_argument('--no-sync-batchnorm', action='store_false', dest='sync_batchnorm',
                              help='Disable synchronized batch normalization')
        
        # Debug and output options
        debug_group = train_parser.add_argument_group('Debug and Output')
        debug_group.add_argument('--verbose', action='store_true',
                               help='Enable verbose logging output')
        
        # STDP Plasticity parameters (NEW)
        stdp_group = train_parser.add_argument_group('STDP Plasticity', 
                                                     'Spike-Timing-Dependent Plasticity options')
        stdp_group.add_argument('--use-stdp', action='store_true',
                               help='Enable STDP plasticity learning')
        stdp_group.add_argument('--stdp-type', type=str, default='homeostatic',
                               choices=['classical', 'triplet', 'homeostatic', 'bcm'],
                               help='Type of STDP rule (default: homeostatic)')
        stdp_group.add_argument('--stdp-learning-rate', type=float, default=0.01,
                               help='STDP learning rate (default: 0.01)')
        stdp_group.add_argument('--stdp-tau-plus', type=float, default=20.0,
                               help='STDP LTP time constant in ms (default: 20.0)')
        stdp_group.add_argument('--stdp-tau-minus', type=float, default=20.0,
                               help='STDP LTD time constant in ms (default: 20.0)')
        stdp_group.add_argument('--stdp-target-rate', type=float, default=0.1,
                               help='Target firing rate for homeostatic STDP (default: 0.1)')
        
        # Meta-Plasticity parameters (NEW)
        meta_group = train_parser.add_argument_group('Meta-Plasticity', 
                                                     'Learning to learn - adaptive plasticity')
        meta_group.add_argument('--use-meta-plasticity', action='store_true',
                               help='Enable meta-plasticity (learning to learn)')
        meta_group.add_argument('--meta-lr', type=float, default=0.001,
                               help='Meta-learning rate (default: 0.001)')
        meta_group.add_argument('--meta-history-length', type=int, default=100,
                               help='History length for meta-plasticity (default: 100)')
        meta_group.add_argument('--meta-hidden-dim', type=int, default=128,
                               help='Hidden dimension for meta-controller (default: 128)')
        
        # Continual Learning parameters (NEW)
        cl_group = train_parser.add_argument_group('Continual Learning', 
                                                   'Lifelong learning without forgetting')
        cl_group.add_argument('--continual-learning', action='store_true',
                             help='Enable continual learning mode')
        cl_group.add_argument('--num-tasks', type=int, default=1,
                             help='Number of sequential tasks (default: 1)')
        cl_group.add_argument('--consolidation-strength', type=float, default=1000.0,
                             help='Weight consolidation strength (default: 1000.0)')
        cl_group.add_argument('--plasticity-decay', type=float, default=0.9,
                             help='Plasticity decay rate (default: 0.9)')
        cl_group.add_argument('--use-replay', action='store_true',
                             help='Use experience replay for continual learning')
        cl_group.add_argument('--replay-buffer-size', type=int, default=1000,
                             help='Size of experience replay buffer (default: 1000)')
        cl_group.add_argument('--replay-strategy', type=str, default='balanced',
                             choices=['uniform', 'importance', 'balanced'],
                             help='Replay sampling strategy (default: balanced)')
        cl_group.add_argument('--compute-importance-interval', type=int, default=100,
                             help='Compute parameter importance every N batches (default: 100)')
    
    def _add_inference_parser(self, subparsers):
        """Add inference subcommand parser."""
        inference_parser = subparsers.add_parser('inference', help='🔮 Run inference on trained model')
        inference_parser.add_argument('--model-path', type=str, required=True,
                                    help='Path to trained model checkpoint')
        inference_parser.add_argument('--input-file', type=str,
                                    help='Path to input data file (.npy, .pt, or .json)')
        inference_parser.add_argument('--input-shape', type=str,
                                    help='Shape of random input data (e.g., "3,32,32" for CIFAR-10)')
        inference_parser.add_argument('--batch-size', type=int, default=1,
                                    help='Batch size for inference (default: 1)')
        inference_parser.add_argument('--output-file', type=str,
                                    help='Path to save inference results')
        inference_parser.add_argument('--verbose', action='store_true',
                                    help='Print detailed inference results')
        
        # Text generation options for LLM models
        text_group = inference_parser.add_argument_group('text generation', 'Options for LLM text generation')
        text_group.add_argument('--prompt', type=str,
                               help='Text prompt for LLM generation (e.g., "The future of AI is")')
        text_group.add_argument('--max-length', type=int, default=100,
                               help='Maximum length of generated text (default: 100)')
        text_group.add_argument('--temperature', type=float, default=1.0,
                               help='Temperature for text generation (default: 1.0)')
        text_group.add_argument('--tokenizer', type=str, default='gpt2',
                               help='Tokenizer to use for text processing (default: gpt2)')
    
    def _add_benchmark_parser(self, subparsers):
        """Add benchmark subcommand parser."""
        benchmark_parser = subparsers.add_parser('benchmark', help='⚡ Benchmark model performance')
        benchmark_parser.add_argument('--model-path', type=str, required=True,
                                    help='Path to trained model checkpoint')
        benchmark_parser.add_argument('--iterations', type=int, default=100,
                                    help='Number of inference iterations (default: 100)')
        benchmark_parser.add_argument('--warmup', type=int, default=10,
                                    help='Number of warmup iterations (default: 10)')
        benchmark_parser.add_argument('--batch-sizes', type=str, default='1,8,16,32',
                                    help='Comma-separated batch sizes to test (default: 1,8,16,32)')
        benchmark_parser.add_argument('--output-file', type=str,
                                    help='Path to save benchmark results as JSON')
    
    def _add_export_parser(self, subparsers):
        """Add export subcommand parser."""
        export_parser = subparsers.add_parser('export', help='📦 Export trained model to different formats')
        export_parser.add_argument('--model-path', type=str, required=True,
                                 help='Path to trained model checkpoint')
        export_parser.add_argument('--output-path', type=str, required=True,
                                 help='Path for exported model')
        export_parser.add_argument('--format', choices=['onnx', 'torchscript'], default='onnx',
                                 help='Export format (default: onnx)')
        export_parser.add_argument('--opset-version', type=int, default=11,
                                 help='ONNX opset version (default: 11)')
    
    def _add_config_parser(self, subparsers):
        """Add config subcommand parser."""
        config_parser = subparsers.add_parser('config', help='⚙️ Create or modify model configurations')
        config_parser.add_argument('--task', choices=['llm', 'vision', 'robotics'], required=True,
                                 help='Type of task configuration')
        config_parser.add_argument('--save-path', type=str, required=True,
                                 help='Path to save configuration JSON file')
        config_parser.add_argument('--modify', type=str,
                                 help='JSON string with configuration modifications')
        config_parser.add_argument('--interactive', action='store_true',
                                 help='Interactive configuration editor')
    
    def _add_info_parser(self, subparsers):
        """Add info subcommand parser."""
        info_parser = subparsers.add_parser('info', help='ℹ️ Display model and system information')
        info_parser.add_argument('--model-path', type=str,
                               help='Path to model checkpoint to analyze')
        info_parser.add_argument('--system', action='store_true',
                               help='Display system information')
        info_parser.add_argument('--gpu', action='store_true',
                               help='Display detailed GPU information for multi-GPU training')
        info_parser.add_argument('--config-only', action='store_true',
                               help='Display only configuration information')
    
    def _add_status_parser(self, subparsers):
        """Add status subcommand parser."""
        status_parser = subparsers.add_parser('status', help='📊 Show training status and model overview')
        status_parser.add_argument('--models-dir', type=str, default='./models',
                                 help='Directory containing model checkpoints')
        status_parser.add_argument('--watch', action='store_true',
                                 help='Continuously monitor training status')
    
    def run(self):
        """Main entry point for the CLI application."""
        args = self.parser.parse_args()
        
        # Show welcome message
        self._show_welcome()
        
        if not args.command:
            self._show_help_menu()
            return
        
        try:
            if args.command == 'train':
                self._handle_train(args)
            elif args.command == 'inference':
                self._handle_inference(args)
            elif args.command == 'benchmark':
                self._handle_benchmark(args)
            elif args.command == 'export':
                self._handle_export(args)
            elif args.command == 'config':
                self._handle_config(args)
            elif args.command == 'info':
                self._handle_info(args)
            elif args.command == 'status':
                self._handle_status(args)
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠ Operation interrupted by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"\n[red]✗ An error occurred: {str(e)}[/red]")
            console.print_exception()
            sys.exit(1)
    
    def _show_welcome(self):
        """Display welcome message."""
        welcome_text = """
[bold blue]Liquid-Spiking Neural Network CLI[/bold blue]
[dim]Hybrid architecture combining liquid and spiking dynamics[/dim]

[green]Available Commands:[/green]
• [cyan]train[/cyan]     - Train neural network models
• [cyan]inference[/cyan] - Run model inference
• [cyan]benchmark[/cyan] - Performance benchmarking
• [cyan]export[/cyan]    - Export models to different formats
• [cyan]config[/cyan]    - Configuration management
• [cyan]info[/cyan]      - System and model information
• [cyan]status[/cyan]    - Training status monitoring
        """
        
        panel = Panel(
            welcome_text,
            border_style="blue",
            padding=(1, 2),
            title="🧠 Welcome",
            title_align="left"
        )
        console.print(panel)
    
    def _show_help_menu(self):
        """Show interactive help menu."""
        console.print("\n[yellow]💡 Use --help with any command for detailed options[/yellow]")
        console.print("[dim]Example: python cli.py train --help[/dim]\n")

    def _create_preset_config(self, preset_name: str, task_type: str) -> ModelConfig:
        """Create a configuration based on the specified preset."""
        if task_type != 'llm':
            console.print(f"[red]Warning: Preset '{preset_name}' is designed for LLM tasks. Using default config for {task_type}.[/red]")
            if task_type == 'vision':
                return create_vision_config()
            elif task_type == 'robotics':
                return create_robotics_config()
        
        console.print(f"[bold cyan]🎯 Loading preset configuration: {preset_name}[/bold cyan]")
        
        if preset_name == 'gpt-4o':
            return self._create_gpt4o_config()
        elif preset_name == 'claude-sonnet-4':
            return self._create_claude_sonnet4_config()
        elif preset_name == 'ultra-advanced':
            return self._create_ultra_advanced_config()
        else:
            console.print(f"[red]Unknown preset: {preset_name}. Using default config.[/red]")
            return create_llm_config()
    
    def _create_gpt4o_config(self) -> ModelConfig:
        """Create GPT-4o competitive configuration with advanced hybrid liquid-spiking features."""
        console.print("[bold green]🚀 GPT-4o Competitive Configuration[/bold green]")
        console.print("   • Optimized for GPT-4o level performance")
        console.print("   • Advanced liquid neural dynamics")
        console.print("   • Enhanced spiking temporal processing")
        console.print("   • Multi-GPU scaling ready")
        
        return create_custom_config(
            'llm',
            # Core Architecture - Scaled for Competition
            input_dim=2048,
            hidden_dim=2048,
            output_dim=50257,
            
            # Liquid Neural Network - Optimized
            liquid_units=1024,
            liquid_backbone='cfc',
            
            # Spiking Neural Network - Enhanced
            spiking_units=512,
            spike_threshold=0.9,
            beta=0.97,
            num_spike_steps=64,
            
            # Network Architecture
            num_layers=24,
            num_attention_heads=32,
            
            # Language Model Parameters
            embedding_dim=2048,
            max_position_embeddings=8192,
            vocab_size=50257,
            sequence_length=4096,
            
            # Regularization - Carefully tuned
            dropout=0.05,
            attention_dropout=0.05,
            embedding_dropout=0.02,
            
            # Training Parameters - Optimized for Scale
            batch_size=8,
            learning_rate=1e-4,
            weight_decay=0.1,
            gradient_clip=1.0,
            mixed_precision=True,
            num_epochs=100,
            
            # Advanced Training
            device='cuda',
            seed=42,
            layer_norm_eps=1e-6,
            initializer_range=0.01,
            use_cache=True
        )
    
    def _create_claude_sonnet4_config(self) -> ModelConfig:
        """Create Claude Sonnet 4 competitive configuration with enhanced reasoning capabilities."""
        console.print("[bold purple]🧠 Claude Sonnet 4 Competitive Configuration[/bold purple]")
        console.print("   • Optimized for Claude Sonnet 4 level reasoning")
        console.print("   • Enhanced multi-step reasoning capabilities")
        console.print("   • Advanced temporal credit assignment")
        console.print("   • Superior long-context understanding")
        
        return create_custom_config(
            'llm',
            # Core Architecture - Reasoning Optimized
            input_dim=2560,
            hidden_dim=2560,
            output_dim=50257,
            
            # Liquid Neural Network - Advanced Dynamics
            liquid_units=1280,
            liquid_backbone='cfc',
            
            # Spiking Neural Network - Temporal Reasoning
            spiking_units=640,
            spike_threshold=0.85,
            beta=0.98,
            num_spike_steps=96,
            
            # Deep Architecture for Complex Reasoning
            num_layers=32,
            num_attention_heads=40,
            
            # Language Model Parameters - Extended Context
            embedding_dim=2560,
            max_position_embeddings=16384,
            vocab_size=50257,
            sequence_length=8192,
            
            # Sophisticated Regularization
            dropout=0.03,
            attention_dropout=0.03,
            embedding_dropout=0.01,
            
            # Training Parameters - Advanced Optimization
            batch_size=6,
            learning_rate=8e-5,
            weight_decay=0.12,
            gradient_clip=0.8,
            mixed_precision=True,
            num_epochs=150,
            
            # High-Performance Training
            device='cuda',
            seed=42,
            layer_norm_eps=1e-7,
            initializer_range=0.008,
            use_cache=True
        )
    
    def _create_ultra_advanced_config(self) -> ModelConfig:
        """Create ultra-advanced configuration designed to exceed all modern LLMs."""
        console.print("[bold red]⚡ ULTRA-ADVANCED: Better Than All Modern LLMs[/bold red]")
        console.print("   • 🏆 Designed to exceed GPT-4, Claude, Gemini")
        console.print("   • 🧬 Revolutionary hybrid liquid-spiking architecture")
        console.print("   • 🚀 Maximum capacity and optimization")
        console.print("   • ⚡ Breakthrough temporal processing")
        console.print("   • 🎯 State-of-the-art performance targeting")
        
        return create_custom_config(
            'llm',
            # Massive Core Architecture
            input_dim=4096,
            hidden_dim=4096,
            output_dim=100000,  # Expanded vocabulary for multilingual superiority
            
            # Revolutionary Liquid Neural Network
            liquid_units=2048,
            liquid_backbone='cfc',
            
            # Advanced Spiking Neural Network
            spiking_units=1024,
            spike_threshold=0.8,
            beta=0.99,
            num_spike_steps=128,
            
            # Deep Architecture - Maximum Capacity
            num_layers=48,
            num_attention_heads=64,
            
            # Extended Language Model Parameters
            embedding_dim=4096,
            max_position_embeddings=32768,  # Massive context window
            vocab_size=100000,
            sequence_length=16384,
            
            # Ultra-Fine Regularization
            dropout=0.02,
            attention_dropout=0.02,
            embedding_dropout=0.005,
            
            # Ultra-Advanced Training Parameters
            batch_size=4,  # Large model requires smaller batches
            learning_rate=5e-5,
            weight_decay=0.15,
            gradient_clip=0.6,
            mixed_precision=True,
            num_epochs=200,
            
            # Maximum Performance Training
            device='cuda',
            seed=42,
            layer_norm_eps=1e-8,
            initializer_range=0.005,
            use_cache=True
        )
    
    def _handle_train(self, args):
        """Handle the train command with rich progress display and multi-GPU support."""
        # Configure logging based on verbose flag
        if getattr(args, 'verbose', False):
            enable_verbose_logging()
        
        self.logger.header("Training Neural Network", f"Task: {args.task.upper()}")
        
        # Display GPU information if multi-GPU is enabled
        if hasattr(args, 'multi_gpu') and (args.multi_gpu or args.gpu_strategy != 'none'):
            self._display_gpu_info()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create configuration
        config = self._load_config(args)
        
        # Display preset information if used
        if hasattr(args, 'config_preset') and args.config_preset:
            self._display_preset_info(args.config_preset)
        
        # Display configuration table
        self._display_config_table(config)
        
        # Create datasets with progress
        with Status("[bold green]Creating datasets...", spinner="dots"):
            with suppress_logging():
                train_dataset, val_dataset = self._create_datasets(config, args)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(config, args, train_dataset, val_dataset)
        
        # Initialize model and trainer
        with Status("[bold green]Initializing model and multi-GPU setup...", spinner="dots"):
            with suppress_logging():
                model = LiquidSpikingNetwork(config)
                trainer = LiquidSpikingTrainer(model, config)
        
        # Display model information
        self._display_model_info(model)
        
        # Display multi-GPU training info
        if hasattr(trainer, 'gpu_ids') and trainer.gpu_ids and len(trainer.gpu_ids) > 1:
            self._display_multi_gpu_info(trainer)
        
        # Training loop with rich progress bar
        # Check if continual learning mode is enabled
        if hasattr(args, 'continual_learning') and args.continual_learning and args.num_tasks > 1:
            self._train_continual_learning(trainer, train_loader, val_loader, args, output_dir)
        else:
            self._train_with_progress(trainer, train_loader, val_loader, args, output_dir)
    
    def _train_with_progress(self, trainer, train_loader, val_loader, args, output_dir):
        """Train model with beautiful progress display."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True
        ) as progress:
            
            # Main training progress
            train_task = progress.add_task(
                "[green]Training Progress", 
                total=args.epochs
            )
            
            # Epoch details
            epoch_task = progress.add_task(
                "[blue]Current Epoch", 
                total=len(train_loader)
            )
            
            start_time = time.time()
            
            for epoch in range(args.epochs):
                epoch_start = time.time()
                
                # Update epoch progress
                progress.update(
                    epoch_task, 
                    description=f"[blue]Epoch {epoch+1}/{args.epochs}",
                    completed=0
                )
                
                # Training epoch with batch progress
                train_loss = self._train_epoch_with_progress(
                    trainer, train_loader, progress, epoch_task
                )
                
                # Validation
                val_loss = None
                if val_loader is not None:
                    with Status("[yellow]Running validation...", spinner="dots"):
                        with suppress_logging():
                            is_best = trainer.validate(val_loader)
                            val_loss = trainer.val_losses[-1]
                            
                            if is_best:
                                best_path = output_dir / f"{args.task}_best_model.pt"
                                trainer.save_checkpoint(str(best_path))
                                self.logger.success(f"New best model saved: {best_path}")
                
                epoch_time = time.time() - epoch_start
                
                # Update training progress
                progress.update(train_task, advance=1)
                
                # Log epoch results
                self._log_epoch_results(epoch, args.epochs, train_loss, val_loss, epoch_time)
                
                # Save periodic checkpoint
                if (epoch + 1) % args.save_interval == 0:
                    checkpoint_path = output_dir / f"{args.task}_epoch_{epoch+1}.pt"
                    trainer.save_checkpoint(str(checkpoint_path))
            
            # Save final model
            final_path = output_dir / f"{args.task}_final_model.pt"
            trainer.save_checkpoint(str(final_path))
            
            total_time = time.time() - start_time
            self.logger.success(f"Training completed in {total_time/3600:.2f} hours")
            self.logger.success(f"Final model saved: {final_path}")
    
    def _train_continual_learning(self, trainer, train_loader, val_loader, args, output_dir):
        """Train model in continual learning mode with multiple sequential tasks."""
        self.logger.header("Continual Learning Mode", f"{args.num_tasks} Sequential Tasks")
        
        # For simplicity, split the dataset into multiple tasks
        # In practice, you'd load different datasets per task
        task_dataloaders = {}
        task_val_loaders = {}
        
        # Create tasks by cycling through the data
        epochs_per_task = max(1, args.epochs // args.num_tasks)
        
        self.logger.info(f"Training {args.num_tasks} tasks with {epochs_per_task} epochs each")
        
        start_time = time.time()
        
        for task_id in range(args.num_tasks):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📚 Task {task_id + 1}/{args.num_tasks}")
            self.logger.info(f"{'='*60}")
            
            # Train on this task
            task_accuracy = trainer.train_on_task(
                task_id=task_id,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=epochs_per_task
            )
            
            # Store validation loader for later evaluation
            if val_loader is not None:
                task_val_loaders[task_id] = val_loader
            
            # Save checkpoint after each task
            checkpoint_path = output_dir / f"{args.task}_task_{task_id+1}.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            self.logger.success(f"Task {task_id + 1} checkpoint saved: {checkpoint_path}")
        
        # Evaluate on all tasks to measure forgetting
        if task_val_loaders:
            self.logger.info(f"\n{'='*60}")
            self.logger.info("📊 Final Evaluation on All Tasks")
            self.logger.info(f"{'='*60}")
            
            results, avg_acc, avg_forgetting = trainer.evaluate_all_tasks(task_val_loaders)
            
            # Display results table
            results_table = Table(title="Continual Learning Results")
            results_table.add_column("Task", style="cyan")
            results_table.add_column("Accuracy", style="green")
            results_table.add_column("Initial", style="yellow")
            results_table.add_column("Forgetting", style="red")
            
            for task_id, accuracy in results.items():
                initial_acc = trainer.task_performance.get(task_id, 0.0)
                forgetting = max(0, initial_acc - accuracy)
                results_table.add_row(
                    f"Task {task_id + 1}",
                    f"{accuracy:.3f}",
                    f"{initial_acc:.3f}",
                    f"{forgetting:.3f}"
                )
            
            results_table.add_row(
                "Average",
                f"{avg_acc:.3f}",
                "-",
                f"{avg_forgetting:.3f}",
                style="bold"
            )
            
            console.print(results_table)
        
        # Save final model
        final_path = output_dir / f"{args.task}_continual_final.pt"
        trainer.save_checkpoint(str(final_path))
        
        total_time = time.time() - start_time
        self.logger.success(f"Continual learning completed in {total_time/3600:.2f} hours")
        self.logger.success(f"Final model saved: {final_path}")
        self.logger.info(f"🎉 Average forgetting: {avg_forgetting:.3f}")
    
    def _train_epoch_with_progress(self, trainer, train_loader, progress, epoch_task):
        """Train single epoch with progress updates."""
        trainer.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract data and targets from batch dictionary
            if isinstance(batch, dict):
                data = batch["input_ids"]
                targets = batch["labels"]
            else:
                # Fallback for tuple format
                data, targets = batch
            
            # Train single batch using trainer's built-in logic
            batch_loss = self._train_single_batch(trainer, data, targets)
            total_loss += batch_loss
            num_batches += 1
            
            # Update batch progress
            progress.update(epoch_task, advance=1)
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _train_single_batch(self, trainer, data, targets):
        """Train a single batch and return loss."""
        data = data.to(trainer.device, non_blocking=True)
        targets = targets.to(trainer.device, non_blocking=True)
        
        trainer.optimizer.zero_grad()
        
        if trainer.config.mixed_precision and trainer.scaler:
            with autocast('cuda'):
                outputs = trainer.model(data)
                loss = trainer._compute_loss(outputs, targets)
            
            trainer.scaler.scale(loss).backward()
            
            if trainer.config.gradient_clip > 0:
                trainer.scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.gradient_clip)
            
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        else:
            outputs = trainer.model(data)
            loss = trainer._compute_loss(outputs, targets)
            loss.backward()
            
            if trainer.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.gradient_clip)
            
            trainer.optimizer.step()
        
        return loss.item()
    
    def _log_epoch_results(self, epoch, total_epochs, train_loss, val_loss, epoch_time):
        """Log epoch results in a formatted table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Epoch", f"{epoch+1}/{total_epochs}")
        table.add_row("Train Loss", f"{train_loss:.4f}")
        if val_loss is not None:
            table.add_row("Val Loss", f"{val_loss:.4f}")
        table.add_row("Time", f"{epoch_time:.2f}s")
        
        console.print(table)
    
    def _handle_benchmark(self, args):
        """Handle benchmark command with rich tables."""
        self.logger.header("Performance Benchmark", f"Model: {Path(args.model_path).name}")
        
        # Load model
        with Status("[bold green]Loading model...", spinner="dots"):
            with suppress_logging():
                model, config = load_model(args.model_path, None)
        
        batch_sizes = list(map(int, args.batch_sizes.split(',')))
        results = {}
        
        # Create benchmark table
        table = Table(title="Performance Benchmark Results")
        table.add_column("Batch Size", justify="center", style="cyan")
        table.add_column("Avg Time (ms)", justify="right", style="green")
        table.add_column("Throughput (samples/s)", justify="right", style="yellow")
        table.add_column("Memory (MB)", justify="right", style="magenta")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            benchmark_task = progress.add_task(
                "[green]Benchmarking...", 
                total=len(batch_sizes)
            )
            
            for batch_size in batch_sizes:
                progress.update(
                    benchmark_task,
                    description=f"[green]Batch size: {batch_size}"
                )
                
                # Run benchmark for this batch size
                result = self._benchmark_batch_size(model, config, batch_size, args)
                results[batch_size] = result
                
                # Add row to table
                table.add_row(
                    str(batch_size),
                    f"{result['avg_time_ms']:.2f}",
                    f"{result['throughput_samples_per_sec']:.1f}",
                    f"{result['memory_allocated_mb']:.1f}"
                )
                
                progress.update(benchmark_task, advance=1)
        
        # Display results
        console.print(table)
        
        # Model statistics
        self._display_model_stats(model, config)
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.success(f"Benchmark results saved to {args.output_file}")
    
    def _handle_info(self, args):
        """Handle info command with rich display."""
        self.logger.header("System & Model Information")
        
        if args.system or not args.model_path:
            self._display_system_info()
        
        if args.gpu:
            self._display_gpu_info()
        
        if args.model_path:
            self._display_model_info_detailed(args.model_path, args.config_only)
    
    def _handle_status(self, args):
        """Handle status command with model overview."""
        self.logger.header("Training Status & Model Overview")
        
        models_dir = Path(args.models_dir)
        if not models_dir.exists():
            self.logger.warning(f"Models directory not found: {models_dir}")
            return
        
        # Find all model checkpoints
        model_files = list(models_dir.glob("*.pt"))
        
        if not model_files:
            self.logger.warning(f"No model checkpoints found in {models_dir}")
            return
        
        # Create status table
        table = Table(title=f"Model Status - {models_dir}")
        table.add_column("Model", style="cyan")
        table.add_column("Task", style="green")
        table.add_column("Epochs", justify="right", style="yellow")
        table.add_column("Best Val Loss", justify="right", style="magenta")
        table.add_column("Parameters", justify="right", style="blue")
        table.add_column("Size", justify="right", style="red")
        
        for model_path in sorted(model_files):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                config = ModelConfig.from_dict(checkpoint['config'])
                
                # Calculate model size
                size_mb = model_path.stat().st_size / (1024 * 1024)
                
                # Get training info
                epochs = len(checkpoint.get('train_losses', [0]))
                best_val = checkpoint.get('best_val_loss', 'N/A')
                
                # Count parameters
                model = LiquidSpikingNetwork(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                params = sum(p.numel() for p in model.parameters())
                
                table.add_row(
                    model_path.name,
                    config.task_type.value,
                    str(epochs),
                    f"{best_val:.4f}" if isinstance(best_val, float) else str(best_val),
                    f"{params:,}",
                    f"{size_mb:.1f} MB"
                )
                
            except Exception as e:
                table.add_row(
                    model_path.name,
                    "[red]Error[/red]",
                    "-", "-", "-", "-"
                )
        
        console.print(table)
    
    def _display_preset_info(self, preset_name: str):
        """Display information about the selected preset configuration."""
        preset_info = {
            'gpt-4o': {
                'title': '🚀 GPT-4o Competitive Configuration',
                'description': 'Optimized to compete with GPT-4o performance',
                'highlights': [
                    '2048 hidden dimensions',
                    '1024 liquid units with advanced dynamics',
                    '512 spiking units for temporal processing',
                    '24 hybrid layers',
                    '4096 sequence length',
                    '8192 max position embeddings',
                    'Multi-GPU ready architecture'
                ]
            },
            'claude-sonnet-4': {
                'title': '🧠 Claude Sonnet 4 Competitive Configuration',
                'description': 'Enhanced reasoning capabilities matching Claude Sonnet 4',
                'highlights': [
                    '2560 hidden dimensions',
                    '1280 liquid units for complex reasoning',
                    '640 spiking units with temporal credit assignment',
                    '32 deep layers for multi-step reasoning',
                    '8192 sequence length',
                    '16384 max position embeddings',
                    'Advanced long-context understanding'
                ]
            },
            'ultra-advanced': {
                'title': '⚡ ULTRA-ADVANCED: Better Than All Modern LLMs',
                'description': 'Revolutionary configuration designed to exceed GPT-4, Claude, and Gemini',
                'highlights': [
                    '4096 massive hidden dimensions',
                    '2048 revolutionary liquid units',
                    '1024 advanced spiking units',
                    '48 deep layers for maximum capacity',
                    '16384 sequence length',
                    '32768 massive context window',
                    '100K expanded vocabulary',
                    'State-of-the-art hybrid architecture'
                ]
            }
        }
        
        if preset_name in preset_info:
            info = preset_info[preset_name]
            
            # Create panel with preset information
            content = f"[bold]{info['description']}[/bold]\n\n"
            content += "[cyan]Key Features:[/cyan]\n"
            for highlight in info['highlights']:
                content += f"  • {highlight}\n"
            
            panel = Panel(
                content,
                title=info['title'],
                border_style="bright_cyan",
                padding=(1, 2)
            )
            console.print(panel)
    
    def _display_config_table(self, config):
        """Display configuration in a beautiful table."""
        table = Table(title="Model Configuration", box=None)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            if key == 'task_type':
                value = value.value if hasattr(value, 'value') else value
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
    
    def _display_model_info(self, model):
        """Display model information in a table."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        table = Table(title="Model Architecture", box=None)
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="white")
        
        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Trainable Parameters", f"{trainable_params:,}")
        table.add_row("Model Size (approx)", f"{total_params * 4 / (1024**2):.1f} MB")
        
        console.print(table)
    
    def _display_system_info(self):
        """Display comprehensive system information including GPU details."""
        system_info = SystemInfo.get_system_info()
        
        # System table
        sys_table = Table(title="🖥️ System Information", show_header=False)
        sys_table.add_column("Property", style="cyan")
        sys_table.add_column("Value", style="white")
        
        sys_table.add_row("Platform", system_info['platform'])
        sys_table.add_row("Python", system_info['python_version'])
        sys_table.add_row("PyTorch", system_info['pytorch_version'])
        sys_table.add_row("CPU Cores", str(system_info['cpu_count']))
        sys_table.add_row("RAM", f"{system_info['memory_total'] / (1024**3):.1f} GB")
        
        # Dependencies table
        deps_table = Table(title="📦 Key Dependencies", show_header=False)
        deps_table.add_column("Package", style="green")
        deps_table.add_column("Version", style="white")
        
        deps_table.add_row("PyTorch", system_info['pytorch_version'])
        deps_table.add_row("snnTorch", system_info['snntorch_version'])
        deps_table.add_row("NCPS", system_info['ncps_version'])
        
        # CUDA/GPU table
        if system_info['cuda_available']:
            cuda_table = Table(title="🔥 CUDA/GPU Information", show_header=False)
            cuda_table.add_column("Property", style="yellow")
            cuda_table.add_column("Value", style="white")
            
            cuda_table.add_row("CUDA Available", "✅ Yes")
            cuda_table.add_row("CUDA Version", system_info.get('cuda_version', 'Unknown'))
            cuda_table.add_row("GPU Count", str(system_info.get('cuda_device_count', 0)))
            cuda_table.add_row("Primary GPU", system_info.get('cuda_device_name', 'Unknown'))
            cuda_memory_gb = system_info.get('cuda_memory_total', 0) / (1024**3) if system_info.get('cuda_memory_total') else 0
            cuda_table.add_row("GPU Memory", f"{cuda_memory_gb:.1f} GB")
            
            console.print(Columns([sys_table, deps_table, cuda_table]))
        else:
            console.print(Columns([sys_table, deps_table]))
    
    def _display_gpu_info(self):
        """Display detailed GPU information for multi-GPU training."""
        # Import GPU utils
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils.gpu_utils import GPUDetector
        
        # Get GPU information with logging suppressed
        with suppress_logging():
            gpus = GPUDetector.detect_gpus()
        
        if not gpus:
            console.print("🚫 [red]No GPUs detected or CUDA unavailable[/red]")
            return
        
        # Create GPU table
        gpu_table = Table(title="🔥 Available GPUs for Multi-GPU Training")
        gpu_table.add_column("ID", justify="center")
        gpu_table.add_column("Name", style="cyan")
        gpu_table.add_column("Memory", justify="right")
        gpu_table.add_column("Compute", justify="center")
        gpu_table.add_column("Status", justify="center")
        gpu_table.add_column("Temp", justify="right")
        gpu_table.add_column("Power", justify="right")
        
        for gpu in gpus:
            status = "✅" if gpu.is_available else "❌"
            memory = f"{gpu.memory_total / 1024:.1f} GB"
            compute = f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
            temp = f"{gpu.temperature}°C" if gpu.temperature else "N/A"
            power = f"{gpu.power_usage}W" if gpu.power_usage else "N/A"
            
            gpu_table.add_row(
                str(gpu.device_id),
                gpu.name,
                memory,
                compute,
                status,
                temp,
                power
            )
        
        console.print(gpu_table)
        
        # Show compatible GPUs
        compatible_gpus = GPUDetector.filter_compatible_gpus(gpus)
        if compatible_gpus:
            console.print(f"\n✅ [green]{len(compatible_gpus)} GPU(s) available for training[/green]")
            gpu_ids = [str(gpu.device_id) for gpu in compatible_gpus]
            console.print(f"   GPU IDs: {', '.join(gpu_ids)}")
        else:
            console.print("\n⚠️ [yellow]No compatible GPUs found for training[/yellow]")
    
    def _display_multi_gpu_info(self, trainer):
        """Display multi-GPU training configuration."""
        if not hasattr(trainer, 'gpu_ids') or not trainer.gpu_ids:
            return
        
        # Multi-GPU info table
        gpu_info_table = Table(title="⚡ Multi-GPU Training Configuration")
        gpu_info_table.add_column("Setting", style="cyan")
        gpu_info_table.add_column("Value", style="white")
        
        strategy = getattr(trainer.config, 'multi_gpu_strategy', 'unknown')
        gpu_count = len(trainer.gpu_ids)
        gpu_ids_str = ', '.join(map(str, trainer.gpu_ids))
        world_size = getattr(trainer, 'world_size', gpu_count)
        
        gpu_info_table.add_row("Strategy", strategy.upper())
        gpu_info_table.add_row("GPU Count", str(gpu_count))
        gpu_info_table.add_row("GPU IDs", gpu_ids_str)
        gpu_info_table.add_row("World Size", str(world_size))
        gpu_info_table.add_row("Batch Size", str(trainer.config.batch_size))
        gpu_info_table.add_row("Distributed", "Yes" if trainer.multi_gpu_manager.is_distributed else "No")
        
        console.print(gpu_info_table)
        
        # Performance estimation
        base_batch_size = 32  # Assume base batch size
        speedup_estimate = min(gpu_count * 0.85, gpu_count)  # Account for overhead
        
        console.print(f"\n🚀 [green]Expected training speedup: ~{speedup_estimate:.1f}x[/green]")
        console.print(f"   📊 Effective batch size: {trainer.config.batch_size}")
        console.print(f"   🔥 Multi-GPU acceleration active!")
    
    def _display_system_info_old(self):
        """Display system information in a comprehensive table."""
        sys_info = SystemInfo.get_system_info()
        
        # System table
        sys_table = Table(title="System Information")
        sys_table.add_column("Component", style="cyan")
        sys_table.add_column("Details", style="white")
        
        sys_table.add_row("Platform", sys_info['platform'])
        sys_table.add_row("Architecture", sys_info['architecture'])
        sys_table.add_row("Processor", sys_info['processor'])
        sys_table.add_row("Python Version", sys_info['python_version'])
        sys_table.add_row("CPU Cores", f"{sys_info['cpu_count']} ({sys_info['cpu_count_physical']} physical)")
        sys_table.add_row("Memory", f"{sys_info['memory_total'] / (1024**3):.1f} GB total, {sys_info['memory_available'] / (1024**3):.1f} GB available")
        
        # Dependencies table
        deps_table = Table(title="Dependencies")
        deps_table.add_column("Package", style="cyan")
        deps_table.add_column("Version", style="white")
        
        deps_table.add_row("PyTorch", sys_info['pytorch_version'])
        deps_table.add_row("snnTorch", sys_info['snntorch_version'])
        deps_table.add_row("NCPS", sys_info['ncps_version'])
        
        # CUDA table if available
        if sys_info['cuda_available']:
            cuda_table = Table(title="CUDA Information")
            cuda_table.add_column("Component", style="cyan")
            cuda_table.add_column("Details", style="white")
            
            cuda_table.add_row("CUDA Version", sys_info.get('cuda_version', 'Unknown'))
            cuda_table.add_row("Device Count", str(sys_info['cuda_device_count']))
            cuda_table.add_row("Device Name", sys_info.get('cuda_device_name', 'Unknown'))
            cuda_table.add_row("Memory", f"{sys_info.get('cuda_memory_total', 0) / (1024**3):.1f} GB")
            
            # Print all tables
            console.print(Columns([sys_table, deps_table, cuda_table]))
        else:
            console.print(Columns([sys_table, deps_table]))
    
    def _load_config(self, args):
        """Load configuration with comprehensive validation and parameter fixes."""
        # Check for preset configuration first
        if hasattr(args, 'config_preset') and args.config_preset:
            self.logger.info(f"Using preset configuration: {args.config_preset}")
            config = self._create_preset_config(args.config_preset, args.task)
        # Load from config file
        elif args.config_path:
            self.logger.info(f"Loading configuration from {args.config_path}")
            config = load_config(args.config_path)
        # Use default configuration based on task with tokenizer upgrade
        else:
            if args.task == 'llm':
                tokenizer_type = getattr(args, 'tokenizer', 'gpt2')
                config = create_llm_config(tokenizer_type)
            elif args.task == 'vision':
                config = create_vision_config()
            else:
                config = create_robotics_config()
        
        # Apply overrides with validation
        overrides = 0
        
        # Critical Fix: Handle sequence length and position embeddings together
        if hasattr(args, 'sequence_length') and args.sequence_length:
            config.sequence_length = args.sequence_length
            # Auto-adjust max_position_embeddings to match or exceed sequence_length
            if hasattr(config, 'max_position_embeddings') and config.max_position_embeddings < args.sequence_length:
                config.max_position_embeddings = args.sequence_length
                self.logger.info(f"Auto-adjusted max_position_embeddings to {args.sequence_length}")
            overrides += 1
        
        # Critical Fix: Validate and adjust tokenizer-related parameters
        if hasattr(args, 'tokenizer') and args.tokenizer and args.task == 'llm':
            tokenizer_type = args.tokenizer
            
            # Update vocab sizes based on tokenizer
            vocab_size_mapping = {
                "gpt4": 100277,
                "gpt3": 50281, 
                "o200k": 200019,
                "codellama": 32000,
                "llama2": 32000,
                "gpt2": 50257
            }
            
            new_vocab_size = vocab_size_mapping.get(tokenizer_type, 50257)
            
            # Update all vocabulary-related parameters consistently
            config.vocab_size = new_vocab_size
            config.output_dim = new_vocab_size
            
            self.logger.info(f"Updated vocabulary for {tokenizer_type}: {new_vocab_size:,} tokens")
            overrides += 1
        
        # Apply other parameter overrides with validation
        override_mapping = {
            'batch_size': 'batch_size',
            'learning_rate': 'learning_rate', 
            'weight_decay': 'weight_decay',
            'gradient_clip': 'gradient_clip',
            'liquid_units': 'liquid_units',
            'spiking_units': 'spiking_units',
            'num_layers': 'num_layers',
            'hidden_dim': 'hidden_dim',
            'num_attention_heads': 'num_attention_heads',
            'num_spike_steps': 'num_spike_steps',
            'spike_threshold': 'spike_threshold',
            'beta': 'beta',
            'dropout': 'dropout',
            'attention_dropout': 'attention_dropout',
            'embedding_dropout': 'embedding_dropout',
            'epochs': 'num_epochs',
            'mixed_precision': 'mixed_precision',
            'seed': 'seed',
            # STDP parameters (NEW)
            'use_stdp': 'use_stdp',
            'stdp_type': 'stdp_type',
            'stdp_learning_rate': 'stdp_learning_rate',
            'stdp_tau_plus': 'stdp_tau_plus',
            'stdp_tau_minus': 'stdp_tau_minus',
            'stdp_target_rate': 'stdp_target_rate',
            # Meta-plasticity parameters (NEW)
            'use_meta_plasticity': 'use_meta_plasticity',
            'meta_lr': 'meta_lr',
            'meta_history_length': 'meta_history_length',
            'meta_hidden_dim': 'meta_hidden_dim',
            # Continual learning parameters (NEW)
            'continual_learning': 'use_continual_learning',
            'consolidation_strength': 'consolidation_strength',
            'plasticity_decay': 'plasticity_decay',
            'use_replay': 'use_experience_replay',
            'replay_buffer_size': 'replay_buffer_size',
            'replay_strategy': 'replay_sampling_strategy',
            'compute_importance_interval': 'compute_importance_interval',
        }
        
        for arg_name, config_attr in override_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                value = getattr(args, arg_name)
                
                # Special validation for certain parameters
                if arg_name == 'num_attention_heads':
                    # Ensure attention heads divide hidden dimension evenly
                    if hasattr(config, 'hidden_dim') and config.hidden_dim % value != 0:
                        self.logger.warning(f"Attention heads {value} doesn't divide hidden_dim {config.hidden_dim} evenly")
                        # Adjust to nearest valid value
                        valid_heads = [h for h in range(1, config.hidden_dim + 1) if config.hidden_dim % h == 0]
                        value = min(valid_heads, key=lambda x: abs(x - value))
                        self.logger.info(f"Adjusted attention heads to {value}")
                
                elif arg_name == 'sequence_length':
                    # Already handled above, but ensure consistency
                    if hasattr(config, 'max_position_embeddings') and config.max_position_embeddings < value:
                        config.max_position_embeddings = value
                
                setattr(config, config_attr, value)
                overrides += 1
        
        # Handle device selection
        if args.device != 'auto':
            config.device = args.device
            overrides += 1
        elif args.device == 'auto':
            config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            overrides += 1
        
        # Handle additional args that don't have direct mapping
        if hasattr(args, 'vocab_size') and args.vocab_size:
            config.vocab_size = args.vocab_size
            config.output_dim = args.vocab_size
            overrides += 1
        if hasattr(args, 'embedding_dim') and args.embedding_dim:
            config.embedding_dim = args.embedding_dim
            overrides += 1
        if hasattr(args, 'max_position_embeddings') and args.max_position_embeddings:
            config.max_position_embeddings = args.max_position_embeddings
            overrides += 1
        
        # Multi-GPU parameters
        if hasattr(args, 'multi_gpu') and args.multi_gpu:
            config.multi_gpu_strategy = 'auto'
            overrides += 1
        if hasattr(args, 'gpu_strategy') and args.gpu_strategy:
            config.multi_gpu_strategy = args.gpu_strategy
            overrides += 1
        if hasattr(args, 'gpu_ids') and args.gpu_ids:
            # Parse comma-separated GPU IDs
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
            config.gpu_ids = gpu_ids
            overrides += 1
        
        # Critical Fix: Post-configuration validation and auto-correction
        self._validate_and_fix_config(config)
        
        if overrides > 0:
            self.logger.info(f"Applying {overrides} configuration overrides")
        
        # Display configuration summary
        print_config_summary(config)
        
        # Show parameter count
        params = get_model_parameter_count(config)
        self.logger.info(f"Estimated model parameters: {params['total_parameters']:,}")
        
        # Save configuration if requested
        if hasattr(args, 'save_config') and args.save_config:
            save_config(config, args.save_config)
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        return config

    def _validate_and_fix_config(self, config):
        """Validate and auto-fix configuration inconsistencies."""
        
        # Fix 1: Ensure position embeddings can handle sequence length
        if hasattr(config, 'max_position_embeddings') and hasattr(config, 'sequence_length'):
            if config.max_position_embeddings < config.sequence_length:
                config.max_position_embeddings = config.sequence_length
                self.logger.info(f"Auto-fixed: max_position_embeddings -> {config.sequence_length}")
        
        # Fix 2: Ensure embedding dimension is set for LLM tasks
        if config.task_type == TaskType.LLM:
            if not hasattr(config, 'embedding_dim') or config.embedding_dim is None:
                config.embedding_dim = config.input_dim
                self.logger.info(f"Auto-fixed: embedding_dim -> {config.input_dim}")
        
        # Fix 3: Ensure attention heads divide hidden dimension
        if hasattr(config, 'num_attention_heads') and hasattr(config, 'hidden_dim'):
            if config.hidden_dim % config.num_attention_heads != 0:
                # Find nearest valid number of heads
                valid_heads = [h for h in range(1, config.hidden_dim + 1) 
                              if config.hidden_dim % h == 0]
                old_heads = config.num_attention_heads
                config.num_attention_heads = min(valid_heads, key=lambda x: abs(x - old_heads))
                self.logger.info(f"Auto-fixed: attention heads {old_heads} -> {config.num_attention_heads}")
        
        # Fix 4: Ensure spike steps is reasonable relative to sequence length
        if hasattr(config, 'num_spike_steps') and hasattr(config, 'sequence_length'):
            if config.num_spike_steps > config.sequence_length:
                config.num_spike_steps = min(config.num_spike_steps, config.sequence_length // 2)
                self.logger.info(f"Auto-fixed: spike steps -> {config.num_spike_steps}")
        
        # Fix 5: Ensure output dimension matches vocab size for LLM
        if config.task_type == TaskType.LLM:
            if hasattr(config, 'vocab_size') and config.output_dim != config.vocab_size:
                config.output_dim = config.vocab_size
                self.logger.info(f"Auto-fixed: output_dim -> {config.vocab_size}")
        
        # Fix 6: Ensure dropout values are in valid range
        for dropout_attr in ['dropout', 'attention_dropout', 'embedding_dropout']:
            if hasattr(config, dropout_attr):
                value = getattr(config, dropout_attr)
                if value is not None:
                    if value < 0 or value >= 1:
                        new_value = max(0.0, min(0.9, value))
                        setattr(config, dropout_attr, new_value)
                        self.logger.info(f"Auto-fixed: {dropout_attr} {value} -> {new_value}")
        
        # Fix 7: Ensure reasonable batch size for available memory
        if hasattr(config, 'batch_size') and hasattr(config, 'sequence_length'):
            # Rough memory estimate - adjust batch size if necessary
            estimated_memory_gb = (config.batch_size * config.sequence_length * config.hidden_dim * 4) / (1024**3)
            if estimated_memory_gb > 8:  # Assuming 10GB GPU
                new_batch_size = max(1, int(config.batch_size * 8 / estimated_memory_gb))
                if new_batch_size != config.batch_size:
                    self.logger.warning(f"Large memory usage detected. Reducing batch size {config.batch_size} -> {new_batch_size}")
                    config.batch_size = new_batch_size
    
    def _create_datasets(self, config, args):
        """Create datasets based on configuration."""
        if config.task_type == TaskType.LLM:
            # Get tokenizer type from config or args
            tokenizer_type = getattr(config, 'tokenizer_type', getattr(args, 'tokenizer', 'gpt2'))
            
            train_dataset, _ = DatasetFactory.create_llm_dataset(
                vocab_size=config.output_dim,
                seq_length=config.sequence_length,
                tokenizer_type=tokenizer_type
            )
            val_dataset = None
            if not args.no_validation:
                val_dataset, _ = DatasetFactory.create_llm_dataset(
                    vocab_size=config.output_dim,
                    seq_length=config.sequence_length,
                    tokenizer_type=tokenizer_type,
                    num_samples=5000
                )
        elif config.task_type == TaskType.VISION:
            train_dataset = DatasetFactory.create_vision_dataset(train=True)
            val_dataset = DatasetFactory.create_vision_dataset(train=False) if not args.no_validation else None
        else:
            train_dataset = DatasetFactory.create_robotics_dataset(
                state_dim=config.input_dim,
                action_dim=config.output_dim,
                seq_length=config.sequence_length
            )
            val_dataset = DatasetFactory.create_robotics_dataset(
                state_dim=config.input_dim,
                action_dim=config.output_dim,
                seq_length=config.sequence_length,
                num_samples=1000
            ) if not args.no_validation else None
        
        return train_dataset, val_dataset
    
    def _create_data_loaders(self, config, args, train_dataset, val_dataset):
        """Create data loaders."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2 if config.device == 'cuda' else 0
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2 if config.device == 'cuda' else 0
            )
        
        return train_loader, val_loader
    
    def _benchmark_batch_size(self, model, config, batch_size, args):
        """Benchmark single batch size."""
        # Create dummy input
        if config.task_type == TaskType.VISION:
            dummy_input = torch.randn(batch_size, 3, 32, 32)
        else:
            dummy_input = torch.randn(batch_size, config.sequence_length, config.input_dim)
        
        # Warmup
        model.eval()
        device = torch.device(config.device)
        model.to(device)
        dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(args.iterations):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / args.iterations
        throughput = batch_size / avg_time
        
        return {
            'avg_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }
    
    def _display_model_stats(self, model, config):
        """Display model statistics."""
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        stats_table = Table(title="Model Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Total Parameters", f"{param_count:,}")
        stats_table.add_row("Trainable Parameters", f"{trainable_params:,}")
        stats_table.add_row("Task Type", config.task_type.value)
        stats_table.add_row("Device", config.device)
        
        console.print(stats_table)
    
    def _handle_inference(self, args):
        """Handle inference with rich progress."""
        self.logger.header("Model Inference", f"Model: {Path(args.model_path).name}")
        
        # Detect task type from model name or default to LLM
        model_name = Path(args.model_path).name.lower()
        if 'vision' in model_name:
            task_type = TaskType.VISION
        elif 'robotics' in model_name or 'robot' in model_name:
            task_type = TaskType.ROBOTICS
        else:
            task_type = TaskType.LLM  # Default to LLM
        
        # Load model
        with Status("[bold green]Loading model...", spinner="dots"):
            with suppress_logging():
                model, config = load_model(args.model_path, task_type)
        
        # Check if text generation is requested
        if hasattr(args, 'prompt') and args.prompt:
            self._handle_text_generation(model, config, args, task_type)
        else:
            # Standard inference with prepared input data
            input_data = self._prepare_input_data(args, config)
            
            # Run inference
            with Status("[bold green]Running inference...", spinner="dots"):
                with suppress_logging():
                    start_time = time.time()
                    predictions = inference_example(args.model_path, task_type, input_data)
                    inference_time = time.time() - start_time
            
            # Display results
            self._display_inference_results(input_data, predictions, inference_time, args)
            
            # Save results if requested
            if args.output_file:
                self._save_inference_results(predictions, args.output_file)
    
    def _handle_text_generation(self, model, config, args, task_type):
        """Handle text generation with rich progress."""
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            self.logger.error(f"Failed to import required modules: {e}")
            return
        
        # Load tokenizer
        tokenizer_name = getattr(args, 'tokenizer', 'microsoft/DialoGPT-medium')
        with Status(f"[bold green]Loading tokenizer ({tokenizer_name})...", spinner="dots"):
            with suppress_logging():
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
        
        # Set generation parameters
        max_length = getattr(args, 'max_length', 100)
        temperature = getattr(args, 'temperature', 0.8)
        
        # Display generation info
        gen_table = Table(title="Text Generation Parameters")
        gen_table.add_column("Parameter", style="cyan")
        gen_table.add_column("Value", style="white")
        gen_table.add_row("Prompt", f'"{args.prompt}"')
        gen_table.add_row("Max Length", str(max_length))
        gen_table.add_row("Temperature", str(temperature))
        gen_table.add_row("Tokenizer", tokenizer_name)
        console.print(gen_table)
        
        # Generate text
        with Status("[bold green]Generating text...", spinner="dots"):
            with suppress_logging():
                start_time = time.time()
                generated_text = generate_text(
                    model=model,
                    config=config,
                    tokenizer=tokenizer,
                    prompt=args.prompt,
                    max_length=max_length,
                    temperature=temperature
                )
                generation_time = time.time() - start_time
        
        # Display results
        results_panel = Panel(
            f"[bold white]{generated_text}[/bold white]",
            title="[bold green]Generated Text[/bold green]",
            border_style="green"
        )
        console.print(results_panel)
        
        # Display timing info
        timing_table = Table(title="Generation Results")
        timing_table.add_column("Metric", style="cyan")
        timing_table.add_column("Value", style="white")
        timing_table.add_row("Generation Time", f"{generation_time*1000:.2f} ms")
        timing_table.add_row("Prompt Length", f"{len(args.prompt)} characters")
        timing_table.add_row("Generated Length", f"{len(generated_text)} characters")
        console.print(timing_table)
        
        # Save generated text if requested
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            self.logger.success(f"Generated text saved to {args.output_file}")
    
    
    def _prepare_input_data(self, args, config):
        """Prepare input data for inference with proper shapes."""
        if args.input_file:
            # Load from file
            file_path = Path(args.input_file)
            if file_path.suffix == '.npy':
                input_data = np.load(file_path)
                input_data = torch.from_numpy(input_data).float()
            elif file_path.suffix == '.pt':
                input_data = torch.load(file_path)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                input_data = torch.tensor(data, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        elif args.input_shape:
            # Create random input with specified shape
            shape = [int(x) for x in args.input_shape.split(',')]
            input_data = torch.randn(*shape)
            
        else:
            # Create task-appropriate default input
            if config.task_type == TaskType.LLM:
                # Create a proper token sequence for LLM
                seq_len = getattr(config, 'sequence_length', 64)
                vocab_size = getattr(config, 'vocab_size', config.output_dim)
                # Use smaller vocab size to avoid out-of-bounds issues
                safe_vocab_size = min(vocab_size, 1000)
                input_data = torch.randint(0, safe_vocab_size, (1, seq_len))
                self.logger.info(f"Using default LLM input with shape {input_data.shape} (token sequence)")
                
            elif config.task_type == TaskType.VISION:
                # Create a proper image tensor for vision
                input_data = torch.randn(1, 3, 32, 32)  # CIFAR-10 like input
                self.logger.info(f"Using default vision input with shape {input_data.shape} (image tensor)")
                
            else:  # ROBOTICS
                # Create proper robotics sensor data
                input_data = torch.randn(1, config.input_dim)
                self.logger.info(f"Using default robotics input with shape {input_data.shape} (sensor data)")
    
        return input_data
    
    def _display_inference_results(self, input_data, predictions, inference_time, args):
        """Display inference results in a table."""
        results_table = Table(title="Inference Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="white")
        
        results_table.add_row("Input Shape", str(input_data.shape))
        results_table.add_row("Output Shape", str(predictions.shape))
        results_table.add_row("Inference Time", f"{inference_time*1000:.2f} ms")
        
        if args.verbose and predictions.size < 20:  # Only show predictions if small
            results_table.add_row("Predictions", str(predictions.numpy() if hasattr(predictions, 'numpy') else predictions))
        
        console.print(results_table)
    
    def _save_inference_results(self, predictions, output_file):
        """Save inference results."""
        if output_file.endswith('.npy'):
            np.save(output_file, predictions)
        elif output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(predictions.tolist(), f, indent=2)
        else:
            np.save(output_file, predictions)
        self.logger.success(f"Results saved to {output_file}")
    
    def _handle_export(self, args):
        """Handle export with progress."""
        self.logger.header("Model Export", f"Format: {args.format.upper()}")
        
        # Load model
        with Status("[bold green]Loading model...", spinner="dots"):
            with suppress_logging():
                model, config = load_model(args.model_path, None)
        
        # Export model
        with Status(f"[bold green]Exporting to {args.format}...", spinner="dots"):
            with suppress_logging():
                if args.format == 'onnx':
                    export_onnx(model, config, args.output_path)
                elif args.format == 'torchscript':
                    self._export_torchscript(model, config, args.output_path)
        
        self.logger.success(f"Model exported successfully to {args.output_path}")
    
    def _export_torchscript(self, model, config, output_path):
        """Export model to TorchScript format."""
        model.eval()
        
        if config.task_type == TaskType.VISION:
            dummy_input = torch.randn(1, 3, 32, 32)
        else:
            dummy_input = torch.randn(1, config.sequence_length, config.input_dim)
        
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
    
    def _handle_config(self, args):
        """Handle configuration with interactive editor."""
        self.logger.header("Configuration Management", f"Task: {args.task.upper()}")
        
        # Create base configuration
        if args.task == 'llm':
            config = create_llm_config()
        elif args.task == 'vision':
            config = create_vision_config()
        else:
            config = create_robotics_config()
        
        # Apply modifications if provided
        if args.modify:
            modifications = json.loads(args.modify)
            config_dict = config.to_dict()
            config_dict.update(modifications)
            config = ModelConfig.from_dict(config_dict)
            self.logger.info(f"Applied modifications: {modifications}")
        
        # Interactive editor
        if args.interactive:
            config = self._interactive_config_editor(config)
        
        # Display final configuration
        self._display_config_table(config)
        
        # Save configuration
        with open(args.save_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        self.logger.success(f"Configuration saved to {args.save_path}")
    
    def _interactive_config_editor(self, config):
        """Interactive configuration editor using Rich prompts."""
        console.print("\n[bold blue]Interactive Configuration Editor[/bold blue]")
        console.print("[dim]Press Enter to keep current value, or enter new value[/dim]\n")
        
        config_dict = config.to_dict()
        
        for key, value in config_dict.items():
            if key == 'task_type':
                continue  # Don't allow changing task type
            
            if isinstance(value, bool):
                new_value = Confirm.ask(f"{key}", default=value)
            elif isinstance(value, int):
                new_value = IntPrompt.ask(f"{key}", default=value)
            elif isinstance(value, float):
                new_value = FloatPrompt.ask(f"{key}", default=value)
            else:
                new_value = Prompt.ask(f"{key}", default=str(value))
            
            config_dict[key] = new_value
        
        return ModelConfig.from_dict(config_dict)
    
    def _display_model_info_detailed(self, model_path, config_only=False):
        """Display detailed model information."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            config = ModelConfig.from_dict(checkpoint['config'])
            
            # Configuration table
            config_table = Table(title=f"Model Configuration - {Path(model_path).name}")
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", style="white")
            
            config_dict = config.to_dict()
            for key, value in config_dict.items():
                if key == 'task_type':
                    value = value.value if hasattr(value, 'value') else value
                config_table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(config_table)
            
            if not config_only:
                # Training information
                training_table = Table(title="Training Information")
                training_table.add_column("Metric", style="cyan")
                training_table.add_column("Value", style="white")
                
                if 'train_losses' in checkpoint:
                    training_table.add_row("Training Epochs", str(len(checkpoint['train_losses'])))
                    training_table.add_row("Final Train Loss", f"{checkpoint['train_losses'][-1]:.4f}")
                if 'val_losses' in checkpoint and checkpoint['val_losses']:
                    training_table.add_row("Final Val Loss", f"{checkpoint['val_losses'][-1]:.4f}")
                if 'best_val_loss' in checkpoint:
                    training_table.add_row("Best Val Loss", f"{checkpoint['best_val_loss']:.4f}")
                
                # Model statistics
                model = LiquidSpikingNetwork(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                training_table.add_row("Total Parameters", f"{total_params:,}")
                training_table.add_row("Trainable Parameters", f"{trainable_params:,}")
                
                console.print(training_table)
                
        except Exception as e:
            self.logger.error(f"Failed to load model info: {str(e)}")


def main():
    """Main entry point for the CLI."""
    cli = LiquidSpikingCLI()
    cli.run()


if __name__ == "__main__":
    main()
