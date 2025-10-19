#!/usr/bin/env python3
"""
Convenience wrapper script for running the hybrid liquid-spiking neural network training.

This script provides easy access to the main training functionality without 
needing to navigate the source directory structure.
"""

import sys
import os

# Add project root to path so we can import from src.*
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    """Main entry point - route to different training options."""
    if len(sys.argv) < 2:
        print("Hybrid Liquid-Spiking Neural Network System")
        print("=" * 50)
        print("Usage:")
        print("  python train.py llm                    # Train language model")
        print("  python train.py llm_optimized          # Train optimized language model")
        print("  python train.py vision                 # Train vision model")
        print("  python train.py robotics               # Train robotics model")
        print("  python train.py cli [args...]          # Use full CLI interface")
        print("")
        print("Multi-GPU Examples:")
        print("  python train.py cli train --task llm --multi-gpu")
        print("  python train.py cli train --task vision --gpu-strategy dp --gpu-ids '0,1'")
        print("  python train.py cli train --task robotics --gpu-strategy ddp --multi-gpu")
        print("  python train.py cli info --gpu         # Show available GPUs")
        print("")
        print("Basic Examples:")
        print("  python train.py llm_optimized")
        print("  python train.py cli train --task vision --epochs 20")
        print("  python train.py cli benchmark --model-path models/llm_final.pt")
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "llm":
        # Import and run basic LLM training
        from src.training.train_llm import main as train_llm_main
        return train_llm_main()
        
    elif command == "llm_optimized":
        # Import and run optimized LLM training
        from src.training.train_llm_optimized import main as train_optimized_main
        return train_optimized_main()
        
    elif command == "vision":
        # Import and run vision training
        from src.core.main import train_vision_model
        print("Training vision model...")
        train_vision_model()
        return 0
        
    elif command == "robotics":
        # Import and run robotics training
        from src.core.main import train_robotics_model
        print("Training robotics model...")
        train_robotics_model()
        return 0
        
    elif command == "cli":
        # Run full CLI with remaining arguments
        import subprocess
        cli_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'cli.py')
        cli_args = [sys.executable, cli_path] + sys.argv[2:]
        return subprocess.call(cli_args)
        
    else:
        print(f"Unknown command: {command}")
        print("Run 'python train.py' for usage information.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
