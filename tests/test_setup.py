#!/usr/bin/env python3
"""
Test script to verify the liquid-spiking neural network setup.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer

def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    
    try:
        import snntorch
        print(f"✅ snntorch: {snntorch.__version__}")
    except ImportError as e:
        print(f"❌ snntorch: {e}")
        return False
    
    try:
        import ncps
        print("✅ ncps: Available")
    except ImportError as e:
        print(f"❌ ncps: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✅ transformers: Available")
    except ImportError as e:
        print(f"❌ transformers: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("✅ datasets: Available")
    except ImportError as e:
        print(f"❌ datasets: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation."""
    print("\n🏗️  Testing model creation...")
    
    try:
        from src.core.main import create_llm_config, LiquidSpikingNetwork
        
        config = create_llm_config()
        print(f"✅ Config created: {config.task_type}")
        
        model = LiquidSpikingNetwork(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_processing():
    """Test text processing pipeline."""
    print("\n📝 Testing text processing...")
    
    try:
        from src.core.main import TextDataset, WikiTextDataset
        
        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded: {tokenizer.name_or_path}")
        
        # Test sample text
        sample_texts = [
            "This is a test sentence for the neural network training. Machine learning models require substantial amounts of training data to learn effectively. Neural networks process information through multiple layers of interconnected nodes. Deep learning has revolutionized artificial intelligence applications.",
            "Machine learning models require training data to learn patterns. The transformer architecture has become dominant in natural language processing. Attention mechanisms allow models to focus on relevant parts of the input sequence. Self-attention computes relationships between all positions in a sequence.",
            "The future of AI is promising and exciting for researchers. Natural language processing enables computers to understand human language. Computer vision allows machines to interpret visual information. Reinforcement learning trains agents through trial and error interactions."
        ]
        
        dataset = TextDataset(sample_texts, tokenizer, seq_length=16)  # Shorter sequence for test
        print(f"✅ TextDataset created with {len(dataset)} examples")
        
        # Test data loading
        example = dataset[0]
        print(f"✅ Data shape: input={example[0].shape}, target={example[1].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Text processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_small_training():
    """Test a small training loop."""
    print("\n🚀 Testing small training loop...")
    
    try:
        from src.core.main import (
            create_llm_config, LiquidSpikingNetwork, 
            LiquidSpikingTrainer, TextDataset
        )
        from torch.utils.data import DataLoader
        
        # Create small config
        config = create_llm_config()
        config.batch_size = 2
        config.sequence_length = 16  # Much shorter for test
        # Keep the same dimensions as main config to avoid mismatch
        # config.hidden_dim = 64
        # config.liquid_units = 32
        # config.spiking_units = 16
        config.num_layers = 2
        
        # Create small dataset
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        sample_texts = [
            "This is test text for training the model effectively. Machine learning requires substantial training data to achieve good performance. Neural networks learn complex patterns from input data through multiple layers of processing.",
            "Machine learning requires training data for effective learning. Deep learning models use multiple layers to process information hierarchically. Natural language processing enables computers to understand and generate human language.",
            "Neural networks learn from examples provided during training. Deep learning uses multiple layers to extract features automatically. Computer vision allows machines to interpret and understand visual information from the world.",
            "Deep learning uses multiple layers for feature extraction. Artificial intelligence has transformed many industries and applications. Machine learning algorithms can identify patterns in large datasets automatically.",
        ] * 10  # Repeat to get more examples
        
        dataset = TextDataset(sample_texts, tokenizer, seq_length=config.sequence_length)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        # Create model and trainer
        model = LiquidSpikingNetwork(config)
        trainer = LiquidSpikingTrainer(model, config)
        
        print(f"✅ Small model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test one batch
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            
            outputs = model(inputs)
            print(f"✅ Forward pass: input={inputs.shape}, output={outputs.shape}")
            
            # Test loss calculation
            loss = trainer.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )
            print(f"✅ Loss computation: {loss.item():.4f}")
            break
        
        return True
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🧠 LIQUID-SPIKING NEURAL NETWORK TEST")
    print("=" * 60)
    
    print(f"🖥️  PyTorch: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_imports()
    all_tests_passed &= test_model_creation()
    all_tests_passed &= test_text_processing()
    all_tests_passed &= test_small_training()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Ready for training.")
    else:
        print("❌ SOME TESTS FAILED! Check the errors above.")
    print("=" * 60)
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    exit(main())
