#!/usr/bin/env python3
"""
Train Liquid-Spiking Neural Network for LLM task with real text data.
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.main import (
    train_llm_model, generate_text, evaluate_perplexity, 
    load_model, TaskType, WikiTextDataset
)

def main():
    print("=" * 80)
    print("🧠 LIQUID-SPIKING NEURAL NETWORK LLM TRAINING")
    print("=" * 80)
    
    # Create output directory
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./data', exist_ok=True)
    
    print("\n📊 System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n🚀 Starting LLM Training...")
    
    try:
        # Train the model
        model, trainer = train_llm_model()
        
        print("\n✅ Training completed successfully!")
        print(f"📈 Best validation loss: {trainer.best_val_loss:.4f}")
        
        # Load tokenizer for text generation
        try:
            tokenizer = AutoTokenizer.from_pretrained('./llm_tokenizer')
        except:
            print("⚠️  Tokenizer not found, using default GPT-2 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        print("\n🎯 Testing text generation...")
        
        # Test prompts
        test_prompts = [
            "The future of artificial intelligence",
            "Machine learning is",
            "Deep learning models",
            "Neural networks can"
        ]
        
        for prompt in test_prompts:
            print(f"\n📝 Prompt: '{prompt}'")
            try:
                # Load the saved model for generation
                loaded_model, config = load_model("llm_model_final.pt", TaskType.LLM)
                generated_text = generate_text(
                    loaded_model, config, tokenizer, 
                    prompt=prompt, max_length=50, temperature=0.8
                )
                if generated_text:
                    print(f"🤖 Generated: {generated_text}")
                else:
                    print("❌ Generation failed")
            except Exception as e:
                print(f"❌ Generation error: {str(e)}")
        
        # Evaluate perplexity
        print("\n📊 Evaluating model perplexity...")
        try:
            test_texts = WikiTextDataset.load_wikitext2('train')[:50]  # Small test set
            perplexity = evaluate_perplexity(model, trainer.config, tokenizer, test_texts)
            if perplexity:
                print(f"📈 Model perplexity: {perplexity:.2f}")
            else:
                print("❌ Perplexity evaluation failed")
        except Exception as e:
            print(f"❌ Perplexity error: {str(e)}")
        
        print("\n🎉 LLM training and evaluation completed!")
        print("📁 Model saved to: llm_model_final.pt")
        print("📁 Tokenizer saved to: ./llm_tokenizer")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
