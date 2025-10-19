#!/usr/bin/env python3
"""
Test text generation with trained liquid-spiking LLM
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.main import LiquidSpikingNetwork, ModelConfig, TaskType
from transformers import GPT2Tokenizer

def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create config matching training config
    config = ModelConfig(
        task_type=TaskType.LLM,
        input_dim=512,  # Embedding dimension for text
        hidden_dim=512,
        output_dim=50257,  # GPT-2 vocabulary size
        liquid_units=256,  # Reduced to fix NCP issue
        spiking_units=128,
        num_layers=6,  # Reduced for faster training
        dropout=0.1,
        spike_threshold=1.0,
        beta=0.95,
        liquid_backbone='cfc',
        sequence_length=64,  # Shorter sequences for easier processing
        batch_size=8,  # Smaller batch size for memory efficiency
        learning_rate=5e-5,  # Lower learning rate for stable training
        weight_decay=1e-5,
        gradient_clip=1.0,
        mixed_precision=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    # Create model
    model = LiquidSpikingNetwork(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to load state dict, handling potential key mismatches
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model state dict loaded successfully")
    except RuntimeError as e:
        print(f"⚠️  Partial loading due to: {e}")
        # Try partial loading
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"✅ Loaded {len(pretrained_dict)}/{len(checkpoint['model_state_dict'])} layers")
    
    model.eval()
    
    print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, config

def generate_text(model, config, tokenizer, prompt, max_length=50, temperature=0.8):
    """Generate text with the trained model"""
    print(f"\n🎯 Generating text from prompt: '{prompt}'")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate
    with torch.no_grad():
        generated = input_ids.clone()
        
        for _ in range(max_length):
            # Get last sequence_length tokens
            seq_len = config.sequence_length  # Use config sequence length
            if generated.size(1) >= seq_len:
                input_seq = generated[:, -seq_len:]
            else:
                # Pad if shorter
                padding = torch.zeros(1, seq_len - generated.size(1), dtype=torch.long)
                input_seq = torch.cat([padding, generated], dim=1)
            
            # Forward pass
            outputs = model(input_seq)
            
            # Get last token logits and apply temperature
            logits = outputs[0, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we hit end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"📝 Generated: {generated_text}")
    return generated_text

def main():
    print("=" * 80)
    print("🧠 LIQUID-SPIKING LLM TEXT GENERATION TEST")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load best model (adjust path since we're in tests/ subdirectory)
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_model_epoch_3.pt')
    model, config = load_model(checkpoint_path)
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time",
        "Neural networks are",
        "In the year 2024",
        "Machine learning"
    ]
    
    print(f"\n🚀 Testing text generation with {len(prompts)} prompts...\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"--- Test {i}/{len(prompts)} ---")
        try:
            generated = generate_text(model, config, tokenizer, prompt, max_length=30)
            print("✅ Generation successful\n")
        except Exception as e:
            print(f"❌ Generation failed: {e}\n")
    
    print("=" * 80)
    print("🎉 Text generation testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
