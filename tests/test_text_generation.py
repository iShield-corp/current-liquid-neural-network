#!/usr/bin/env python3
"""
Test text generation with spike enhancement
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'core'))

from src.core.main import LiquidSpikingNetwork, create_llm_config, generate_text

def test_text_generation():
    """Test text generation with spike enhancement"""
    try:
        print("Creating model...")
        config = create_llm_config()
        model = LiquidSpikingNetwork(config)
        
        # Create a simple mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 50257
                self.eos_token_id = 50256
                
            def encode(self, text):
                # Simple word-based encoding
                words = text.split()
                return [hash(word) % (self.vocab_size - 100) for word in words]
                
            def decode(self, tokens):
                # Simple mock decoding
                return " ".join([f"word_{token % 1000}" for token in tokens])
        
        tokenizer = MockTokenizer()
        
        print("Testing text generation...")
        
        # Test with spike enhancement
        result_enhanced = generate_text(
            model=model,
            config=config,
            tokenizer=tokenizer,
            prompt="Hello world",
            max_length=20,
            temperature=0.7,
            use_spike_enhancement=True
        )
        
        print(f"Enhanced generation result: {result_enhanced}")
        
        # Test without spike enhancement
        result_standard = generate_text(
            model=model,
            config=config,
            tokenizer=tokenizer,
            prompt="Hello world",
            max_length=20,
            temperature=0.7,
            use_spike_enhancement=False
        )
        
        print(f"Standard generation result: {result_standard}")
        
        print("✅ Text generation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing text generation...")
    success = test_text_generation()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")