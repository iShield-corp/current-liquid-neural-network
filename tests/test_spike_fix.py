#!/usr/bin/env python3
"""
Test script to verify the spike enhancement fix
"""
import torch
import torch.nn as nn
import sys
import traceback

# Mock the missing imports for testing
sys.modules['snntorch'] = type(sys)('snntorch')
sys.modules['snntorch.functional'] = type(sys)('snntorch.functional')
sys.modules['ncps'] = type(sys)('ncps')
sys.modules['ncps.torch'] = type(sys)('ncps.torch')

# Add mock functions
class MockSNNTorch:
    class Leaky:
        def __init__(self, beta, init_hidden=False):
            self.beta = beta
            self.init_hidden = init_hidden
        
        def __call__(self, x, mem):
            # Simple mock leaky integrate-and-fire
            spk = torch.zeros_like(x)
            mem = self.beta * mem + x
            spk = (mem > 1.0).float()
            mem = mem * (spk == 0)  # Reset membrane potential after spike
            return spk, mem

sys.modules['snntorch'].Leaky = MockSNNTorch.Leaky

class MockNCPS:
    class LTC:
        def __init__(self, input_size, hidden_size):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.fc = nn.Linear(input_size, hidden_size)
        
        def __call__(self, x, h):
            if h is None:
                h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            return self.fc(x), h

sys.modules['ncps.torch'].LTC = MockNCPS.LTC

# Import our code after mocking
try:
    from src.core.main import (
        TaskType, ModelConfig, LiquidSpikingNetwork, 
        ResidualLiquidSpikingBlock, generate_text
    )
    from transformers import GPT2TokenizerFast
    
    def test_spike_enhancement():
        """Test that the spike enhancement no longer throws tuple unpacking errors."""
        
        print("🔍 Testing spike enhancement fix...")
        
        # Create a minimal config for testing
        config = ModelConfig(
            task_type=TaskType.LLM,
            input_dim=512,
            hidden_dim=512,
            output_dim=1000,  # Smaller vocab for testing
            liquid_units=64,
            spiking_units=32,
            num_layers=2,
            num_spike_steps=8,
            vocab_size=1000,
            embedding_dim=512,
            max_position_embeddings=64,
            sequence_length=32,
            device='cpu'  # Use CPU for testing
        )
        
        # Create model
        model = LiquidSpikingNetwork(config)
        model.eval()
        
        # Create a simple tokenizer mock
        class MockTokenizer:
            def __init__(self):
                self.eos_token_id = 0
                self.vocab_size = 1000
            
            def __call__(self, text, **kwargs):
                # Return mock tokens
                return {
                    'input_ids': torch.randint(1, 100, (1, 5))  # Random tokens
                }
            
            def decode(self, tokens, **kwargs):
                return "test output text"
        
        tokenizer = MockTokenizer()
        
        try:
            # Test text generation with spike enhancement
            result = generate_text(
                model=model,
                config=config,
                tokenizer=tokenizer,
                prompt="Test",
                max_length=10,
                temperature=1.0,
                use_spike_enhancement=True
            )
            print("✅ Spike enhancement test PASSED - no tuple unpacking errors!")
            print(f"Generated result: {result}")
            return True
            
        except Exception as e:
            print(f"❌ Spike enhancement test FAILED: {e}")
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        success = test_spike_enhancement()
        sys.exit(0 if success else 1)
        
except Exception as e:
    print(f"❌ Test setup failed: {e}")
    traceback.print_exc()
    sys.exit(1)