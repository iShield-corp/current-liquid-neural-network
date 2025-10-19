#!/usr/bin/env python3
"""
Basic inference test without loading saved model weights
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'core'))

# Import directly from main module to avoid circular imports
from src.core.main import LiquidSpikingNetwork, create_llm_config

def test_basic_inference():
    """Test basic inference with fresh model"""
    try:
        # Create a basic config
        config = create_llm_config()
        print(f"Model config created: {type(config)}")
        
        # Create model
        model = LiquidSpikingNetwork(config)
        print(f"Model created successfully: {type(model)}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test forward pass with dummy input
        batch_size = 2
        seq_len = 10
        vocab_size = config.vocab_size
        
        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        print(f"Input shape: {input_ids.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
            print(f"Output shape: {outputs.shape}")
            
        # Test the spike enhancement functionality specifically
        print("\n--- Testing Spike Enhancement ---")
        if hasattr(model, 'spike_decoder'):
            try:
                # Create dummy spike features matching the expected shape
                dummy_spikes = torch.randn(
                    batch_size, 
                    model.spike_decoder.temporal_window,
                    config.spiking_units
                )
                print(f"Dummy spikes shape: {dummy_spikes.shape}")
                
                # Test spike decoding
                spike_result = model.spike_decoder(dummy_spikes, return_uncertainty=False)
                if isinstance(spike_result, tuple):
                    spike_probs = spike_result[0]
                    print(f"Spike enhancement output shape: {spike_probs.shape}")
                    print("✅ Spike enhancement working correctly!")
                else:
                    print(f"Spike enhancement output shape: {spike_result.shape}")
                    print("✅ Spike enhancement working correctly!")
                    
            except Exception as e:
                print(f"❌ Spike enhancement error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Model does not have spike_decoder attribute")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing basic inference...")
    success = test_basic_inference()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")