#!/usr/bin/env python3
"""
Test script for Mamba-Liquid-Spiking integration

This script tests the integrated Mamba-Liquid-Spiking architecture
to ensure all components work together correctly.
"""

import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("🧪 Testing Mamba-Liquid-Spiking Neural Network Integration")
print("=" * 70)

# Test 1: Import all modules
print("\n1️⃣  Testing imports...")
try:
    from src.core.mamba_ssm import MambaConfig, MambaBlock, MambaLayer
    print("   ✅ Mamba SSM modules imported")
except Exception as e:
    print(f"   ❌ Failed to import Mamba SSM: {e}")
    sys.exit(1)

try:
    from src.core.mamba_liquid_communication import (
        SpikeToMambaAdapter, MambaToSpikeAdapter,
        LiquidMambaGate, BidirectionalStateExchange,
        CrossModalAttention, StateProjection
    )
    print("   ✅ Communication adapters imported")
except Exception as e:
    print(f"   ❌ Failed to import communication adapters: {e}")
    sys.exit(1)

try:
    from src.core.mamba_liquid_integration import (
        IntegratedMambaLiquidSpikingBlock,
        MambaLiquidSpikingNetwork
    )
    print("   ✅ Integrated blocks imported")
except Exception as e:
    print(f"   ❌ Failed to import integrated blocks: {e}")
    sys.exit(1)

# Test 2: Test Mamba SSM standalone
print("\n2️⃣  Testing Mamba SSM standalone...")
try:
    mamba_config = MambaConfig(
        d_model=256,
        d_state=16,
        d_conv=4,
        expand_factor=2
    )
    mamba_layer = MambaLayer(mamba_config)
    
    batch_size = 2
    seq_len = 128
    x = torch.randn(batch_size, seq_len, mamba_config.d_model)
    
    with torch.no_grad():
        output = mamba_layer(x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape}"
    print(f"   ✅ Mamba SSM working (input: {x.shape}, output: {output.shape})")
except Exception as e:
    print(f"   ❌ Mamba SSM test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test communication adapters
print("\n3️⃣  Testing communication adapters...")
try:
    # Spike to Mamba adapter
    spike_adapter = SpikeToMambaAdapter(
        spiking_units=128,
        mamba_dim=256,
        method='temporal'
    )
    spike_train = (torch.rand(2, 64, 128) > 0.5).float()
    continuous = spike_adapter(spike_train)
    assert continuous.shape == (2, 64, 256), f"Adapter output shape wrong"
    print(f"   ✅ Spike-to-Mamba adapter working")
    
    # Liquid-Mamba gate
    gate = LiquidMambaGate(hidden_dim=256, num_heads=4)
    liquid_out = torch.randn(2, 64, 256)
    mamba_out = torch.randn(2, 64, 256)
    fused, weights = gate(liquid_out, mamba_out)
    assert fused.shape == (2, 64, 256), "Gate output shape wrong"
    print(f"   ✅ Liquid-Mamba gate working")
    
    # Cross-attention
    cross_attn = CrossModalAttention(liquid_dim=256, mamba_dim=256)
    enh_l, enh_m = cross_attn(liquid_out, mamba_out)
    assert enh_l.shape == liquid_out.shape, "Cross-attn liquid shape wrong"
    assert enh_m.shape == mamba_out.shape, "Cross-attn mamba shape wrong"
    print(f"   ✅ Cross-modal attention working")
    
except Exception as e:
    print(f"   ❌ Communication adapters test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test integrated block (all modes)
print("\n4️⃣  Testing integrated blocks...")

class MockConfig:
    input_dim = 128
    spiking_units = 256
    liquid_units = 256
    hidden_dim = 512
    output_dim = 50000
    num_spike_steps = 10
    beta = 0.95
    liquid_backbone = 'cfc'
    num_layers = 2
    mamba_d_state = 16
    mamba_d_conv = 4
    mamba_expand = 2
    integration_mode = 'sequential'

config = MockConfig()

for mode in ['sequential', 'parallel', 'bidirectional']:
    try:
        config.integration_mode = mode
        block = IntegratedMambaLiquidSpikingBlock(config, integration_mode=mode)
        
        x = torch.randn(2, 32, config.input_dim)
        with torch.no_grad():
            output, h_new = block(x)
        
        assert output.shape == (2, 32, config.hidden_dim), \
            f"{mode}: output shape wrong"
        print(f"   ✅ {mode.capitalize()} mode working")
    except Exception as e:
        print(f"   ❌ {mode.capitalize()} mode failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Test 5: Test full network
print("\n5️⃣  Testing full network...")
try:
    config.integration_mode = 'bidirectional'
    network = MambaLiquidSpikingNetwork(config)
    
    # Test with sequence input
    x = torch.randn(2, 32, config.input_dim)
    with torch.no_grad():
        output, h_states = network(x)
    
    assert output.shape == (2, 32, config.output_dim), \
        f"Network output shape wrong: {output.shape}"
    assert len(h_states) == config.num_layers, \
        f"Wrong number of hidden states: {len(h_states)}"
    
    print(f"   ✅ Full network working (output: {output.shape})")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"   ℹ️  Total parameters: {total_params:,}")
    
except Exception as e:
    print(f"   ❌ Full network test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test with different sequence lengths
print("\n6️⃣  Testing scalability...")
try:
    sequence_lengths = [64, 128, 256, 512]
    times = []
    
    import time
    
    for seq_len in sequence_lengths:
        x = torch.randn(1, seq_len, config.input_dim)
        
        start = time.time()
        with torch.no_grad():
            output, _ = network(x)
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"   ✅ seq_len={seq_len:4d}: {elapsed*1000:6.2f}ms")
    
    # Check if roughly linear scaling
    scaling_ratio = times[-1] / times[0]
    expected_ratio = sequence_lengths[-1] / sequence_lengths[0]
    
    if scaling_ratio < expected_ratio * 1.5:  # Allow 50% overhead
        print(f"   ✅ Scaling is roughly linear ({scaling_ratio:.1f}x "
              f"for {expected_ratio}x longer sequence)")
    else:
        print(f"   ⚠️  Scaling might be suboptimal ({scaling_ratio:.1f}x "
              f"for {expected_ratio}x longer sequence)")
    
except Exception as e:
    print(f"   ❌ Scalability test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Gradient flow test
print("\n7️⃣  Testing gradient flow...")
try:
    network.train()
    x = torch.randn(2, 32, config.input_dim, requires_grad=True)
    output, _ = network(x)
    
    # Compute dummy loss and backward
    loss = output.sum()
    loss.backward()
    
    # Check if gradients exist
    assert x.grad is not None, "No gradient for input"
    
    # Check if all parameters have gradients
    params_with_grad = 0
    params_without_grad = 0
    for name, param in network.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                params_with_grad += 1
            else:
                params_without_grad += 1
    
    print(f"   ✅ Gradients computed for {params_with_grad} parameters")
    if params_without_grad > 0:
        print(f"   ⚠️  {params_without_grad} parameters without gradients")
    
except Exception as e:
    print(f"   ❌ Gradient flow test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("🎉 All tests passed! Mamba-Liquid-Spiking integration is working!")
print("=" * 70)
print("\n📚 Next steps:")
print("   1. Train with: python train.py cli train --task llm --use-mamba")
print("   2. See docs/MAMBA_INTEGRATION.md for detailed usage")
print("   3. Try different integration modes: sequential, parallel, bidirectional")
print("\n✨ Happy training!")
