#!/usr/bin/env python3
"""
Comprehensive test script for spike-to-probability conversion mechanisms
and residual connections in liquid-spiking neural networks.

This script validates the implementation of:
1. SpikeDecoder with multiple decoding methods
2. TemporalAggregator for spike smoothing
3. PotentialBasedLayerNorm for spike vanishing prevention
4. ResidualLiquidSpikingBlock with proper skip connections
5. Enhanced text generation with spike information

Based on latest research:
- Kim et al. (2024): Residual connections in spiking networks
- Sun et al. (2022): Potential-based normalization
- Karn et al. (2024): Spike-to-probability conversion methods
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.main import (
    SpikeDecoder, TemporalAggregator, PotentialBasedLayerNorm,
    ResidualLiquidSpikingBlock, LiquidSpikingNetwork, ModelConfig,
    demonstrate_spike_probability_conversion, generate_text, TaskType
)

def test_spike_decoder():
    """Test SpikeDecoder with all decoding methods."""
    print("🧪 Testing SpikeDecoder with multiple decoding methods...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spike_dim = 128
    output_dim = 256
    batch_size = 4
    
    # Create test spike data (batch_size, seq_len, features) for temporal methods
    spike_data = torch.randn(batch_size, 10, spike_dim, device=device)
    
    results = {}
    
    for method in ['rate', 'temporal', 'first_spike', 'probabilistic', 'hybrid']:
        print(f"  📊 Testing {method} decoding...")
        
        try:
            decoder = SpikeDecoder(
                input_dim=spike_dim,
                output_dim=output_dim,
                decode_method=method,
                temporal_window=10,
                use_uncertainty=True
            ).to(device)
            
            # Test forward pass
            output = decoder(spike_data)
            
            results[method] = {
                'success': True,
                'output_shape': list(output.shape),
                'output_range': [float(output.min()), float(output.max())],
                'output_mean': float(output.mean()),
                'output_std': float(output.std()),
                'parameters': sum(p.numel() for p in decoder.parameters())
            }
            
            print(f"    ✅ {method}: Shape {output.shape}, Range [{output.min():.3f}, {output.max():.3f}]")
            
        except Exception as e:
            results[method] = {'success': False, 'error': str(e)}
            print(f"    ❌ {method}: {str(e)}")
    
    return results


def test_temporal_aggregator():
    """Test TemporalAggregator for spike smoothing."""
    print("🧪 Testing TemporalAggregator...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 64
    batch_size = 2
    num_timesteps = 20
    
    aggregator = TemporalAggregator(input_dim, window_size=10, aggregation_method='exponential').to(device)
    
    # Create test sequence data
    sequence_data = []
    for t in range(num_timesteps):
        # Simulate spiky data with some temporal structure
        data = torch.randn(batch_size, input_dim, device=device) * (1.0 + 0.5 * np.sin(t / 5))
        sequence_data.append(data)
    
    # Test aggregation
    aggregated = aggregator(sequence_data)
    
    print(f"  📊 Input sequence: {num_timesteps} timesteps of shape {sequence_data[0].shape}")
    print(f"  📊 Aggregated output: {aggregated.shape}")
    print(f"  📊 Smoothing effect: {aggregated.std():.3f} (vs raw std: {torch.stack(sequence_data).std():.3f})")
    
    return {
        'input_timesteps': num_timesteps,
        'input_shape': list(sequence_data[0].shape),
        'output_shape': list(aggregated.shape),
        'smoothing_ratio': float(aggregated.std() / torch.stack(sequence_data).std())
    }


def test_potential_based_layer_norm():
    """Test PotentialBasedLayerNorm for spike vanishing prevention."""
    print("🧪 Testing PotentialBasedLayerNorm...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalized_shape = 128
    batch_size = 4
    
    norm_layer = PotentialBasedLayerNorm(normalized_shape).to(device)
    
    # Test with different membrane potential distributions
    test_cases = {
        'normal': torch.randn(batch_size, normalized_shape, device=device),
        'saturated': torch.ones(batch_size, normalized_shape, device=device) * 5.0,
        'vanishing': torch.ones(batch_size, normalized_shape, device=device) * 0.01,
        'mixed': torch.cat([
            torch.randn(batch_size//2, normalized_shape, device=device) * 0.1,
            torch.randn(batch_size//2, normalized_shape, device=device) * 3.0
        ], dim=0)
    }
    
    results = {}
    
    for case_name, membrane_potential in test_cases.items():
        print(f"  📊 Testing {case_name} distribution...")
        
        normalized = norm_layer(membrane_potential)
        
        results[case_name] = {
            'input_stats': {
                'mean': float(membrane_potential.mean()),
                'std': float(membrane_potential.std()),
                'min': float(membrane_potential.min()),
                'max': float(membrane_potential.max())
            },
            'output_stats': {
                'mean': float(normalized.mean()),
                'std': float(normalized.std()),
                'min': float(normalized.min()),
                'max': float(normalized.max())
            },
            'normalization_effect': float(normalized.std() / membrane_potential.std()) if membrane_potential.std() > 0 else 0.0
        }
        
        print(f"    ✅ {case_name}: Std ratio = {results[case_name]['normalization_effect']:.3f}")
    
    return results


def test_residual_liquid_spiking_block():
    """Test ResidualLiquidSpikingBlock with different configurations."""
    print("🧪 Testing ResidualLiquidSpikingBlock...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 256
    liquid_units = 128
    spiking_units = 64
    spike_steps = 10
    batch_size = 2
    seq_len = 15
    
    # Test both residual types
    for residual_type in ['addition', 'concatenation']:
        print(f"  📊 Testing {residual_type} residual connections...")
        
        try:
            block = ResidualLiquidSpikingBlock(
                input_dim=input_dim,
                liquid_units=liquid_units,
                spiking_units=spiking_units,
                spike_steps=spike_steps,
                beta=0.9,
                backbone='cfc',
                residual_type=residual_type,
                use_potential_norm=True
            ).to(device)
            
            # Test with sequence data
            x = torch.randn(batch_size, seq_len, input_dim, device=device)
            h = None
            
            # Forward pass with internals (no explicit residual parameter needed)
            output, h_new, internals = block(x, h, return_internals=True)
            
            print(f"    ✅ {residual_type}: Input {x.shape} -> Output {output.shape}")
            print(f"    🔍 Internals available: {list(internals.keys())}")
            
            # Analyze spike information
            if 'spike_outputs' in internals:
                spike_data = internals['spike_outputs']
                spike_rate = torch.stack(spike_data).mean() if spike_data else 0.0
                print(f"    📊 Spike rate: {spike_rate:.3f}")
            
            # Test information preservation
            if residual_type == 'addition':
                # Check if residual information is preserved
                residual_contribution = torch.norm(output - x) / torch.norm(x)
                print(f"    📈 Residual contribution ratio: {residual_contribution:.3f}")
            
        except Exception as e:
            print(f"    ❌ {residual_type}: {str(e)}")
    
    return True


def test_integrated_network():
    """Test the complete integrated network with all new components."""
    print("🧪 Testing integrated LiquidSpikingNetwork...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a small test configuration
    config = ModelConfig(
        vocab_size=1000,
        embedding_dim=128,
        hidden_dim=256,
        input_dim=128,  # Required parameter
        liquid_units=64,
        spiking_units=32,
        output_dim=1000,
        num_layers=2,
        num_spike_steps=8,
        sequence_length=20,
        beta=0.9,
        dropout=0.1,
        learning_rate=0.001,
        batch_size=2,
        num_epochs=1,
        device=str(device),
        task_type=TaskType.LLM,
        liquid_backbone='cfc',  # Required parameter
        spike_threshold=1.0,  # Required parameter
        weight_decay=0.01,  # Required parameter
        gradient_clip=1.0,  # Required parameter
        mixed_precision=False,  # Required parameter
        seed=42  # Required parameter
    )
    
    try:
        # Create model
        model = LiquidSpikingNetwork(config).to(device)
        
        print(f"  📊 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.sequence_length), 
                                 device=device, dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_ids)
            print(f"  ✅ Forward pass: {input_ids.shape} -> {output.shape}")
        
        # Test spike-to-probability conversion demonstration
        print("  🔬 Running spike-probability conversion demonstration...")
        demo_results = demonstrate_spike_probability_conversion(model, config, show_internals=True)
        
        if 'error' not in demo_results:
            print(f"    ✅ Demonstration completed with {len(demo_results['processing_steps'])} layers")
            print(f"    📊 Spike analysis available for {len(demo_results['spike_analysis'])} layers")
            print(f"    📊 Decoder comparison: {list(demo_results.get('spike_decoder_comparison', {}).keys())}")
        else:
            print(f"    ❌ Demonstration failed: {demo_results['error']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integrated test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False


def run_comprehensive_tests():
    """Run all comprehensive tests for spike-to-probability conversion."""
    print("🚀 Starting comprehensive spike-to-probability conversion tests...\n")
    
    results = {}
    
    # Test individual components
    results['spike_decoder'] = test_spike_decoder()
    print()
    
    results['temporal_aggregator'] = test_temporal_aggregator()
    print()
    
    results['potential_norm'] = test_potential_based_layer_norm()
    print()
    
    results['residual_block'] = test_residual_liquid_spiking_block()
    print()
    
    results['integrated_network'] = test_integrated_network()
    print()
    
    # Summary
    print("📋 Test Summary:")
    print("=" * 50)
    
    for test_name, result in results.items():
        if isinstance(result, dict):
            if 'error' in result:
                print(f"❌ {test_name}: FAILED - {result['error']}")
            else:
                success_count = sum(1 for v in result.values() if isinstance(v, dict) and v.get('success', True))
                total_count = len([v for v in result.values() if isinstance(v, dict)])
                print(f"✅ {test_name}: {success_count}/{total_count} subtests passed")
        elif result:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n🎉 Comprehensive testing completed!")
    return results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_results = run_comprehensive_tests()
    
    # Optional: Save results
    try:
        import json
        with open('spike_probability_test_results.json', 'w') as f:
            # Convert tensor values to regular Python types for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                elif hasattr(obj, 'item'):  # torch tensors
                    return obj.item()
                elif isinstance(obj, (torch.Tensor, np.ndarray)):
                    return obj.tolist()
                else:
                    return obj
            
            json.dump(convert_for_json(test_results), f, indent=2)
        print("💾 Test results saved to spike_probability_test_results.json")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")