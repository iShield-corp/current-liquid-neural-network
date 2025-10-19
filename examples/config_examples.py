#!/usr/bin/env python3
"""
Example script demonstrating how to create and use custom configurations
for the Hybrid Liquid-Spiking Neural Network.
"""

import sys
import os

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.main import (
    create_llm_config, create_vision_config, create_robotics_config,
    create_custom_config, save_config, load_config, print_config_summary,
    get_model_parameter_count, ModelConfig, TaskType
)

def example_basic_configs():
    """Show the default configurations for each task type."""
    print("🚀 Basic Configuration Examples")
    print("=" * 60)
    
    # LLM Configuration
    print("\n📝 Language Model Configuration:")
    llm_config = create_llm_config()
    print_config_summary(llm_config)
    
    # Vision Configuration
    print("\n👁️  Vision Model Configuration:")
    vision_config = create_vision_config()
    print_config_summary(vision_config)
    
    # Robotics Configuration
    print("\n🤖 Robotics Model Configuration:")
    robotics_config = create_robotics_config()
    print_config_summary(robotics_config)

def example_custom_configs():
    """Show how to create custom configurations."""
    print("\n🔧 Custom Configuration Examples")
    print("=" * 60)
    
    # Example 1: Large LLM with more parameters
    print("\n📈 Large Language Model Configuration:")
    large_llm_config = create_custom_config(
        'llm',
        liquid_units=512,           # Double the liquid units
        spiking_units=256,          # Double the spiking units
        num_layers=12,              # More layers
        num_attention_heads=16,     # More attention heads
        hidden_dim=768,             # Larger hidden dimension
        embedding_dim=768,          # Larger embedding dimension
        sequence_length=256,        # Longer sequences
        batch_size=8,               # Smaller batch size due to larger model
        num_epochs=20
    )
    print_config_summary(large_llm_config)
    
    # Show parameter count
    params = get_model_parameter_count(large_llm_config)
    print(f"\n📊 Estimated Parameters:")
    for component, count in params.items():
        if component != 'total':
            print(f"   • {component}: {count:,}")
    print(f"   • Total: {params['total']:,}")
    
    # Example 2: Lightweight vision model
    print("\n🪶 Lightweight Vision Model Configuration:")
    light_vision_config = create_custom_config(
        'vision',
        liquid_units=64,            # Fewer liquid units
        spiking_units=32,           # Fewer spiking units
        num_layers=3,               # Fewer layers
        num_attention_heads=4,      # Fewer attention heads
        hidden_dim=128,             # Smaller hidden dimension
        conv_channels=[16, 32, 64], # Smaller conv channels
        dropout=0.2,                # Higher dropout for regularization
        batch_size=256,             # Larger batch size for efficiency
        learning_rate=2e-3          # Higher learning rate
    )
    print_config_summary(light_vision_config)
    
    # Example 3: High-precision robotics model
    print("\n🎯 High-Precision Robotics Model Configuration:")
    precision_robotics_config = create_custom_config(
        'robotics',
        liquid_units=128,           # More liquid units for precision
        spiking_units=64,           # More spiking units
        num_layers=5,               # More layers
        hidden_dim=256,             # Larger hidden dimension
        spike_threshold=0.5,        # Lower spike threshold for sensitivity
        beta=0.9,                   # Higher beta for longer memory
        sequence_length=200,        # Longer sequences for complex behaviors
        dropout=0.05,               # Lower dropout for precision
        learning_rate=1e-4,         # Lower learning rate for stability
        num_epochs=50               # More epochs for convergence
    )
    print_config_summary(precision_robotics_config)

def example_save_load_configs():
    """Show how to save and load configurations."""
    print("\n💾 Configuration Save/Load Examples")
    print("=" * 60)
    
    # Create a custom config
    custom_config = create_custom_config(
        'llm',
        liquid_units=384,
        spiking_units=192,
        num_layers=8,
        hidden_dim=640
    )
    
    # Save to file
    config_path = "configs/custom_example_config.json"
    os.makedirs("configs", exist_ok=True)
    save_config(custom_config, config_path)
    
    # Load from file
    loaded_config = load_config(config_path)
    
    # Verify they're the same
    print(f"✅ Configs match: {custom_config.to_dict() == loaded_config.to_dict()}")
    
    print(f"\nLoaded configuration:")
    print_config_summary(loaded_config)

def example_parameter_scaling():
    """Show how parameter count scales with different configurations."""
    print("\n📈 Parameter Scaling Examples")
    print("=" * 60)
    
    configurations = [
        ("Small LLM", {
            'liquid_units': 128, 'spiking_units': 64, 'num_layers': 4, 
            'hidden_dim': 256, 'num_attention_heads': 4
        }),
        ("Medium LLM", {
            'liquid_units': 256, 'spiking_units': 128, 'num_layers': 6, 
            'hidden_dim': 512, 'num_attention_heads': 8
        }),
        ("Large LLM", {
            'liquid_units': 512, 'spiking_units': 256, 'num_layers': 12, 
            'hidden_dim': 768, 'num_attention_heads': 16
        }),
        ("XL LLM", {
            'liquid_units': 1024, 'spiking_units': 512, 'num_layers': 24, 
            'hidden_dim': 1024, 'num_attention_heads': 32
        })
    ]
    
    print(f"{'Model Size':<12} {'Total Params':<15} {'Liquid Units':<12} {'Spiking Units':<13} {'Layers':<7} {'Hidden Dim':<11}")
    print("-" * 80)
    
    for name, custom_params in configurations:
        config = create_custom_config('llm', **custom_params)
        params = get_model_parameter_count(config)
        
        print(f"{name:<12} {params['total']:>14,} {config.liquid_units:>11} {config.spiking_units:>12} {config.num_layers:>6} {config.hidden_dim:>10}")

def main():
    """Main function to run all examples."""
    print("🧠 Hybrid Liquid-Spiking Neural Network")
    print("📋 Configuration Examples and Guide")
    print("=" * 80)
    
    # Run all examples
    example_basic_configs()
    example_custom_configs()
    example_save_load_configs()
    example_parameter_scaling()
    
    print(f"\n✨ Configuration examples completed!")
    print(f"💡 Use these patterns to create your own custom configurations.")
    print(f"📚 See the ModelConfig class for all available parameters.")

if __name__ == "__main__":
    main()
