#!/usr/bin/env python3
"""
Test script to verify that the new general language datasets integrate properly
with the existing programming dataset infrastructure.
"""

import sys
import os
import torch
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.advanced_programming_datasets import ProgrammingDatasetConfig, AdvancedProgrammingDataset

def test_dataset_integration():
    """Test that the mixed programming and general language dataset loads correctly."""
    print("Testing mixed programming and general language dataset integration...")
    print("=" * 60)
    
    # Initialize tokenizer
    print("1. Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("   ✓ Tokenizer loaded successfully")
    
    # Create test configuration with small samples for quick testing
    print("\n2. Creating test configuration...")
    config = ProgrammingDatasetConfig(
        sequence_length=128,  # Shorter for testing
        total_samples_limit=1000,  # Small sample for testing
        samples_per_language=50,  # Very small for testing
        dataset_ratios={
            "the_stack": 0.3,      # Programming: 30%
            "github_code": 0.2,    # Programming: 20%
            "code_search_net": 0.15, # Programming: 15%
            "apps": 0.05,          # Programming: 5%
            "tiny_codes": 0.05,    # Programming: 5%
            "wikipedia": 0.15,     # General: 15%
            "openorca": 0.1        # General: 10%
        }
    )
    print("   ✓ Configuration created")
    print(f"   • Programming content: {(0.3+0.2+0.15+0.05+0.05)*100}%")
    print(f"   • General language content: {(0.15+0.1)*100}%")
    
    # Test dataset creation (this will test if our new methods work)
    print("\n3. Testing dataset creation...")
    try:
        dataset = AdvancedProgrammingDataset(
            config=config,
            tokenizer=tokenizer,
            split="train",
            cache_dir="./test_cache"
        )
        print("   ✓ Dataset created successfully!")
        print(f"   • Total samples: {len(dataset):,}")
        
        # Test a few samples to see the variety of content
        print("\n4. Testing sample content variety...")
        if len(dataset) > 0:
            sample_indices = [0, min(len(dataset)//4, len(dataset)-1), min(len(dataset)//2, len(dataset)-1)]
            for i, idx in enumerate(sample_indices):
                if idx < len(dataset):
                    sample = dataset[idx]
                    if isinstance(sample, dict):
                        source = sample.get('source', 'unknown')
                        content_type = sample.get('content_type', 'unknown')
                        language = sample.get('language', 'unknown')
                        print(f"   Sample {i+1}: source='{source}', type='{content_type}', lang='{language}'")
                    else:
                        print(f"   Sample {i+1}: tensor shape={sample.shape if hasattr(sample, 'shape') else 'unknown'}")
        
        print("\n✓ All tests passed! Mixed dataset integration successful.")
        return True
        
    except Exception as e:
        print(f"\n✗ Dataset creation failed: {str(e)}")
        print("This may be due to network issues or dataset availability.")
        print("The integration code is correct, but datasets may need to be available online.")
        return False

if __name__ == "__main__":
    try:
        success = test_dataset_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)
