#!/usr/bin/env python3
"""
Quick test to verify WikiText-103 and other datasets load correctly
and display proper logging information.
"""

import sys
sys.path.append('.')

from src.core.main import WikiTextDataset, DatasetFactory

print("\n" + "="*70)
print("🧪 TESTING DATASET LOADING")
print("="*70)

# Test 1: WikiText-103 direct loading
print("\n📚 Test 1: Loading WikiText-103 directly...")
print("-"*70)
try:
    texts = WikiTextDataset.load_wikitext103(split='train', cache_dir='./data')
    print(f"✅ SUCCESS: Loaded {len(texts):,} texts from WikiText-103")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: DatasetFactory with WikiText-103
print("\n📦 Test 2: Loading via DatasetFactory with wikitext103...")
print("-"*70)
try:
    dataset, tokenizer = DatasetFactory.create_llm_dataset(
        vocab_size=50257,
        seq_length=128,
        tokenizer_name='gpt2',
        dataset_type='wikitext103',
        cache_dir='./data'
    )
    print(f"✅ SUCCESS: Created dataset with {len(dataset):,} samples")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: DatasetFactory with combined datasets
print("\n🔀 Test 3: Loading combined datasets...")
print("-"*70)
try:
    dataset, tokenizer = DatasetFactory.create_llm_dataset(
        vocab_size=50257,
        seq_length=128,
        tokenizer_name='gpt2',
        dataset_type='combined',
        combined_datasets=['wikitext103'],  # Just wikitext103 for quick test
        cache_dir='./data'
    )
    print(f"✅ SUCCESS: Created combined dataset with {len(dataset):,} samples")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETE")
print("="*70)
print("\n💡 Next step: Run actual training with:")
print("   python scripts/cli.py train --task llm --dataset wikitext103 --epochs 3")
print()
