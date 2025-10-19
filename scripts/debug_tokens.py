#!/usr/bin/env python3

import sys
sys.path.append('src')

from src.core.tokenizer_upgrade import AdvancedTokenizerManager

def debug_tokenizer():
    print("🔍 Debugging tokenizer...")
    
    # Initialize tokenizer manager
    manager = AdvancedTokenizerManager(tokenizer_type="gpt4")
    tokenizer = manager.tokenizer
    
    print(f"📚 Tokenizer vocab size: {tokenizer.vocab_size:,}")
    print(f"🔤 Special token IDs:")
    print(f"   PAD: {tokenizer.pad_token_id}")
    print(f"   EOS: {tokenizer.eos_token_id}")
    print(f"   BOS: {tokenizer.bos_token_id}")
    print(f"   UNK: {tokenizer.unk_token_id}")
    
    # Test encoding a simple text
    test_text = "Hello world! This is a test sentence."
    tokens = tokenizer.encode(test_text)
    print(f"\n🧪 Test encoding: '{test_text}'")
    print(f"   Tokens: {tokens}")
    print(f"   Min token ID: {min(tokens)}")
    print(f"   Max token ID: {max(tokens)}")
    print(f"   Token range: 0 to {tokenizer.vocab_size - 1}")
    
    # Check if any tokens exceed vocab size
    invalid_tokens = [t for t in tokens if t >= tokenizer.vocab_size or t < 0]
    if invalid_tokens:
        print(f"❌ Invalid tokens found: {invalid_tokens}")
    else:
        print(f"✅ All tokens are valid")
    
    # Test with longer text
    longer_text = "The quick brown fox jumps over the lazy dog. " * 10
    longer_tokens = tokenizer.encode(longer_text)
    print(f"\n📖 Longer text test ({len(longer_tokens)} tokens):")
    print(f"   Min token ID: {min(longer_tokens)}")
    print(f"   Max token ID: {max(longer_tokens)}")
    
    invalid_longer = [t for t in longer_tokens if t >= tokenizer.vocab_size or t < 0]
    if invalid_longer:
        print(f"❌ Invalid tokens in longer text: {len(invalid_longer)} tokens")
        print(f"   Example invalid tokens: {invalid_longer[:10]}")
    else:
        print(f"✅ All longer tokens are valid")

if __name__ == "__main__":
    debug_tokenizer()