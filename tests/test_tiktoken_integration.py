#!/usr/bin/env python3

import sys
sys.path.append('src')

from src.core.tokenizer_upgrade import AdvancedTokenizerManager

def test_tiktoken_integration():
    print("🧪 Testing TikToken GPT-3/GPT-4 Integration")
    print("=" * 60)
    
    # Test all tokenizers
    tokenizers = ["gpt2", "gpt3", "gpt4", "o200k"]
    
    for tokenizer_type in tokenizers:
        print(f"\n🔤 Testing {tokenizer_type.upper()} tokenizer:")
        try:
            manager = AdvancedTokenizerManager(tokenizer_type=tokenizer_type)
            tokenizer = manager.tokenizer
            
            print(f"   ✅ Loaded successfully")
            print(f"   📚 Vocab size: {tokenizer.vocab_size:,}")
            
            # Test encoding/decoding
            test_text = "Hello, world! This is a test of the tiktoken integration."
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            
            print(f"   🔢 Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"   🔄 Round-trip: {'✅' if decoded.strip() == test_text.strip() else '❌'}")
            
            # Validate token range
            valid_tokens = all(0 <= t < tokenizer.vocab_size for t in tokens)
            print(f"   🎯 Token range valid: {'✅' if valid_tokens else '❌'}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 TikToken integration test complete!")
    print("\n📋 Summary:")
    print("   • TikTokenWrapper: HuggingFace-compatible wrapper")
    print("   • AdvancedTokenizerManager: Multi-tokenizer support")
    print("   • DatasetFactory: Enhanced with tokenizer_type parameter")
    print("   • CLI Integration: --tokenizer argument support")
    print("   • Training: Corrected vocab sizes for all tokenizers")

if __name__ == "__main__":
    test_tiktoken_integration()