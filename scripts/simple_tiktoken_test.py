#!/usr/bin/env python3
"""
Simple test script to verify tiktoken integration.
"""

import sys
import os
import importlib.util

def load_module_from_path(module_name, file_path):
    """Load a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_tiktoken_tokenizers():
    """Test tiktoken tokenizers basic functionality."""
    print("🧪 Testing TikToken Tokenizers")
    print("=" * 50)
    
    # Load the tokenizer module
    project_root = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(project_root, "src", "core", "tokenizer_upgrade.py")
    
    try:
        tokenizer_module = load_module_from_path("tokenizer_upgrade", tokenizer_path)
        AdvancedTokenizerManager = tokenizer_module.AdvancedTokenizerManager
        TikTokenWrapper = tokenizer_module.TikTokenWrapper
        print("✅ Successfully loaded tokenizer modules")
    except Exception as e:
        print(f"❌ Failed to load tokenizer modules: {e}")
        return
    
    # Test different tokenizers
    tokenizers_to_test = ['gpt2', 'gpt3', 'gpt4', 'o200k']
    
    for tokenizer_name in tokenizers_to_test:
        print(f"\n🔍 Testing {tokenizer_name} tokenizer...")
        
        try:
            # Initialize tokenizer manager
            tokenizer_manager = AdvancedTokenizerManager(tokenizer_type=tokenizer_name)
            tokenizer = tokenizer_manager.tokenizer
            vocab_size = tokenizer_manager.get_vocab_size()
            
            print(f"  ✅ Tokenizer initialized: {tokenizer_name}")
            print(f"  📊 Vocabulary size: {vocab_size:,}")
            
            # Test basic tokenization
            test_text = "Hello, world! This is a test of the tiktoken integration with GPT-4 style tokenizers."
            
            # Test direct encoding/decoding
            if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
                tokens = tokenizer.encode(test_text)
                decoded = tokenizer.decode(tokens)
                
                print(f"  🔤 Test text: '{test_text}'")
                print(f"  🔢 Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
                print(f"  🔤 Decoded: '{decoded}'")
                
                # Verify round-trip
                if decoded.strip() == test_text.strip():
                    print("  ✅ Round-trip encoding/decoding successful")
                else:
                    print("  ⚠️ Round-trip mismatch")
                    print(f"    Original: '{test_text}'")
                    print(f"    Decoded:  '{decoded}'")
            
            # Test HuggingFace compatibility
            try:
                hf_result = tokenizer(test_text, return_tensors='pt')
                print(f"  🤗 HuggingFace format: {list(hf_result.keys())}")
                print(f"  📏 Input IDs shape: {hf_result['input_ids'].shape}")
                
                # Test batch processing
                batch_texts = [
                    "Hello world",
                    "This is a longer text to test batch processing capabilities",
                    "Short"
                ]
                batch_result = tokenizer(batch_texts, return_tensors='pt', padding=True)
                print(f"  📦 Batch processing: {batch_result['input_ids'].shape}")
                
            except Exception as hf_error:
                print(f"  ❌ HuggingFace compatibility test failed: {hf_error}")
            
            # Test special tokens
            try:
                print(f"  🏷️ Special tokens:")
                if hasattr(tokenizer, 'pad_token_id'):
                    print(f"    PAD: {tokenizer.pad_token_id}")
                if hasattr(tokenizer, 'eos_token_id'):
                    print(f"    EOS: {tokenizer.eos_token_id}")
                if hasattr(tokenizer, 'bos_token_id'):
                    print(f"    BOS: {tokenizer.bos_token_id}")
                if hasattr(tokenizer, 'unk_token_id'):
                    print(f"    UNK: {tokenizer.unk_token_id}")
            except Exception as token_error:
                print(f"  ⚠️ Special token check failed: {token_error}")
            
            print(f"  ✅ {tokenizer_name} test completed successfully")
            
        except Exception as e:
            print(f"  ❌ {tokenizer_name} tokenizer test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎉 TikToken tokenizer testing completed!")

def test_tiktoken_wrapper_directly():
    """Test TikTokenWrapper class directly."""
    print("\n🔧 Testing TikTokenWrapper directly...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(project_root, "src", "core", "tokenizer_upgrade.py")
    
    try:
        tokenizer_module = load_module_from_path("tokenizer_upgrade", tokenizer_path)
        TikTokenWrapper = tokenizer_module.TikTokenWrapper
        
        # Test different encodings
        encodings = ['cl100k_base', 'p50k_base', 'o200k_base']
        
        for encoding_name in encodings:
            print(f"\n🔍 Testing {encoding_name} encoding...")
            
            try:
                wrapper = TikTokenWrapper(encoding_name)
                
                test_text = "The quick brown fox jumps over the lazy dog. 🦊"
                tokens = wrapper.encode(test_text)
                decoded = wrapper.decode(tokens)
                
                print(f"  ✅ Wrapper created for {encoding_name}")
                print(f"  📊 Vocab size: {wrapper.vocab_size:,}")
                print(f"  🔢 Tokens: {len(tokens)} -> {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
                print(f"  🔤 Round-trip: {'✅' if decoded == test_text else '❌'}")
                
                # Test HuggingFace interface
                hf_result = wrapper(test_text, return_tensors='pt')
                print(f"  🤗 HF interface: {hf_result['input_ids'].shape}")
                
            except Exception as e:
                print(f"  ❌ {encoding_name} test failed: {e}")
        
    except Exception as e:
        print(f"❌ TikTokenWrapper direct test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting TikToken Integration Tests")
    test_tiktoken_tokenizers()
    test_tiktoken_wrapper_directly()
    print("\n🏁 All tests completed!")