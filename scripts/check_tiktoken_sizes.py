#!/usr/bin/env python3

import tiktoken

def check_tiktoken_sizes():
    print("🔍 Checking TikToken vocabulary sizes...")
    
    encodings = [
        ('cl100k_base', 'GPT-4'),
        ('p50k_base', 'GPT-3'),
        ('o200k_base', 'GPT-4o'),
        ('r50k_base', 'GPT-3 (old)')
    ]
    
    for encoding_name, model_name in encodings:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            vocab_size = encoding.n_vocab
            print(f"✅ {model_name} ({encoding_name}): {vocab_size:,} tokens")
            
            # Test special token
            try:
                test_token = encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
                print(f"   <|endoftext|> token ID: {test_token[0] if test_token else 'None'}")
            except Exception as e:
                print(f"   <|endoftext|> failed: {e}")
                
        except Exception as e:
            print(f"❌ {model_name} ({encoding_name}): Error - {e}")
    
    print("\n🔧 Recommended vocab sizes:")
    print("gpt4: 100277 (cl100k_base)")
    print("gpt3: 50281 (p50k_base)")  
    print("o200k: 200000+ (o200k_base)")

if __name__ == "__main__":
    check_tiktoken_sizes()