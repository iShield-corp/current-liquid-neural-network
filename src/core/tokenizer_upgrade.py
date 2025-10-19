"""
Advanced Tokenizer Integration for Liquid-Spiking Neural Networks

This module provides upgraded tokenization using GPT-3/GPT-4 level tokenizers
for better performance than the current GPT-2 tokenizer.
"""

import tiktoken
import torch
import json
import os
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class TikTokenWrapper:
    """
    HuggingFace-compatible wrapper for tiktoken encodings.
    
    This wrapper makes tiktoken encodings work seamlessly with existing
    HuggingFace-based training pipelines while providing all the benefits
    of modern GPT-3/GPT-4 tokenization.
    """
    
    def __init__(self, encoding_name: str, special_tokens: Optional[Dict[str, int]] = None):
        """
        Initialize TikToken wrapper.
        
        Args:
            encoding_name: Name of tiktoken encoding ('cl100k_base', 'p50k_base', etc.)
            special_tokens: Additional special tokens to add
        """
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Standard special tokens for compatibility
        self.special_tokens = special_tokens or {}
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.bos_token = "<|endoftext|>"
        self.unk_token = "<|endoftext|>"
        
        # Token IDs - handle special tokens properly
        try:
            self.pad_token_id = self.encoding.encode(self.pad_token, allowed_special={self.pad_token})[0] if self.pad_token else 0
            self.eos_token_id = self.encoding.encode(self.eos_token, allowed_special={self.eos_token})[0] if self.eos_token else 0
            self.bos_token_id = self.encoding.encode(self.bos_token, allowed_special={self.bos_token})[0] if self.bos_token else 0
            self.unk_token_id = self.encoding.encode(self.unk_token, allowed_special={self.unk_token})[0] if self.unk_token else 0
        except Exception as e:
            logger.warning(f"Failed to encode special tokens: {e}")
            # Fallback to reasonable defaults
            self.pad_token_id = 0
            self.eos_token_id = 0  
            self.bos_token_id = 0
            self.unk_token_id = 0
        
        # Vocabulary info
        self.vocab_size = self.encoding.n_vocab
        
        logger.info(f"✅ Initialized TikToken wrapper for {encoding_name}")
        logger.info(f"   📚 Vocabulary size: {self.vocab_size:,}")
        logger.info(f"   🔤 Special tokens: {len(self.special_tokens)}")
    
    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        try:
            # Allow all special tokens to prevent errors
            return self.encoding.encode(text, allowed_special="all", **kwargs)
        except Exception as e:
            logger.warning(f"Encoding error: {e}, falling back to disallowed_special=() mode")
            return self.encoding.encode(text, disallowed_special=(), **kwargs)
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True, **kwargs) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        try:
            return self.encoding.decode(token_ids, **kwargs)
        except Exception as e:
            # Fallback for invalid tokens
            logger.warning(f"Decode error: {e}, returning partial decode")
            valid_tokens = [t for t in token_ids if 0 <= t < self.vocab_size]
            return self.encoding.decode(valid_tokens, **kwargs) if valid_tokens else ""
    
    def __call__(self, text: Union[str, List[str]], 
                 max_length: Optional[int] = None,
                 padding: Union[bool, str] = False,
                 truncation: bool = False,
                 return_tensors: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        HuggingFace-compatible tokenization call.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length', True, False)
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch)
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Handle single text vs batch
        if isinstance(text, str):
            texts = [text]
            is_batch = False
        else:
            texts = text
            is_batch = True
        
        # Tokenize all texts
        all_token_ids = []
        for txt in texts:
            try:
                token_ids = self.encoding.encode(txt, allowed_special="all")
            except Exception as e:
                logger.warning(f"Encoding error for text '{txt[:50]}...': {e}")
                token_ids = self.encoding.encode(txt, disallowed_special=())
            
            # Apply truncation
            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            all_token_ids.append(token_ids)
        
        # Apply padding
        if padding and max_length:
            padded_ids = []
            for token_ids in all_token_ids:
                if len(token_ids) < max_length:
                    # Pad with pad_token_id
                    padding_length = max_length - len(token_ids)
                    token_ids = token_ids + [self.pad_token_id] * padding_length
                padded_ids.append(token_ids)
            all_token_ids = padded_ids
        
        # Create attention masks
        attention_masks = []
        for token_ids in all_token_ids:
            # 1 for real tokens, 0 for padding
            mask = [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]
            attention_masks.append(mask)
        
        # Prepare result
        result = {
            'input_ids': all_token_ids,
            'attention_mask': attention_masks
        }
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.long)
            result['attention_mask'] = torch.tensor(result['attention_mask'], dtype=torch.long)
        
        # Return single item if not batch
        if not is_batch and return_tensors == 'pt':
            result['input_ids'] = result['input_ids'].squeeze(0)
            result['attention_mask'] = result['attention_mask'].squeeze(0)
        elif not is_batch:
            result['input_ids'] = result['input_ids'][0]
            result['attention_mask'] = result['attention_mask'][0]
        
        return result
    
    def __len__(self) -> int:
        """Return vocabulary size for compatibility."""
        return self.vocab_size
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration for compatibility."""
        os.makedirs(save_directory, exist_ok=True)
        
        config = {
            'encoding_name': self.encoding_name,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'pad_token': self.pad_token,
            'eos_token': self.eos_token,
            'bos_token': self.bos_token,
            'unk_token': self.unk_token,
            'tokenizer_type': 'tiktoken'
        }
        
        config_path = os.path.join(save_directory, 'tokenizer_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Saved TikToken config to {config_path}")
    
    @classmethod
    def from_pretrained(cls, save_directory: str):
        """Load tokenizer from saved configuration."""
        config_path = os.path.join(save_directory, 'tokenizer_config.json')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Tokenizer config not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(
            encoding_name=config['encoding_name'],
            special_tokens=config.get('special_tokens', {})
        )

class AdvancedTokenizerManager:
    """
    Advanced tokenizer manager supporting multiple modern tokenizers.
    
    Supports:
    - GPT-3/GPT-4 tokenizers (cl100k_base, p50k_base) via tiktoken
    - Modern HuggingFace tokenizers
    - Custom vocabulary extensions
    """
    
    def __init__(self, tokenizer_type: str = "gpt4", vocab_size: Optional[int] = None):
        self.tokenizer_type = tokenizer_type
        self.tokenizer = None
        self.vocab_size = vocab_size
        self.is_tiktoken = False
        
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize the appropriate tokenizer."""
        if self.tokenizer_type == "gpt4":
            # GPT-4 tokenizer (cl100k_base) - Best overall
            try:
                self.tokenizer = TikTokenWrapper("cl100k_base")
                self.vocab_size = self.tokenizer.vocab_size
                self.is_tiktoken = True
                logger.info(f"✅ Loaded GPT-4 tokenizer (cl100k_base) with {self.vocab_size:,} tokens")
            except Exception as e:
                logger.error(f"Failed to load GPT-4 tokenizer: {e}")
                self._fallback_to_gpt3()
        
        elif self.tokenizer_type == "gpt3":
            # GPT-3 tokenizer (p50k_base)
            try:
                self.tokenizer = TikTokenWrapper("p50k_base")
                self.vocab_size = self.tokenizer.vocab_size
                self.is_tiktoken = True
                logger.info(f"✅ Loaded GPT-3 tokenizer (p50k_base) with {self.vocab_size:,} tokens")
            except Exception as e:
                logger.error(f"Failed to load GPT-3 tokenizer: {e}")
                self._fallback_to_modern_hf()
        
        elif self.tokenizer_type == "o200k":
            # GPT-4o tokenizer (o200k_base) - Newest and largest
            try:
                self.tokenizer = TikTokenWrapper("o200k_base")
                self.vocab_size = self.tokenizer.vocab_size
                self.is_tiktoken = True
                logger.info(f"✅ Loaded GPT-4o tokenizer (o200k_base) with {self.vocab_size:,} tokens")
            except Exception as e:
                logger.error(f"Failed to load GPT-4o tokenizer: {e}")
                self._fallback_to_gpt4()
        
        elif self.tokenizer_type == "codellama":
            # Code Llama tokenizer - Best for code
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.vocab_size = len(self.tokenizer)
                self.is_tiktoken = False
                logger.info(f"✅ Loaded Code Llama tokenizer with {self.vocab_size:,} tokens")
            except Exception as e:
                logger.error(f"Failed to load Code Llama tokenizer: {e}")
                self._fallback_to_modern_hf()
        
        elif self.tokenizer_type == "llama2":
            # Llama 2 tokenizer - Good general purpose
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.vocab_size = len(self.tokenizer)
                self.is_tiktoken = False
                logger.info(f"✅ Loaded Llama 2 tokenizer with {self.vocab_size:,} tokens")
            except Exception as e:
                logger.error(f"Failed to load Llama 2 tokenizer: {e}")
                self._fallback_to_modern_hf()
        
        else:
            # Modern HuggingFace tokenizer
            self._fallback_to_modern_hf()
    
    def _fallback_to_gpt4(self):
        """Fallback to GPT-4 tokenizer."""
        try:
            self.tokenizer = TikTokenWrapper("cl100k_base")
            self.vocab_size = self.tokenizer.vocab_size
            self.is_tiktoken = True
            logger.info(f"🔄 Fallback: Using GPT-4 tokenizer with {self.vocab_size:,} tokens")
        except:
            self._fallback_to_gpt3()
    
    def _fallback_to_gpt3(self):
        """Fallback to GPT-3 tokenizer."""
        try:
            self.tokenizer = TikTokenWrapper("p50k_base")
            self.vocab_size = self.tokenizer.vocab_size
            self.is_tiktoken = True
            logger.info(f"🔄 Fallback: Using GPT-3 tokenizer with {self.vocab_size:,} tokens")
        except:
            self._fallback_to_modern_hf()
    
    def _fallback_to_modern_hf(self):
        """Fallback to modern HuggingFace tokenizer."""
        try:
            # Try modern tokenizers in order of preference
            tokenizer_options = [
                "microsoft/DialoGPT-large",  # Good conversation tokenizer
                "EleutherAI/gpt-neo-2.7B",   # Modern GPT-style
                "gpt2"                        # Final fallback
            ]
            
            for tokenizer_name in tokenizer_options:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.vocab_size = len(self.tokenizer)
                    self.is_tiktoken = False
                    logger.info(f"🔄 Fallback: Using {tokenizer_name} with {self.vocab_size:,} tokens")
                    break
                except:
                    continue
            
        except Exception as e:
            logger.error(f"All tokenizer options failed: {e}")
            raise RuntimeError("Could not initialize any tokenizer")
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        if self.is_tiktoken:
            return self.tokenizer.encode(text, **kwargs)
        else:
            # HuggingFace tokenizer
            return self.tokenizer.encode(text, add_special_tokens=True, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text."""
        if self.is_tiktoken:
            return self.tokenizer.decode(token_ids, **kwargs)
        else:
            # HuggingFace tokenizer
            return self.tokenizer.decode(token_ids, skip_special_tokens=True, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make tokenizer callable for HuggingFace compatibility."""
        return self.tokenizer(*args, **kwargs)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer."""
        self.tokenizer.save_pretrained(save_directory)
    
    def tokenize_for_training(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize text for training with proper format."""
        result = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Ensure labels are included
        result["labels"] = result["input_ids"].clone()
        
        return {key: tensor.squeeze(0) for key, tensor in result.items()}

def upgrade_model_config_for_advanced_tokenizer(config, tokenizer_type: str = "gpt4"):
    """
    Upgrade model configuration to use advanced tokenizer.
    
    Args:
        config: ModelConfig instance
        tokenizer_type: Type of tokenizer ('gpt4', 'gpt3', 'codellama', 'llama2')
    
    Returns:
        Updated config with proper vocabulary size
    """
    tokenizer_manager = AdvancedTokenizerManager(tokenizer_type)
    
    # Update vocabulary size
    config.vocab_size = tokenizer_manager.get_vocab_size()
    config.output_dim = config.vocab_size
    
    logger.info(f"🔧 Upgraded config for {tokenizer_type} tokenizer:")
    logger.info(f"   📚 Vocabulary size: {config.vocab_size:,}")
    logger.info(f"   🎯 Output dimension: {config.output_dim:,}")
    
    return config

# Integration functions for existing codebase

def create_advanced_llm_config(tokenizer_type: str = "gpt4"):
    """Create LLM config with advanced tokenizer."""
    from src.core.main import create_llm_config
    
    config = create_llm_config()
    config = upgrade_model_config_for_advanced_tokenizer(config, tokenizer_type)
    
    return config

def create_advanced_tokenizer_for_training(tokenizer_type: str = "gpt4"):
    """Create advanced tokenizer for training."""
    return AdvancedTokenizerManager(tokenizer_type)