# TikToken Integration for Liquid-Spiking Neural Networks

## Overview

This implementation provides complete integration of modern TikToken tokenizers (GPT-3, GPT-4, GPT-4o) with the liquid-spiking neural network architecture. The integration maintains full compatibility with existing HuggingFace-based training pipelines while providing access to state-of-the-art tokenization capabilities.

## Features

### 🔧 TikTokenWrapper Class
- **HuggingFace Compatibility**: Drop-in replacement for HuggingFace tokenizers
- **Modern Encodings**: Supports cl100k_base (GPT-4), p50k_base (GPT-3), o200k_base (GPT-4o)
- **Special Token Handling**: Proper handling of padding, EOS, BOS, and UNK tokens
- **Error Recovery**: Robust fallback mechanisms for encoding failures

### 📚 AdvancedTokenizerManager
- **Multiple Tokenizers**: gpt2, gpt3, gpt4, o200k, codellama, llama2
- **Smart Fallbacks**: Automatic fallback to compatible tokenizers on failure
- **Vocabulary Management**: Dynamic vocabulary size handling (50K-200K+ tokens)
- **Unified Interface**: Consistent API regardless of underlying tokenizer

### 🏭 Enhanced DatasetFactory
- **Tokenizer Selection**: Create datasets with specific tokenizer types
- **Automatic Configuration**: Proper vocabulary size and sequence length setup
- **Backward Compatibility**: Maintains existing dataset creation interface

### 💻 CLI Integration
- **Command-Line Options**: Full tokenizer selection through CLI arguments
- **Easy Experimentation**: Simple switching between tokenizers for testing
- **Configuration Override**: Override vocabulary sizes and tokenizer settings

## Supported Tokenizers

| Tokenizer | Vocabulary Size | Encoding | Best For |
|-----------|----------------|----------|----------|
| `gpt2` | 50,257 | gpt2 | Legacy compatibility |
| `gpt3` | 50,281 | p50k_base | General text generation |
| `gpt4` | 100,277 | cl100k_base | Advanced language tasks |
| `o200k` | 200,019 | o200k_base | Next-generation performance |
| `codellama` | 32,000 | HuggingFace | Code generation |
| `llama2` | 32,000 | HuggingFace | General purpose |

## Usage Examples

### Basic Training

```bash
# Train with GPT-4 tokenizer
python scripts/cli.py train --task llm --tokenizer gpt4 --epochs 10

# Train with GPT-4o tokenizer (largest vocabulary)  
python scripts/cli.py train --task llm --tokenizer o200k --epochs 20

# Train with Code Llama for code generation
python scripts/cli.py train --task llm --tokenizer codellama --epochs 15
```

### Advanced Configuration

```bash
# Override vocabulary size
python scripts/cli.py train --task llm --tokenizer gpt4 --tokenizer-vocab-size 100000

# Custom sequence length with modern tokenizer
python scripts/cli.py train --task llm --tokenizer o200k --sequence-length 1024

# Multi-GPU training with modern tokenization
python scripts/cli.py train --task llm --tokenizer gpt4 --multi-gpu --epochs 50
```

### Programmatic Usage

```python
from src.core.tokenizer_upgrade import AdvancedTokenizerManager
from src.core.main import create_llm_config, DatasetFactory

# Initialize tokenizer manager
tokenizer_manager = AdvancedTokenizerManager(tokenizer_type='gpt4')
tokenizer = tokenizer_manager.tokenizer
vocab_size = tokenizer_manager.get_vocab_size()

# Create configuration with proper tokenizer
config = create_llm_config(tokenizer_type='gpt4')

# Create dataset with tokenizer integration
dataset, _ = DatasetFactory.create_llm_dataset(
    vocab_size=vocab_size,
    seq_length=512,
    tokenizer_type='gpt4'
)
```

## Architecture Integration

### Liquid-Spiking Network Compatibility
- **Dynamic Vocabulary**: Automatically adjusts embedding layers for different vocabulary sizes
- **Memory Efficiency**: Optimized for large vocabularies (up to 200K+ tokens)
- **Spike Enhancement**: Compatible with spike-to-probability conversion systems
- **Multi-GPU Support**: Seamless scaling across multiple GPUs

### Configuration Management
- **Automatic Setup**: Tokenizer-specific configurations generated automatically
- **Override Support**: Manual vocabulary size and parameter overrides available
- **Preset Integration**: Works with existing configuration presets
- **Error Handling**: Graceful degradation when tokenizers are unavailable

## Implementation Details

### File Structure
```
src/core/tokenizer_upgrade.py  # TikToken implementation
src/core/main.py              # Enhanced DatasetFactory and configs
scripts/cli.py                # CLI integration
```

### Key Classes
- `TikTokenWrapper`: HuggingFace-compatible tiktoken wrapper
- `AdvancedTokenizerManager`: High-level tokenizer management
- Enhanced `DatasetFactory`: Tokenizer-aware dataset creation
- Enhanced `create_llm_config`: Tokenizer-specific configuration

### Error Handling
- **Fallback Chains**: Automatic fallback to compatible tokenizers
- **Special Token Safety**: Proper handling of disallowed special tokens
- **Vocabulary Validation**: Ensures vocabulary sizes match tokenizer capabilities
- **Import Safety**: Graceful handling of missing dependencies

## Performance Benefits

### Tokenization Quality
- **Modern Algorithms**: GPT-4 level tokenization quality
- **Efficiency**: Better compression ratios than GPT-2 tokenizer
- **Multilingual**: Improved handling of non-English text
- **Code Support**: Specialized tokenizers for code generation

### Training Improvements
- **Faster Convergence**: Better tokenization leads to faster training
- **Higher Quality**: Improved text generation quality
- **Larger Context**: Support for larger vocabulary and context windows
- **Specialized Tasks**: Task-specific tokenizers for better performance

## Requirements

### Dependencies
```bash
pip install tiktoken transformers torch datasets
```

### Optional Dependencies
```bash
pip install ncps snntorch  # For full neural network functionality
```

## Testing

Run the integration tests:
```bash
python simple_tiktoken_test.py
```

This will verify:
- Tokenizer initialization
- Encoding/decoding functionality
- HuggingFace compatibility
- Batch processing
- Special token handling

## Future Enhancements

### Planned Features
- **Custom Tokenizers**: Support for user-defined tokenizer vocabularies
- **Streaming Tokenization**: Efficient processing of very long sequences
- **Tokenizer Caching**: Persistent caching of tokenizer states
- **Advanced Metrics**: Tokenization quality and efficiency metrics

### Performance Optimizations
- **Batch Optimization**: Further optimizations for batch tokenization
- **Memory Efficiency**: Reduced memory footprint for large vocabularies
- **GPU Acceleration**: GPU-accelerated tokenization where possible
- **Lazy Loading**: On-demand tokenizer loading for memory efficiency

## Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
pip install tiktoken transformers
```

**Special Token Errors**
- The implementation automatically handles special token conflicts
- Fallback mechanisms ensure continued operation

**Vocabulary Size Mismatches**
- Use `--tokenizer-vocab-size` to override automatic detection
- Check tokenizer logs for actual vocabulary sizes

**Memory Issues with Large Vocabularies**
- Reduce batch size for tokenizers with >100K vocabulary
- Use gradient checkpointing for memory efficiency

## Conclusion

This TikToken integration brings state-of-the-art tokenization to liquid-spiking neural networks, enabling training with modern tokenizers comparable to GPT-4 and GPT-4o. The implementation maintains full backward compatibility while providing significant improvements in tokenization quality and efficiency.

The integration is production-ready with comprehensive error handling, automatic fallbacks, and seamless CLI integration, making it easy to experiment with different tokenizers and find the optimal configuration for specific tasks.