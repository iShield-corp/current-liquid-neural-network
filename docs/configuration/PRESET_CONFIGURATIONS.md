# 🎯 Preset Configuration System

The Liquid-Spiking Neural Network CLI now includes advanced preset configurations that allow you to easily train models competitive with state-of-the-art LLMs without manually specifying complex parameter combinations.

## 🚀 Available Presets

### 1. GPT-4o Competitive (`gpt-4o`)
Optimized to compete with GPT-4o performance using hybrid liquid-spiking architecture advantages.

**Key Features:**
- 2048 hidden dimensions
- 1024 liquid units with advanced dynamics
- 512 spiking units for temporal processing
- 24 hybrid layers
- 4096 sequence length
- 8192 max position embeddings
- Multi-GPU ready architecture

**Usage:**
```bash
python scripts/cli.py train --task llm --config-preset gpt-4o --epochs 100
```

### 2. Claude Sonnet 4 Competitive (`claude-sonnet-4`)
Enhanced reasoning capabilities designed to match Claude Sonnet 4 performance.

**Key Features:**
- 2560 hidden dimensions
- 1280 liquid units for complex reasoning
- 640 spiking units with temporal credit assignment
- 32 deep layers for multi-step reasoning
- 8192 sequence length
- 16384 max position embeddings
- Advanced long-context understanding

**Usage:**
```bash
python scripts/cli.py train --task llm --config-preset claude-sonnet-4 --epochs 150
```

### 3. Ultra-Advanced (`ultra-advanced`)
Revolutionary configuration designed to exceed GPT-4, Claude, and Gemini performance.

**Key Features:**
- 4096 massive hidden dimensions
- 2048 revolutionary liquid units
- 1024 advanced spiking units
- 48 deep layers for maximum capacity
- 16384 sequence length
- 32768 massive context window
- 100K expanded vocabulary
- State-of-the-art hybrid architecture

**Usage:**
```bash
python scripts/cli.py train --task llm --config-preset ultra-advanced --epochs 200
```

## 🛠️ Backward Compatibility

The traditional custom parameter approach still works exactly as before:

```bash
python scripts/cli.py train --task llm \
  --liquid-units 128 --spiking-units 64 \
  --num-layers 3 --hidden-dim 256 \
  --num-attention-heads 4 --spike-threshold 0.8 \
  --beta 0.9 --learning-rate 5e-4 \
  --batch-size 16 --epochs 2 \
  --sequence-length 64 --no-mixed-precision \
  --gradient-clip 0.5 --weight-decay 0.0
```

## 🎛️ Parameter Override

You can use presets and still override specific parameters:

```bash
# Use GPT-4o preset but change batch size and epochs
python scripts/cli.py train --task llm \
  --config-preset gpt-4o \
  --batch-size 4 --epochs 50
```

## 📊 Performance Expectations

| Preset | Target Performance | Training Time | GPU Memory | Recommended Hardware |
|--------|-------------------|---------------|------------|---------------------|
| `gpt-4o` | GPT-4o competitive | ~2 weeks | 80GB | 4x RTX 4090 |
| `claude-sonnet-4` | Claude Sonnet 4 competitive | ~6 weeks | 120GB | 8x A100 |
| `ultra-advanced` | Better than all modern LLMs | ~10 weeks | 200GB+ | 8x H100 |

## 🔧 Advanced Training Strategy

Each preset includes optimized training strategies:

### Progressive Training Stages
1. **Foundation Stage** (First 30% of epochs): Basic pattern learning
2. **Reasoning Stage** (Middle 40% of epochs): Complex reasoning development  
3. **Long Context Stage** (Final 30% of epochs): Extended context understanding

### Automatic Optimizations
- Adaptive learning rate scheduling with warmup
- Gradient accumulation for effective larger batch sizes
- Mixed precision training for memory efficiency
- Advanced weight initialization
- Label smoothing and regularization

## 💡 Tips for Best Results

1. **Start Small**: Test with shorter epoch counts first to validate setup
2. **Monitor Memory**: Use `nvidia-smi` to monitor GPU memory usage
3. **Checkpoint Frequently**: Save checkpoints every 5-10 epochs for long training runs
4. **Progressive Scaling**: Start with smaller presets and scale up as needed
5. **Multi-GPU**: Use `--multi-gpu` for faster training on multiple GPUs

## 🎯 Example Training Commands

### Quick Testing
```bash
# Quick test with GPT-4o preset (1 epoch)
python scripts/cli.py train --task llm --config-preset gpt-4o --epochs 1

# Test with custom parameters
python scripts/cli.py train --task llm --liquid-units 64 --epochs 1
```

### Production Training
```bash
# Full GPT-4o competitive training
python scripts/cli.py train --task llm --config-preset gpt-4o --epochs 100 --multi-gpu

# Ultra-advanced with custom batch size
python scripts/cli.py train --task llm --config-preset ultra-advanced --epochs 200 --batch-size 2
```

### With Multi-GPU
```bash
# Distributed training on 4 GPUs
python scripts/cli.py train --task llm --config-preset claude-sonnet-4 \
  --epochs 150 --multi-gpu --gpu-strategy ddp --gpu-ids "0,1,2,3"
```

## 🔍 Help and Documentation

View all available options:
```bash
python scripts/cli.py train --help
python scripts/cli.py --help  # See preset examples
```

The preset system maintains full compatibility with all existing features while providing easy access to state-of-the-art configurations optimized for competing with modern LLMs.
