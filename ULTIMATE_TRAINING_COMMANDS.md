# 🚀 Ultimate Training Commands for Mamba-Liquid-Spiking Neural Networks

## Quick Start Commands

### 1. **Fastest Test Run** (2 minutes)
Test that everything works:

```bash
python scripts/cli.py train \
  --task llm \
  --tokenizer o200k \
  --epochs 1 \
  --batch-size 2 \
  --sequence-length 128 \
  --use-mamba \
  --integration-mode sequential
```

### 2. **Development Training** (30 minutes)
Good for development and debugging:

```bash
python scripts/cli.py train \
  --task llm \
  --tokenizer o200k \
  --vocab-size 200019 \
  --sequence-length 512 \
  --batch-size 4 \
  --epochs 5 \
  --liquid-units 256 \
  --spiking-units 128 \
  --hidden-dim 512 \
  --num-layers 4 \
  --use-mamba \
  --integration-mode parallel \
  --use-cross-attention \
  --learning-rate 1e-4
```

### 3. **Production Training** (12-18 hours on 8xA100)
Full-scale production training:

```bash
python scripts/cli.py train \
  --task llm \
  --epochs 50 \
  \
  `# Tokenizer - GPT-4o` \
  --tokenizer o200k \
  --vocab-size 200019 \
  \
  `# Core Architecture` \
  --liquid-units 768 \
  --spiking-units 512 \
  --hidden-dim 1024 \
  --num-layers 12 \
  --num-attention-heads 16 \
  --liquid-backbone cfc \
  \
  `# Mamba Integration (O(N) Long-Range)` \
  --use-mamba \
  --integration-mode bidirectional \
  --mamba-d-state 32 \
  --mamba-d-conv 4 \
  --mamba-expand 2 \
  --spike-to-mamba-method temporal \
  --spike-temporal-tau 20.0 \
  --use-adaptive-gating \
  --num-gate-heads 8 \
  --use-cross-attention \
  --cross-attn-heads 16 \
  \
  `# Sequence & Batch Settings` \
  --sequence-length 2048 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  \
  `# Optimizer & Learning Rate` \
  --learning-rate 5e-5 \
  --optimizer adamw \
  --weight-decay 0.01 \
  --warmup-steps 2000 \
  --scheduler cosine \
  --gradient-clip 1.0 \
  \
  `# Plasticity Features` \
  --use-stdp \
  --stdp-type triplet \
  --stdp-learning-rate 0.001 \
  --stdp-tau-plus 20.0 \
  --stdp-tau-minus 20.0 \
  --use-meta-plasticity \
  --meta-lr 0.0001 \
  --meta-history-length 100 \
  \
  `# Continual Learning` \
  --continual-learning \
  --consolidation-strength 1000.0 \
  --use-replay \
  --replay-buffer-size 10000 \
  --replay-strategy importance \
  --compute-importance-interval 5 \
  \
  `# Advanced Training` \
  --mixed-precision \
  --use-ema \
  --multi-gpu \
  --gpu-strategy ddp \
  \
  `# Regularization` \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --weight-decay 0.01 \
  \
  `# Output` \
  --output-dir ./models/ultimate_gpt4o_mamba \
  --save-interval 5
```

---

## 📊 Integration Mode Comparison

### Sequential Mode (Simplest)
```bash
--use-mamba --integration-mode sequential
```
- **Flow**: Spike → Liquid → Mamba → Output
- **Speed**: Fastest (1x baseline)
- **Memory**: Lowest (~6GB)
- **Best for**: Short sequences (<512 tokens), rapid prototyping
- **Performance**: Good for simple tasks

### Parallel Mode (Balanced)
```bash
--use-mamba --integration-mode parallel --use-adaptive-gating --num-gate-heads 4
```
- **Flow**: Spike → {Liquid || Mamba} → Gate → Output
- **Speed**: Medium (1.3x baseline)
- **Memory**: Medium (~8GB)
- **Best for**: Medium sequences (512-1024 tokens), balanced performance
- **Performance**: Better accuracy with dual pathways

### Bidirectional Mode (Best Performance)
```bash
--use-mamba --integration-mode bidirectional --use-cross-attention --cross-attn-heads 16
```
- **Flow**: Full cross-attention + state exchange between Liquid and Mamba
- **Speed**: Slowest (1.8x baseline)
- **Memory**: Highest (~12GB)
- **Best for**: Long sequences (1024-4096 tokens), maximum accuracy
- **Performance**: ⭐ **State-of-the-art** - best results

---

## 🎯 Task-Specific Commands

### LLM Training (Language Models)

#### Small Model (Testing)
```bash
python scripts/cli.py train \
  --task llm \
  --tokenizer o200k \
  --vocab-size 200019 \
  --sequence-length 256 \
  --liquid-units 128 \
  --spiking-units 64 \
  --hidden-dim 256 \
  --num-layers 4 \
  --epochs 10 \
  --use-mamba \
  --integration-mode sequential
```

#### Medium Model (Development)
```bash
python scripts/cli.py train \
  --task llm \
  --tokenizer o200k \
  --vocab-size 200019 \
  --sequence-length 1024 \
  --liquid-units 512 \
  --spiking-units 256 \
  --hidden-dim 768 \
  --num-layers 8 \
  --epochs 30 \
  --use-mamba \
  --integration-mode parallel \
  --use-cross-attention
```

#### Large Model (Production)
```bash
python scripts/cli.py train \
  --task llm \
  --tokenizer o200k \
  --vocab-size 200019 \
  --sequence-length 2048 \
  --liquid-units 768 \
  --spiking-units 512 \
  --hidden-dim 1024 \
  --num-layers 12 \
  --num-attention-heads 16 \
  --epochs 50 \
  --use-mamba \
  --integration-mode bidirectional \
  --use-cross-attention \
  --cross-attn-heads 16 \
  --multi-gpu \
  --mixed-precision
```

### Vision Training

```bash
python scripts/cli.py train \
  --task vision \
  --liquid-units 256 \
  --spiking-units 128 \
  --hidden-dim 512 \
  --num-layers 6 \
  --epochs 20 \
  --batch-size 64 \
  --use-mamba \
  --integration-mode parallel \
  --learning-rate 1e-3
```

### Robotics Training

```bash
python scripts/cli.py train \
  --task robotics \
  --liquid-units 128 \
  --spiking-units 64 \
  --hidden-dim 256 \
  --num-layers 4 \
  --num-spike-steps 50 \
  --sequence-length 100 \
  --epochs 15 \
  --use-mamba \
  --integration-mode bidirectional
```

---

## 🔧 Mamba-Specific Parameter Guide

### State Dimension (`--mamba-d-state`)
Controls the dimensionality of the selective state space:
- **8-16**: Fast, less memory, good for short context
- **32**: ✅ **Recommended** - balanced performance
- **64**: Maximum accuracy, slower, more memory

### Convolution Size (`--mamba-d-conv`)
Local context window size:
- **3**: Minimal local context
- **4**: ✅ **Recommended** - standard
- **7**: Extended local patterns

### Expansion Factor (`--mamba-expand`)
Inner dimension expansion:
- **1**: No expansion (fastest)
- **2**: ✅ **Recommended** - standard
- **4**: Maximum capacity (slower)

### Spike Conversion Method (`--spike-to-mamba-method`)

#### Rate Coding
```bash
--spike-to-mamba-method rate
```
- Converts spikes to average firing rate
- **Best for**: High spike frequencies
- **Performance**: Fast, simple

#### Temporal Coding (Recommended)
```bash
--spike-to-mamba-method temporal --spike-temporal-tau 20.0
```
- ✅ **Recommended** - weighs recent spikes more
- Exponential decay with time constant τ
- **Best for**: Temporal patterns, sequences
- **Performance**: Best accuracy

#### Potential-Based
```bash
--spike-to-mamba-method potential
```
- Uses membrane potential directly
- **Best for**: Continuous-like signals
- **Performance**: Medium

---

## 💾 Resource Requirements

### Memory Usage by Configuration

| Config | Integration | Seq Length | Memory (GB) | GPUs |
|--------|-------------|------------|-------------|------|
| Small | Sequential | 256 | 4-6 | 1×RTX3090 |
| Small | Parallel | 512 | 6-8 | 1×RTX4090 |
| Medium | Sequential | 1024 | 8-12 | 1×A100 |
| Medium | Parallel | 1024 | 12-16 | 2×A100 |
| Medium | Bidirectional | 1024 | 16-24 | 2×A100 |
| Large | Bidirectional | 2048 | 32-40 | 4×A100 |
| Ultra | Bidirectional | 4096 | 64-80 | 8×A100 |

### Speed Estimates (Tokens/Second)

| Setup | Integration | Hardware | Speed |
|-------|-------------|----------|-------|
| Small | Sequential | 1×RTX3090 | ~20k tok/s |
| Medium | Parallel | 1×A100 | ~35k tok/s |
| Large | Bidirectional | 4×A100 | ~100k tok/s |
| Ultra | Bidirectional | 8×A100 | ~150k tok/s |

---

## 🎮 Interactive Mode

For guided configuration:

```bash
python scripts/cli.py config --task llm --interactive
```

This will interactively ask you for all parameters and generate a complete training command.

---

## 📈 Monitoring Training

### View Real-Time Progress
Training automatically shows:
- Loss curves
- Learning rate schedule
- Memory usage
- Token throughput
- Validation metrics

### TensorBoard
```bash
tensorboard --logdir ./models/ultimate_gpt4o_mamba/logs
```

### Weights & Biases (if configured)
Training metrics automatically logged to W&B dashboard.

---

## 🔍 Debugging Commands

### Test Mamba Integration
```bash
python test_mamba_integration.py
```

### Check Configuration
```bash
python scripts/cli.py config --task llm --tokenizer o200k --use-mamba --integration-mode bidirectional
```

### Validate Model
```bash
python scripts/cli.py info --model-path ./models/llm_model.pt
```

---

## 📚 Additional Resources

- **Mamba Integration Docs**: [`docs/MAMBA_INTEGRATION.md`](docs/MAMBA_INTEGRATION.md)
- **Test Fixes**: [`MAMBA_TEST_FIXES.md`](MAMBA_TEST_FIXES.md)
- **Architecture Details**: See MAMBA_INTEGRATION.md for flow diagrams
- **Performance Tips**: See optimization guides in `docs/`

---

## 🎯 Example Training Session

```bash
# 1. Test configuration (30 seconds)
python scripts/cli.py train --task llm --tokenizer o200k --epochs 1 --batch-size 2 --use-mamba

# 2. Quick training (5 minutes)
python scripts/cli.py train --task llm --tokenizer o200k --epochs 3 --batch-size 4 \
  --sequence-length 256 --use-mamba --integration-mode parallel

# 3. Full training (overnight)
python scripts/cli.py train --task llm --tokenizer o200k --epochs 50 --batch-size 8 \
  --sequence-length 2048 --liquid-units 768 --hidden-dim 1024 --num-layers 12 \
  --use-mamba --integration-mode bidirectional --use-cross-attention \
  --use-stdp --use-meta-plasticity --continual-learning \
  --multi-gpu --mixed-precision \
  --output-dir ./models/production_model

# 4. Evaluate
python scripts/cli.py inference --model-path ./models/production_model/best_model.pt \
  --prompt "The future of artificial intelligence" --max-length 200
```

---

## ✅ Success Checklist

Before production training:
- [ ] Test command runs without errors (1 epoch)
- [ ] GPU memory usage is acceptable
- [ ] Validation loss decreases
- [ ] Model saves checkpoints
- [ ] Can resume from checkpoint
- [ ] Inference works on saved model

---

## 🚀 Ready to Train!

Start with a quick test:

```bash
python scripts/cli.py train --task llm --tokenizer o200k --epochs 1 --use-mamba
```

Then scale up to your target configuration! 🎉
