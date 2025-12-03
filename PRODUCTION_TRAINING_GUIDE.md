# Production Training Guide

## 🚀 Quick Start (RECOMMENDED)

The production training script includes all critical fixes and best practices:

```bash
# Simple production training (uses all fixes automatically)
python train.py llm_production --epochs 10 --dataset wikitext103

# With custom configuration
python train.py llm_production \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --dataset wikitext103 \
  --model-size small

# Resume from checkpoint
python train.py llm_production \
  --resume models/checkpoint_epoch_5.pt \
  --epochs 30
```

## 🔧 What Was Fixed

### 1. Trainer Class Fixes (src/core/main.py)

**Bug #1: LR Scheduler Timing**
- **Problem**: `scheduler.step()` called AFTER validation, causing LR=0 at epoch 0
- **Fix**: Move scheduler.step() to BEFORE training, skip on first epoch
- **Impact**: LR now correctly set from the start (5e-5 instead of 0)

**Bug #2: Memory Cleanup**
- **Problem**: `del data, targets, outputs, loss` inside training loop caused crashes
- **Fix**: Use memory manager cleanup without explicit tensor deletion
- **Impact**: Stable training without memory errors

**Bug #3: Progress Bar LR Display**
- **Problem**: Displayed LR from wrong timing, showed 0 even when LR was correct
- **Fix**: Capture LR before scheduler step for accurate logging
- **Impact**: Accurate real-time monitoring

### 2. Production Training Features

✅ **Automatic Dataset Selection**
- WikiText-2 (2M tokens) - For tiny models only
- WikiText-103 (100M tokens) - RECOMMENDED for most models
- Combined datasets - Maximum quality

✅ **Checkpointing**
- Automatic checkpoints every N epochs
- Best model saving based on validation loss
- Resume training from any checkpoint

✅ **Early Stopping**
- Configurable patience (default: 10 epochs)
- Prevents overfitting
- Saves training time

✅ **Proper Validation**
- Validation every epoch
- Perplexity and accuracy metrics
- Best model tracking

✅ **Configuration Management**
- Model size presets (tiny/small/medium/large)
- Training config saved to JSON
- Full reproducibility

## 📊 Training Results

### Before Fixes
```
Epoch 1/3: Loss=12.0344, LR=0.00e+00, Grad=0.000000
Epoch 2/3: Loss=12.0355, LR=0.00e+00, Grad=0.000000
Epoch 3/3: Loss=12.0362, LR=0.00e+00, Grad=0.000000
❌ No learning - loss flat, gradients zero
```

### After Fixes (Production Training)
```
Epoch 1/3: Loss=12.0686 → 10.1348, LR=5.00e-05, Grad=2.208
Epoch 2/3: Loss=8.9837 → continuing to drop
✅ Model learning - loss decreasing, proper gradients
```

## 🎯 Usage Examples

### CLI Integration

```bash
# Production mode through CLI
python scripts/cli.py train \
  --task llm \
  --production-mode \
  --dataset wikitext103 \
  --epochs 30 \
  --model-size small

# Legacy mode (original trainer)
python scripts/cli.py train \
  --task llm \
  --dataset wikitext103 \
  --epochs 30
```

### Direct Script Usage

```bash
# Full control with production script
python scripts/train_production.py \
  --model-size medium \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --gradient-clip 0.5 \
  --dataset wikitext103 \
  --checkpoint-dir ./checkpoints \
  --patience 15 \
  --mixed-precision
```

### Model Size Presets

| Preset | Layers | Hidden | Input | Parameters | Best For |
|--------|--------|--------|-------|------------|----------|
| tiny   | 2      | 256    | 384   | ~20M       | Testing, tiny datasets |
| small  | 4      | 512    | 768   | ~97M       | WikiText-103, general use |
| medium | 8      | 768    | 1024  | ~250M      | Large datasets, serious training |
| large  | 12     | 1024   | 1536  | ~500M      | Maximum quality, research |

## 📁 Output Structure

```
models/
├── training_config.json      # Full configuration
├── best_model.pt             # Best validation loss
├── checkpoint_epoch_5.pt     # Periodic checkpoints
├── checkpoint_epoch_10.pt
├── final_model.pt            # Last epoch
└── interrupted_model.pt      # If Ctrl+C
```

## 🐛 Troubleshooting

### Issue: "lr=0.00e+00" during training
**Solution**: Use production training - this is fixed automatically

### Issue: Loss not decreasing
**Diagnostics**:
```bash
# Run with diagnostics
python scripts/diagnose_training.py
```

**Common causes**:
1. Dataset too small (use WikiText-103, not WikiText-2)
2. Learning rate too low (try 5e-5 or 1e-4)
3. Model too large for dataset (reduce --model-size)

### Issue: Out of memory
**Solutions**:
```bash
# Reduce batch size
--batch-size 4

# Use gradient accumulation
--accumulation-steps 4  # effective batch = 4 * 4 = 16

# Disable mixed precision
--no-mixed-precision
```

### Issue: Training too slow
**Solutions**:
```bash
# Smaller model
--model-size tiny

# Shorter sequences
--seq-length 128

# More workers
--num-workers 8
```

## 🔬 Advanced Configuration

### Custom Architecture

```bash
python scripts/train_production.py \
  --num-layers 6 \
  --hidden-dim 640 \
  --epochs 100 \
  --gradient-clip 1.0 \
  --learning-rate 1e-4
```

### Multi-Dataset Training

```bash
# Combined datasets (WikiText-103 + BookCorpus + OpenWebText)
python train.py llm_production \
  --dataset combined \
  --epochs 50 \
  --batch-size 16
```

### Resume Training

```bash
# Continue from best model
python scripts/train_production.py \
  --resume models/best_model.pt \
  --epochs 100  # will train up to epoch 100
```

## 📈 Monitoring Training

The production script provides detailed logging:

```
================================================================================
🚀 PRODUCTION TRAINING - Liquid-Spiking Neural Network
================================================================================

🖥️  Device: cuda
   GPU: NVIDIA GeForce RTX 3090
   Memory: 24.0 GB

⚙️  Model Configuration:
   Size preset: small
   Layers: 4
   Hidden dim: 512
   Total parameters: 96,791,030

📚 Loading wikitext103 dataset (train split)...
✅ Dataset ready: 235,104 examples

🎯 STARTING TRAINING
================================================================================

Epoch 1/10: 100%|██████████| 304/304 [30:38<00:00, 6.05s/it]
  🔥 Train Loss: 10.1348
  ✅ Val Loss: 10.0521
  🎯 Val Accuracy: 0.15%
  📈 Grad Norm: 2.208
  ⏱️  Time: 1840.3s
  📚 LR: 5.00e-05
  🏆 New best model! Validation loss: 10.0521

Epoch 2/10: 100%|██████████| 304/304 [30:12<00:00, 5.96s/it]
  🔥 Train Loss: 8.9837
  ✅ Val Loss: 8.8234
  🎯 Val Accuracy: 1.23%
  ...
```

## 🎓 Best Practices

1. **Start with WikiText-103**: Much better than WikiText-2
2. **Use production mode**: All fixes included automatically  
3. **Monitor first epoch**: Loss should decrease within 100 batches
4. **Check gradients**: Should be in range 0.1-10.0
5. **Save checkpoints**: Every 5-10 epochs
6. **Use early stopping**: Prevent overfitting (patience=10-15)
7. **Track validation**: Should follow train loss closely

## 🔗 Integration

### With GUI (gui.py)

The production training script can be called from the GUI:

```python
import subprocess

result = subprocess.run([
    'python', 'scripts/train_production.py',
    '--epochs', '30',
    '--dataset', 'wikitext103',
    '--model-size', 'small'
])
```

### With CLI (cli.py)

Use `--production-mode` flag:

```bash
python scripts/cli.py train \
  --task llm \
  --production-mode \
  --epochs 30
```

## 📝 Summary

**✅ USE THIS** for production LLM training:
```bash
python train.py llm_production --epochs 30 --dataset wikitext103
```

**❌ AVOID** the old training without fixes - it has the lr=0 bug and other issues.

**🔧 ALL FIXES INCLUDED**:
- ✅ Fixed LR scheduler (no more lr=0)
- ✅ Fixed gradient flow (slope=25)
- ✅ Fixed memory cleanup
- ✅ Proper checkpointing
- ✅ Early stopping
- ✅ Best model saving
- ✅ Full logging

Happy training! 🚀
