# Memory Optimization Guide for RTX 3080 (10.7GB)

## ✨ NEW: Adaptive Memory Manager

**The memory manager now automatically detects your GPU VRAM and sets optimal thresholds!**

When you start training, you'll see:
```
🎯 GPU Memory Configuration:
   Found 1 CUDA device(s)

   GPU 0: NVIDIA GeForce RTX 3080
      Total VRAM: 10922 MB
      Currently allocated: 0 MB (0.0%)
      Free: 10922 MB
      Safe threshold: 9284 MB (85% of total)
      Reserved buffer: 1638 MB (15%)

🎯 GPU 0 Memory Manager initialized:
   Total VRAM: 10922 MB
   Safe threshold: 9284 MB (85% of total)
   Reserved buffer: 1638 MB (15%)
```

**This means:**
- Your GPU has 10.9GB total VRAM
- Training will trigger cleanup at 9.3GB (85% usage)
- 1.6GB is reserved as safety buffer
- **No manual configuration needed!**

---

## Problem
Your 496M parameter model with current settings exceeds GPU memory:
- **Memory spikes**: 13GB → 22GB (crashes at 10.7GB limit)
- **Root cause**: batch_size=16 + seq_length=256 + bidirectional Mamba = too large

## IMMEDIATE SOLUTION - Resume Training with These Settings:

```bash
python3 scripts/cli.py train --task llm --tokenizer o200k --epochs 50 \
  --batch-size 4 \
  --sequence-length 128 \
  --accumulation-steps 4 \
  --checkpoint-activations \
  --num-layers 8 --liquid-units 384 \
  --spiking-units 192 --hidden-dim 768 --num-attention-heads 12 --use-mamba \
  --integration-mode bidirectional --mamba-d-state 32 --gradient-clip 1.0 \
  --learning-rate 0.0001 --use-cross-attention --use-adaptive-gating --use-stdp \
  --use-meta-plasticity --dataset combined \
  --combined-datasets programming,wikitext103,bookcorpus,openwebtext \
  --mixed-precision --save-interval 5 --output-dir ./models/best_model --production-mode
```

### Key Changes:
1. **`--batch-size 4`** (was 16): Reduces memory by 4x
2. **`--sequence-length 128`** (was 256): Reduces memory by 2x  
3. **`--accumulation-steps 4`**: Maintains effective batch size of 16 (4×4)
4. **`--checkpoint-activations`**: Gradient checkpointing saves ~40% memory

**Expected memory usage**: ~6-8GB (safe for your 10.7GB GPU)

## Memory Reduction Techniques (Applied)

### 1. Gradient Checkpointing (NEW)
- **What**: Recomputes activations during backward pass instead of storing them
- **Benefit**: 30-40% memory reduction
- **Cost**: ~20% slower training
- **Enabled by default** in production script

### 2. Gradient Accumulation
- **What**: Accumulate gradients over multiple small batches before updating
- **Benefit**: Effective batch size = batch_size × accumulation_steps
- **Example**: batch_size=4, accumulation_steps=4 → effective batch=16

### 3. Reduced Sequence Length
- **Impact**: Memory scales with seq_length²  for attention
- **128 vs 256**: Saves 75% attention memory
- **Quality trade-off**: Minimal for most tasks

### 4. Mixed Precision (Already Enabled)
- **What**: FP16 for most operations, FP32 for critical ops
- **Benefit**: 50% memory reduction
- **Already active** in your training

## Memory Monitoring During Training

Watch for these patterns:
```
✅ HEALTHY:
INFO: Memory cleanup #X: GPU memory: 6000-8000MB
(stable, not growing)

⚠️ WARNING:
INFO: Memory cleanup #X: GPU memory: 9000-10000MB
(approaching limit, reduce batch size)

❌ CRITICAL:
INFO: Memory cleanup #X: GPU memory: 10500MB+
(will crash soon, stop and reduce settings)
```

## Alternative Configurations

### Option A: Ultra-Safe (Slowest, 100% stable)
```bash
--batch-size 2 --sequence-length 128 --accumulation-steps 8 --checkpoint-activations
# Memory: ~5-6GB, Speed: 0.5x
```

### Option B: Balanced (Recommended)
```bash
--batch-size 4 --sequence-length 128 --accumulation-steps 4 --checkpoint-activations
# Memory: ~6-8GB, Speed: 1x
```

### Option C: Aggressive (Faster, needs monitoring)
```bash
--batch-size 6 --sequence-length 128 --accumulation-steps 3 --checkpoint-activations
# Memory: ~8-9GB, Speed: 1.3x
```

### Option D: Maximum Speed (No gradient checkpointing)
```bash
--batch-size 4 --sequence-length 128 --accumulation-steps 4 --no-checkpoint-activations
# Memory: ~9-10GB, Speed: 1.2x (risky, monitor closely)
```

## Model Size vs GPU Memory

Your current model (496M params):
- **FP32 weights**: 1,984 MB
- **FP16 weights**: 992 MB  
- **Optimizer states** (AdamW): 3,968 MB (FP32)
- **Gradients**: 1,984 MB
- **Activations** (varies): 2,000-15,000 MB depending on batch/seq_length

**Total minimum**: ~9GB just for model + optimizer  
**With activations**: 11-24GB (THIS is why you OOM)

## What Changed in Code

### 1. Production Script Defaults (`train_production.py`)
```python
# OLD (unsafe for 10.7GB GPU):
parser.add_argument('--batch-size', default=8)
parser.add_argument('--seq-length', default=256)
parser.add_argument('--accumulation-steps', default=1)

# NEW (safe defaults):
parser.add_argument('--batch-size', default=4)
parser.add_argument('--seq-length', default=128)
parser.add_argument('--accumulation-steps', default=4)
parser.add_argument('--checkpoint-activations', default=True)
```

### 2. Model Architecture (`src/core/main.py`)
```python
# Added gradient checkpointing support:
if self.use_gradient_checkpointing and self.training:
    result = torch.utils.checkpoint.checkpoint(
        create_custom_forward(block),
        x, hidden_states[i],
        use_reentrant=False
    )
```

## Expected Training Speed

With new settings (batch=4, seq=128, checkpointing):
- **~110 seconds/batch** (similar to before)
- **8,384 batches/epoch** (unchanged)
- **~256 hours/epoch** (~10.7 days)
- **50 epochs** = ~535 days total

⚠️ **This is VERY long**. Consider:
1. Reducing to 10-20 epochs
2. Using smaller model (6 layers instead of 8)
3. Training on multiple GPUs if available

## Quick Reference: Resume Your Training

```bash
# Navigate to project
cd /home/sovr610/ssn-cfc
source nn/bin/activate

# Resume with safe settings
python3 scripts/cli.py train --task llm --tokenizer o200k --epochs 50 \
  --batch-size 4 --sequence-length 128 --accumulation-steps 4 \
  --checkpoint-activations --num-layers 8 --liquid-units 384 \
  --spiking-units 192 --hidden-dim 768 --num-attention-heads 12 \
  --use-mamba --integration-mode bidirectional --mamba-d-state 32 \
  --gradient-clip 1.0 --learning-rate 0.0001 --use-cross-attention \
  --use-adaptive-gating --use-stdp --use-meta-plasticity \
  --dataset combined \
  --combined-datasets programming,wikitext103,bookcorpus,openwebtext \
  --mixed-precision --save-interval 5 \
  --output-dir ./models/best_model --production-mode \
  --resume ./models/best_model/oom_checkpoint.pt
```

## Monitoring Memory in Real-Time

```bash
# In separate terminal, watch GPU memory:
watch -n 1 nvidia-smi

# Look for:
# - Memory usage staying under 9GB
# - No continuous growth
# - Stable after 5-10 batches
```

## If Still Getting OOM

1. **Reduce batch size further**: `--batch-size 2 --accumulation-steps 8`
2. **Reduce sequence length**: `--sequence-length 96`
3. **Reduce model size**: `--num-layers 6 --hidden-dim 512`
4. **Disable cross-attention**: Remove `--use-cross-attention` flag

## Success Indicators

✅ Training runs for 30+ batches without OOM  
✅ GPU memory stable at 6-8GB  
✅ Loss decreasing gradually  
✅ No memory cleanup warnings  

---

**Bottom line**: Your model is too large for a single RTX 3080 at the original settings. The new defaults are optimized for 10.7GB GPUs.
