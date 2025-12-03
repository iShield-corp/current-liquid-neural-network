# CLI Training with Fixes - Quick Reference

## Updated CLI Commands

The CLI now supports the training fixes with the `--use-fixes` flag!

### ✅ YES - Fixes ARE Applied (Recommended)

```bash
# Train LLM with fixes (RECOMMENDED for zero-learning issues)
python scripts/cli.py train --task llm --use-fixes --epochs 10

# Train with fixes + diagnostics
python scripts/cli.py train --task llm --use-fixes --diagnose --epochs 10

# Full example with all options
python scripts/cli.py train \
    --task llm \
    --use-fixes \
    --epochs 20 \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --dataset wikitext103 \
    --output-dir ./models

# Convenience wrapper (uses fixes automatically)
python train.py llm_fixed --epochs 10
```

### ❌ NO - Fixes NOT Applied (Default behavior)

```bash
# Standard training WITHOUT fixes (may experience zero-learning)
python scripts/cli.py train --task llm --epochs 10

# This is equivalent to old behavior
python train.py llm
```

---

## What `--use-fixes` Does

When you add `--use-fixes` to your CLI command, it automatically:

1. ✅ **Replaces spiking encoders** with `FixedSpikingEncoder`
   - Surrogate gradient slope: 5 → 25 (5x stronger)
   - Xavier initialization for better gradient flow
   - Proper membrane potential initialization

2. ✅ **Upgrades spike decoder** to `ImprovedSpikeDecoder`
   - Multi-path decoding (rate + temporal)
   - Learnable temperature scaling
   - Better gradient propagation

3. ✅ **Re-initializes output head**
   - Smaller weights (std=0.01 instead of 0.02)
   - Prevents overly large initial predictions

4. ✅ **Adds warmup scheduler**
   - 2 epochs of linear warmup
   - Then cosine decay
   - Stabilizes early training

5. ✅ **Enables gradient monitoring**
   - Real-time gradient health checks
   - Automatic warning if gradients vanish/explode
   - Diagnosis when training stalls

---

## Examples

### Quick Test with Fixes
```bash
python scripts/cli.py train \
    --task llm \
    --use-fixes \
    --epochs 5 \
    --batch-size 4 \
    --dataset wikitext2
```

### Production Training with Fixes
```bash
python scripts/cli.py train \
    --task llm \
    --use-fixes \
    --epochs 30 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --dataset wikitext103 \
    --gradient-clip 0.5 \
    --output-dir ./models/with_fixes
```

### With Diagnostics (Check Gradient Health)
```bash
python scripts/cli.py train \
    --task llm \
    --use-fixes \
    --diagnose \
    --epochs 10
```

### Compare With/Without Fixes
```bash
# Without fixes (may show flat loss)
python scripts/cli.py train --task llm --epochs 3 --output-dir ./models/no_fixes

# With fixes (should show improvement)
python scripts/cli.py train --task llm --use-fixes --epochs 3 --output-dir ./models/with_fixes
```

---

## Expected Output Differences

### Without `--use-fixes` (Old Behavior)
```
🧠 Initializing model...
🖥️  Using device: cuda
📚 Loading dataset: wikitext103
📊 Training samples: 90,000
📊 Validation samples: 10,000

Epoch 1/10: Train Loss=12.0392, Val Loss=12.0673, Val Acc=0.0000
Epoch 2/10: Train Loss=12.0392, Val Loss=12.0673, Val Acc=0.0000  ❌ No change
Epoch 3/10: Train Loss=12.0392, Val Loss=12.0673, Val Acc=0.0000  ❌ No change
```

### With `--use-fixes` (New Behavior)
```
🧠 Initializing model...
🖥️  Using device: cuda
🔧 Applying training fixes for improved gradient flow...
✅ Training fixes applied!
   • Surrogate gradient slope: 5 → 25
   • Xavier weight initialization
   • Multi-path spike decoder
   • Optimized output head
📚 Loading dataset: wikitext103
📊 Training samples: 90,000
📊 Validation samples: 10,000
📈 Added warmup scheduler (180 steps)
📊 Gradient health monitoring enabled

Epoch 1/10: Train Loss=11.7234, Val Loss=11.8901, Val Acc=0.0145 [✓ Grad: 0.001234]
Epoch 2/10: Train Loss=10.1456, Val Loss=10.3234, Val Acc=0.0823 [✓ Grad: 0.002156]  ✅
Epoch 3/10: Train Loss=8.6789, Val Loss=9.0123, Val Acc=0.1567 ⭐ [✓ Grad: 0.001987]  ✅
```

---

## All Available Training Commands

### Option 1: Direct CLI with Fixes
```bash
python scripts/cli.py train --task llm --use-fixes
```

### Option 2: Dedicated Fixes Script
```bash
python scripts/train_with_fixes.py --epochs 10
```

### Option 3: Wrapper Command
```bash
python train.py llm_fixed --epochs 10
```

**All three methods apply the same fixes!** Choose whichever is most convenient.

---

## When to Use `--use-fixes`

### ✅ Use `--use-fixes` when:
- You're experiencing zero-learning (flat loss)
- Accuracy remains at 0% after multiple epochs
- Loss stuck at ~12.04 (≈ ln(vocab_size))
- You want stable, reliable training
- Training a new model from scratch
- **RECOMMENDED for all new training runs**

### ❌ Don't use `--use-fixes` when:
- Continuing training from a checkpoint (model already has different architecture)
- Experimenting with the original architecture
- Comparing with baseline results

---

## Additional Flags

```bash
# Full list of fix-related flags
--use-fixes           # Apply all gradient flow fixes (RECOMMENDED)
--diagnose            # Run diagnostics before training (requires --use-fixes)
--learning-rate 5e-5  # Recommended LR when using fixes
--gradient-clip 0.5   # Recommended gradient clipping
--batch-size 4        # Smaller batches for stability
```

---

## Troubleshooting

### "Training fixes module not available"
```bash
# Make sure the fixes file exists
ls -l src/training/training_fixes.py

# If missing, you need to create it first
# (Refer to ZERO_LEARNING_FIXES_SUMMARY.md)
```

### Fixes applied but still no learning
```bash
# Try even lower learning rate
python scripts/cli.py train --task llm --use-fixes --learning-rate 1e-5

# Use smaller model
python scripts/cli.py train --task llm --use-fixes --num-layers 2 --hidden-dim 256

# Run with diagnostics
python scripts/cli.py train --task llm --use-fixes --diagnose
```

---

## Summary

- **Default CLI behavior**: NO fixes applied
- **To apply fixes**: Add `--use-fixes` flag
- **Recommended**: Always use `--use-fixes` for new training
- **Alternative**: Use `python train.py llm_fixed` (applies fixes automatically)

**Start here**:
```bash
python scripts/cli.py train --task llm --use-fixes --epochs 10
```
