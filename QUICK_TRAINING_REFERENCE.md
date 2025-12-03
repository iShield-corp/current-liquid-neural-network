# 🚀 Quick Training Reference

## Recommended Commands

### Production Training (BEST - All Fixes Included)
```bash
# Small model, WikiText-103, 30 epochs
python train.py llm_production --epochs 30 --dataset wikitext103

# Medium model, combined datasets, 50 epochs  
python train.py llm_production \
  --model-size medium \
  --dataset combined \
  --epochs 50 \
  --batch-size 16

# Resume from checkpoint
python train.py llm_production \
  --resume models/best_model.pt \
  --epochs 100
```

### Via CLI (Same Thing)
```bash
python scripts/cli.py train \
  --task llm \
  --production-mode \
  --dataset wikitext103 \
  --epochs 30
```

## Model Size Presets

| Size   | Params | RAM  | Best Dataset      |
|--------|--------|------|-------------------|
| tiny   | ~20M   | 2GB  | WikiText-2        |
| small  | ~97M   | 4GB  | WikiText-103      |
| medium | ~250M  | 8GB  | Combined datasets |
| large  | ~500M  | 16GB | Combined datasets |

## Dataset Options

| Dataset   | Tokens | Quality | Use For              |
|-----------|--------|---------|----------------------|
| wikitext2 | 2M     | Good    | Tiny models only     |
| wikitext103 | 100M | Good    | RECOMMENDED         |
| combined  | 200M+  | Best    | Large models         |

## Common Options

```bash
--epochs 30                    # Training epochs
--batch-size 8                 # Batch size
--learning-rate 5e-5           # Learning rate
--gradient-clip 0.5            # Gradient clipping
--seq-length 256               # Sequence length
--model-size small             # Model preset
--dataset wikitext103          # Dataset choice
--checkpoint-dir ./models      # Save location
--patience 15                  # Early stopping
--accumulation-steps 4         # Gradient accumulation
--resume path/to/model.pt      # Resume training
--no-mixed-precision           # Disable FP16 (if issues)
```

## What to Expect

### First Epoch
```
Epoch 1/30: 100%|██████████| 304/304 [30:38<00:00]
  🔥 Train Loss: 10.1348
  ✅ Val Loss: 10.0521
  🎯 Val Accuracy: 0.15%
  📈 Grad Norm: 2.208
  📚 LR: 5.00e-05
```

### Healthy Training Signs
- ✅ LR shows `5.00e-05` (NOT `0.00e+00`)
- ✅ Loss decreases each epoch
- ✅ Grad norm between 0.5-5.0
- ✅ GPU memory stable

### Warning Signs
- ❌ LR shows `0.00e+00` → Use production mode!
- ❌ Loss stuck/increasing → Check dataset size
- ❌ Grad norm = 0 → Gradients vanished (shouldn't happen with fixes)
- ❌ Grad norm > 100 → Gradient explosion (reduce LR)

## Troubleshooting

### Loss Not Decreasing
```bash
# 1. Run diagnostics
python scripts/diagnose_training.py

# 2. Try smaller model
python train.py llm_production --model-size tiny --epochs 5

# 3. Check dataset
# WikiText-2 TOO SMALL for models > 10M params
# Use WikiText-103 instead
```

### Out of Memory
```bash
# Reduce batch size + use accumulation
python train.py llm_production \
  --batch-size 4 \
  --accumulation-steps 4  # Effective batch = 16
```

### lr=0.00e+00 Still Showing
```bash
# You're using old trainer! Switch to production:
python train.py llm_production  # NOT train.py llm
```

## Output Files

After training, you'll have:
```
models/
├── training_config.json      # Your exact config
├── best_model.pt             # Best validation loss
├── checkpoint_epoch_10.pt    # Periodic saves
├── final_model.pt            # Last epoch
```

## Integration

### From Python Code
```python
from src.core.main import LiquidSpikingTrainer, LiquidSpikingNetwork, create_llm_config

config = create_llm_config('gpt2')
model = LiquidSpikingNetwork(config)
trainer = LiquidSpikingTrainer(model, config)  # Uses fixed training!
trainer.train(train_loader, val_loader, num_epochs=30)
```

### From GUI
```python
import subprocess

subprocess.run([
    'python', 'scripts/train_production.py',
    '--epochs', '30',
    '--dataset', 'wikitext103'
])
```

## Quick Checks

### Verify Fixes Applied
```bash
python scripts/verify_fixes.py  # Should show "ALL FIXES IN PLACE"
```

### Test LR Scheduler
```bash
python scripts/test_lr_scheduler.py  # Should show LR never 0
```

### Run Full Diagnostic
```bash
python scripts/diagnose_training.py  # Should show "Model CAN memorize"
```

## Performance

- **Speed**: ~6 sec/batch (small model, RTX 3090)
- **Memory**: ~2-4 GB GPU (small model)
- **Time**: ~30 min per epoch (WikiText-103, small model)

## Summary

**✅ RECOMMENDED**:
```bash
python train.py llm_production --epochs 30 --dataset wikitext103
```

This command:
- Uses ALL fixes automatically
- Saves best model
- Has early stopping
- Shows proper LR
- Actually learns!

**❌ DON'T USE**:
```bash
python train.py llm  # Old trainer, has lr=0 bug
```

---

**Need help?** Check:
- `PRODUCTION_TRAINING_GUIDE.md` - Full guide
- `TRAINING_FIXES_SUMMARY.md` - Technical details
- `scripts/diagnose_training.py` - Diagnostic tool
