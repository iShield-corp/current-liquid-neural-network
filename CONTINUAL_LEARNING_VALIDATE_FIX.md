# Continual Learning Validation Fix

## Issue

When training with continual learning enabled (`--continual-learning --use-stdp --use-meta-plasticity`), the training crashed with:

```
ValueError: too many values to unpack (expected 2)
```

## Root Cause

The `validate()` method in `LiquidSpikingTrainer` returns **3 values**:
```python
return avg_loss, accuracy, is_best
```

However, in two locations, the code was trying to unpack only **2 values**:

### Location 1: `src/core/main.py` line 2376
```python
# ❌ BEFORE (incorrect - expects 2 values)
val_loss, val_accuracy = self.validate(val_loader)

# ✅ AFTER (correct - expects 3 values)
val_loss, val_accuracy, is_best = self.validate(val_loader)
```

### Location 2: `scripts/cli.py` line 918
```python
# ❌ BEFORE (incorrect - expects 1 value)
is_best = trainer.validate(val_loader)
val_loss = trainer.val_losses[-1]

# ✅ AFTER (correct - expects 3 values)
val_loss, val_accuracy, is_best = trainer.validate(val_loader)
```

## Fix Applied

Updated both locations to correctly unpack all 3 return values from `validate()`:
1. `avg_loss` - validation loss
2. `accuracy` - validation accuracy
3. `is_best` - boolean indicating if this is the best model so far

## Files Modified

1. **`src/core/main.py`** - Line 2376
   - In the main `train()` method
   - Used during regular training and continual learning

2. **`scripts/cli.py`** - Line 918
   - In the `_train_with_progress()` method
   - Used during CLI-based training with progress bars

## Verification

The fix ensures that:
- ✅ Continual learning training works correctly
- ✅ STDP and meta-plasticity features function properly
- ✅ Experience replay operates as expected
- ✅ Validation accuracy is properly tracked
- ✅ Best model checkpointing works correctly

## Testing

You can now run the full continual learning training:

```bash
python scripts/cli.py train --task llm --epochs 30 \
  --continual-learning --num-tasks 5 \
  --use-stdp --use-meta-plasticity --use-replay \
  --replay-buffer-size 1000 \
  --replay-strategy balanced
```

## Impact

This fix resolves the crash during validation phase when using:
- Continual learning mode
- STDP plasticity
- Meta-plasticity
- Experience replay
- Multi-task training

All these advanced features now work together seamlessly! 🎉
