# GradScaler Mixed Precision Training Fix

## Issue
**Error**: `AssertionError: No inf checks were recorded for this optimizer.`

**Location**: `src/core/main.py`, line 2205 in `train_epoch()` method

**Root Cause**: 
The `GradScaler.step()` method was being called without a corresponding backward pass through the scaler. This occurred in the "Handle remaining accumulated gradients" section where leftover gradients are processed after the main training loop completes.

## Problem Details

The training loop has two paths:
1. **Mixed Precision Path**: Uses `autocast()` and `scaler.scale(loss).backward()`
2. **Standard Precision Path**: Direct `loss.backward()` without scaler

The bug occurred because:
- When using standard precision (or when leftover gradients exist), the scaler was NOT used for backward pass
- However, the cleanup code unconditionally called `scaler.step()` and `scaler.update()`
- PyTorch's `GradScaler` requires that if you call `scaler.step()`, you must have previously called `scaler.scale(loss).backward()`
- This caused the assertion error: "No inf checks were recorded"

## Solution

Modified two instances of the "Handle remaining accumulated gradients" section to only use scaler methods when mixed precision is actually enabled:

### Before (Buggy Code):
```python
if accumulated_loss > 0:
    if self.config.gradient_clip > 0:
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(...)
    
    if self.scaler:  # BUG: Uses scaler even if mixed precision not used
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        self.optimizer.step()
```

### After (Fixed Code):
```python
if accumulated_loss > 0:
    if self.config.gradient_clip > 0:
        # Only unscale if using mixed precision
        if self.config.mixed_precision and self.scaler:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(...)
    
    # Only use scaler.step() if using mixed precision
    if self.config.mixed_precision and self.scaler:
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        self.optimizer.step()
```

## Key Changes

1. **Line 2157**: Added `self.config.mixed_precision` check before `scaler.unscale_()`
2. **Line 2164**: Added `self.config.mixed_precision` check before `scaler.step()`
3. **Line 2199**: Same fix applied to second occurrence

## Testing

Run the continual learning training command to verify:
```bash
python3 scripts/cli.py train --task llm --epochs 30 \
  --continual-learning --num-tasks 5 \
  --use-stdp --use-meta-plasticity --use-replay
```

**Expected Result**: Training should proceed without the GradScaler assertion error.

## Related Files

- `src/core/main.py` - Lines 2153-2176 and 2193-2216 (train_epoch method)

## Impact

- ✅ Fixes crash during continual learning training
- ✅ Maintains proper mixed precision training when enabled
- ✅ Correctly handles standard precision training path
- ✅ Preserves gradient accumulation functionality
- ✅ No impact on STDP, meta-plasticity, or continual learning features

## Date
October 18, 2025
