# GradScaler Mixed Precision Training Fix

## Problem Summary

Training crashed after completing the first epoch with the error:
```
AssertionError: No inf checks were recorded for this optimizer.
```

This occurred at line 2258 in `src/core/main.py` during the `scaler.step(self.optimizer)` call.

## Root Cause

The issue was in the `train_epoch()` method's handling of gradient accumulation with PyTorch's `GradScaler` for mixed precision training:

1. **Duplicate Code Block**: There were TWO identical code blocks (lines 2201-2268) attempting to handle "remaining accumulated gradients" after the training loop
2. **Missing Backward Check**: The code tried to call `scaler.step()` without verifying that a backward pass had actually occurred
3. **Scaler State Corruption**: When `scaler.step()` is called without a preceding `scaler.scale(loss).backward()`, the scaler has no recorded inf checks, causing the assertion error

### The Problem Flow

```python
for batch_idx, batch in enumerate(progress_bar):
    # ... forward pass with autocast ...
    self.scaler.scale(loss).backward()  # Records inf checks
    
    if (batch_idx + 1) % self.accumulation_steps == 0:
        self.scaler.unscale_(self.optimizer)  # Uses inf checks
        self.scaler.step(self.optimizer)      # Requires inf checks
        self.scaler.update()                  # Resets state

# After loop: duplicate handling block tries to step AGAIN
if accumulated_loss > 0:
    # No backward() was called here!
    self.scaler.step(self.optimizer)  # ❌ ERROR: No inf checks!
```

## The Fix

**File**: `src/core/main.py`  
**Lines Modified**: 2203-2268

### Changes Made

1. **Removed Duplicate Code**: Eliminated the second duplicate handling block (lines 2229-2268)
2. **Added Proper Guard**: Modified the remaining gradient handler to only execute if:
   - `accumulated_loss > 0` (gradients were accumulated)
   - `(batch_idx + 1) % self.accumulation_steps != 0` (not already processed in loop)

### Before (Buggy Code)

```python
# Handle remaining accumulated gradients
if accumulated_loss > 0:
    if self.config.gradient_clip > 0:
        if self.config.mixed_precision and self.scaler:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(...)
        gradient_norm_sum += grad_norm.item()
    
    if self.config.mixed_precision and self.scaler:
        self.scaler.step(self.optimizer)  # ❌ May fail!
        self.scaler.update()
    else:
        self.optimizer.step()
    
    # ... [40 lines of duplicate code] ...

# DUPLICATE: Handle remaining accumulated gradients AGAIN
if accumulated_loss > 0:
    # ... [exact same code as above] ...
```

### After (Fixed Code)

```python
# Handle remaining accumulated gradients after loop completes
# Only process if we actually accumulated gradients (check if backward was called)
if accumulated_loss > 0 and (batch_idx + 1) % self.accumulation_steps != 0:
    if self.config.gradient_clip > 0:
        if self.config.mixed_precision and self.scaler:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(...)
        gradient_norm_sum += grad_norm.item()
    
    if self.config.mixed_precision and self.scaler:
        self.scaler.step(self.optimizer)  # ✅ Only if backward was called
        self.scaler.update()
    else:
        self.optimizer.step()
    
    self.optimizer.zero_grad()
    self._update_ema()
    total_loss += accumulated_loss
    num_batches += 1
# No duplicate code!
```

## Key Improvements

1. **Proper State Management**: Only calls `scaler.step()` when gradients were actually computed
2. **Code Deduplication**: Removed 40+ lines of duplicate code
3. **Correct Guard Condition**: Checks `(batch_idx + 1) % self.accumulation_steps != 0` to ensure we're not double-processing

## When This Fix Applies

This fix is critical when:
- ✅ Using `mixed_precision=True` with `GradScaler`
- ✅ Using gradient accumulation (`accumulation_steps > 1`)
- ✅ Training for multiple epochs with continual learning
- ✅ Using STDP, meta-plasticity, or experience replay
- ✅ Training completes first epoch successfully but crashes at epoch boundary

## Testing Verification

After this fix, training should:
1. ✅ Complete first epoch without crashes
2. ✅ Transition smoothly between epochs
3. ✅ Properly handle gradient accumulation boundaries
4. ✅ Maintain correct scaler state throughout training
5. ✅ Work with all advanced features (Mamba, STDP, continual learning)

## Related Issues

- **Previous Fix**: Fixed `validate()` unpacking bug (returned 3 values, expected 2)
- **Context**: Training with 30 epochs, 5 continual learning tasks, Mamba integration, STDP, meta-plasticity
- **Symptom**: Training crashed after 7h 5m (batch 1131/1131) at epoch completion

## Command to Resume Training

```bash
python scripts/cli.py train --task llm --epochs 30 \
  --continual-learning --num-tasks 5 \
  --use-stdp --use-meta-plasticity --use-replay \
  --use-mamba --integration-mode bidirectional \
  --spike-to-mamba-method temporal --mamba-expand 2 \
  --mamba-d-state 32 --mamba-d-conv 4 \
  --use-adaptive-gating --num-gate-heads 8 \
  --use-cross-attention --cross-attn-heads 16
```

Training will resume from the saved checkpoint.

---

**Fix Applied**: October 20, 2025  
**Status**: ✅ Complete - Ready for training
