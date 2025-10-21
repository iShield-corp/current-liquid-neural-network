# GradScaler Mixed Precision Training Fix - FINAL

## Problem Summary

Training crashed after completing the first epoch (6h 50m, 1131/1131 batches) with the error:
```
AssertionError: No inf checks were recorded for this optimizer.
```

This occurred at line 2258 in `src/core/main.py` during the `scaler.step(self.optimizer)` call in the gradient cleanup section after the training loop.

## Root Cause

The issue was in the `train_epoch()` method's handling of **remaining accumulated gradients** after the main training loop completes:

1. **Incomplete Logic**: The code at line 2239-2260 tried to handle remaining gradients but the condition check was insufficient
2. **Scaler State Mismatch**: When all batches complete evenly (e.g., 1131 batches with accumulation_steps dividing evenly), `accumulated_loss > 0` can still be true (contains the last loss value), but gradients have already been stepped
3. **Double-Step Attempt**: The cleanup code tried to call `scaler.step()` on gradients that were already processed in the loop, causing the scaler to have no inf checks recorded

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

### Before (Buggy Code - Attempt 1)

```python
# Handle remaining accumulated gradients
if accumulated_loss > 0:
    if self.config.gradient_clip > 0:
        if self.config.mixed_precision and self.scaler:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(...)
        gradient_norm_sum += grad_norm.item()
    
    if self.config.mixed_precision and self.scaler:
        self.scaler.step(self.optimizer)  # ❌ FAILS! No inf checks!
        self.scaler.update()
    else:
        self.optimizer.step()
```

**Problem**: Checked `accumulated_loss > 0` but this doesn't guarantee gradients need stepping. When the loop completes with all batches processed, `accumulated_loss` may still be > 0 (last loss value), but gradients were already stepped in the loop!

### After (Fixed Code - FINAL)

```python
# Handle remaining accumulated gradients after loop completes
# ONLY step if we have unprocessed gradients from incomplete accumulation
remaining_batches = (batch_idx + 1) % self.accumulation_steps
if accumulated_loss > 0 and remaining_batches != 0:
    # We have gradients that were accumulated but not yet stepped
    # The backward() was already called in the loop via scaler
    
    if self.config.gradient_clip > 0:
        # Unscale before clipping (only for mixed precision)
        if self.config.mixed_precision and self.scaler:
            self.scaler.unscale_(self.optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.gradient_clip
        )
        gradient_norm_sum += grad_norm.item()
    
    # Step optimizer
    if self.config.mixed_precision and self.scaler:
        self.scaler.step(self.optimizer)  # ✅ SAFE! Gradients exist!
        self.scaler.update()
    else:
        self.optimizer.step()
    
    self.optimizer.zero_grad()
    self._update_ema()
    total_loss += accumulated_loss
    num_batches += 1
```

## Key Improvements

1. **Proper Remainder Check**: Uses `remaining_batches = (batch_idx + 1) % self.accumulation_steps` to detect incomplete accumulation
2. **Two-Condition Guard**: Checks BOTH `accumulated_loss > 0` AND `remaining_batches != 0` 
3. **Prevents Double-Step**: Only steps when gradients genuinely haven't been processed yet
4. **Clearer Logic**: Explicit variable name (`remaining_batches`) makes intent obvious

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
