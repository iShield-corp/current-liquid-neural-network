# Adaptive GPU Memory Manager

## Overview
The memory manager now **automatically detects GPU VRAM** and adjusts cleanup thresholds based on your hardware.

## What Changed

### Auto-Detection on Startup
When training starts, the memory manager:
1. **Detects all GPUs** and their total VRAM
2. **Sets safe thresholds** at 85% of total VRAM
3. **Reserves 15% buffer** for system overhead and safety
4. **Logs configuration** so you know the limits

### Example Output on RTX 3080 (10.7GB):
```
🎯 GPU 0 Memory Manager initialized:
   Total VRAM: 10922 MB
   Safe threshold: 9284 MB (85% of total)
   Reserved buffer: 1638 MB (15%)
```

### Example Output on RTX 4090 (24GB):
```
🎯 GPU 0 Memory Manager initialized:
   Total VRAM: 24564 MB
   Safe threshold: 20879 MB (85% of total)
   Reserved buffer: 3685 MB (15%)
```

## Benefits

### 1. **Automatic Optimization**
- No manual threshold configuration needed
- Works optimally on any GPU (3080, 4090, A100, etc.)
- Scales from 8GB to 80GB GPUs automatically

### 2. **Better Memory Warnings**
Memory logs now show **percentage** and warnings:
```
✅ SAFE:   GPU: 6000MB (55%) 
⚠️ HIGH:   GPU: 8800MB (80%) ⚠️ HIGH
🔴 CRITICAL: GPU: 9900MB (90%) 🔴 CRITICAL
```

### 3. **GPU-Specific Reports**
```
=== Memory Usage Report ===
GPU 0: 7234.5MB / 10922MB (66.2%)
       Reserved: 8456.2MB, Threshold: 9284MB
```

## Technical Details

### Memory Manager Parameters

```python
SpikingMemoryManager(
    cleanup_threshold_mb=None,      # Auto-detect from GPU
    auto_cleanup=True,              # Automatic cleanup enabled
    memory_reserve_percent=0.15     # 15% safety buffer
)
```

### Threshold Calculation
```
Total VRAM: 10922 MB (RTX 3080)
Safe Threshold = Total × 0.85 = 9284 MB
Reserved Buffer = Total × 0.15 = 1638 MB

Cleanup triggers when: allocated > 9284 MB
```

### Multi-GPU Support
Each GPU gets its own threshold:
```python
# GPU 0: RTX 3080 (10.7GB) → threshold: 9284 MB
# GPU 1: RTX 4090 (24GB)   → threshold: 20879 MB
```

## Code Changes

### 1. MemoryManager.__init__() (`src/utils/memory_manager.py`)
```python
# OLD: Fixed threshold
def __init__(self, cleanup_threshold_mb: float = 500.0):
    self.cleanup_threshold_mb = 500.0  # Always 500MB

# NEW: Auto-detect from GPU
def __init__(self, cleanup_threshold_mb: float = None, 
             memory_reserve_percent: float = 0.15):
    # Detect GPU VRAM
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    # Set threshold at 85% of total
    self.cleanup_threshold_mb = total_mb * 0.85
```

### 2. Enhanced Logging
```python
# OLD: Just numbers
"GPU: 7234.5MB, CPU: 1234.0MB"

# NEW: Percentages and warnings
"GPU: 7234.5MB (66.2%), CPU: 1234.0MB"
"GPU: 8800MB (80% ⚠️ HIGH), CPU: 1234.0MB"
```

### 3. Production Script Integration
```python
# Auto-display GPU config at startup
from src.utils.memory_manager import print_gpu_memory_info

def main():
    ...
    if device.type == 'cuda':
        print_gpu_memory_info()  # Shows VRAM and thresholds
```

## Usage

### Default Behavior (Automatic)
```bash
# Just run training - memory manager auto-configures
python3 scripts/cli.py train --task llm --production-mode
```

### Custom Reserve Percentage
```python
from src.utils.memory_manager import SpikingMemoryManager, set_memory_manager

# Keep 20% buffer instead of 15%
manager = SpikingMemoryManager(memory_reserve_percent=0.20)
set_memory_manager(manager)
```

### Manual Threshold Override
```python
# Force specific threshold (not recommended)
manager = SpikingMemoryManager(cleanup_threshold_mb=8000)
```

## GPU-Specific Recommendations

### RTX 3080 (10.7GB)
- **Auto threshold**: 9284 MB (85%)
- **Recommended settings**:
  ```bash
  --batch-size 4 --sequence-length 128
  --checkpoint-activations --accumulation-steps 4
  ```

### RTX 3090 (24GB)
- **Auto threshold**: 20480 MB (85%)
- **Recommended settings**:
  ```bash
  --batch-size 8 --sequence-length 256
  --checkpoint-activations --accumulation-steps 2
  ```

### RTX 4090 (24GB)
- **Auto threshold**: 20879 MB (85%)
- **Recommended settings**:
  ```bash
  --batch-size 12 --sequence-length 256
  --checkpoint-activations --accumulation-steps 2
  ```

### A100 (40GB)
- **Auto threshold**: 34133 MB (85%)
- **Recommended settings**:
  ```bash
  --batch-size 16 --sequence-length 512
  --checkpoint-activations --accumulation-steps 1
  ```

## Monitoring

### Watch Memory in Real-Time
```bash
# Terminal 1: Training
python3 scripts/cli.py train ...

# Terminal 2: GPU monitor
watch -n 1 nvidia-smi
```

### Check Memory Manager Logs
Look for these key messages:
```
✅ Initialization:
INFO: 🎯 GPU 0 Memory Manager initialized:
INFO:    Total VRAM: 10922 MB
INFO:    Safe threshold: 9284 MB (85% of total)

✅ During Training:
INFO: Memory usage (batch 100): GPU: 7234.5MB (66.2%), CPU: 1234.0MB

⚠️ Warning Signs:
INFO: Memory usage (batch 200): GPU: 8800MB (80% ⚠️ HIGH), CPU: 1234.0MB
INFO: GPU 0 memory (9500MB) exceeds threshold (9284MB)
```

## Troubleshooting

### Still Getting OOM?
1. Check if threshold was detected correctly:
   ```
   Look for: "Safe threshold: XXXX MB" in startup logs
   ```

2. Reduce batch size or sequence length:
   ```bash
   --batch-size 2 --sequence-length 64
   ```

3. Enable gradient checkpointing:
   ```bash
   --checkpoint-activations
   ```

### Memory Not Cleaning Up?
The auto-cleanup runs every 30 seconds. For immediate cleanup:
```python
from src.utils.memory_manager import cleanup_memory
cleanup_memory(force=True)
```

## Summary

| Feature | Before | After |
|---------|--------|-------|
| Threshold | Fixed 500MB | Auto-detect per GPU |
| Multi-GPU | Single threshold | Per-GPU thresholds |
| Logging | "7234.5MB" | "7234.5MB (66.2%)" |
| Warnings | None | "⚠️ HIGH", "🔴 CRITICAL" |
| Configuration | Manual | Automatic |

**Result**: Memory manager now works optimally on **any GPU** without manual configuration!
