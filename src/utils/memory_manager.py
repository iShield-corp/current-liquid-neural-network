"""
Memory Management Utilities for Hybrid Liquid-Spiking Neural Networks

This module provides comprehensive memory management tools to prevent memory leaks
in training and inference of spiking neural networks with temporal dynamics.
"""

import gc
import torch
import psutil
import logging
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import weakref
import threading
import time


class MemoryManager:
    """
    Comprehensive memory management for neural network training and inference.
    
    Features:
    - GPU memory monitoring and cleanup
    - Memory leak detection and prevention
    - Automatic garbage collection
    - Memory profiling and reporting
    - Context managers for safe memory operations
    """
    
    def __init__(self, cleanup_threshold_mb: float = 500.0, auto_cleanup: bool = True):
        """
        Initialize memory manager.
        
        Args:
            cleanup_threshold_mb: Trigger cleanup when GPU memory exceeds this threshold (MB)
            auto_cleanup: Whether to automatically cleanup memory periodically
        """
        self.cleanup_threshold_mb = cleanup_threshold_mb
        self.auto_cleanup = auto_cleanup
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.initial_memory = self._get_memory_info()
        self.peak_memory = self.initial_memory.copy()
        self.cleanup_count = 0
        
        # Weak references to track tensors
        self._tracked_tensors = weakref.WeakSet()
        
        # Thread-safe operations
        self._lock = threading.Lock()
        
        # Auto cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        if auto_cleanup:
            self._start_auto_cleanup()
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {}
        
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        info['cpu_memory_mb'] = memory_info.rss / 1024 / 1024
        info['cpu_memory_percent'] = process.memory_percent()
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                info[f'gpu_{i}_allocated_mb'] = allocated
                info[f'gpu_{i}_reserved_mb'] = reserved
        
        return info
    
    def _start_auto_cleanup(self):
        """Start automatic cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(30):  # Check every 30 seconds
                try:
                    if self._should_cleanup():
                        self.cleanup_memory()
                except Exception as e:
                    self.logger.warning(f"Auto cleanup failed: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _should_cleanup(self) -> bool:
        """Check if cleanup should be triggered."""
        if not torch.cuda.is_available():
            return False
        
        for i in range(torch.cuda.device_count()):
            allocated_mb = torch.cuda.memory_allocated(i) / 1024 / 1024
            if allocated_mb > self.cleanup_threshold_mb:
                return True
        
        return False
    
    def track_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Track a tensor for memory monitoring."""
        with self._lock:
            self._tracked_tensors.add(tensor)
        return tensor
    
    def cleanup_memory(self, force: bool = False):
        """
        Comprehensive memory cleanup.
        
        Args:
            force: Force cleanup even if threshold not reached
        """
        with self._lock:
            if not force and not self._should_cleanup():
                return
            
            # Clear tracked tensors
            self._tracked_tensors.clear()
            
            # Python garbage collection
            collected = gc.collect()
            
            # CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.cleanup_count += 1
            
            if self.logger.isEnabledFor(logging.INFO):
                memory_info = self._get_memory_info()
                self.logger.info(f"Memory cleanup #{self.cleanup_count}: "
                               f"collected {collected} objects, "
                               f"GPU memory: {memory_info.get('gpu_0_allocated_mb', 0):.1f}MB")
    
    def get_memory_report(self) -> str:
        """Generate detailed memory usage report."""
        current = self._get_memory_info()
        
        report = []
        report.append("=== Memory Usage Report ===")
        report.append(f"CPU Memory: {current['cpu_memory_mb']:.1f}MB ({current['cpu_memory_percent']:.1f}%)")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = current.get(f'gpu_{i}_allocated_mb', 0)
                reserved = current.get(f'gpu_{i}_reserved_mb', 0)
                report.append(f"GPU {i}: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
        
        report.append(f"Cleanup operations: {self.cleanup_count}")
        report.append(f"Tracked tensors: {len(self._tracked_tensors)}")
        
        return "\n".join(report)
    
    def log_memory_usage(self, step_name: str = ""):
        """Log current memory usage with optional step name."""
        memory_info = self._get_memory_info()
        
        # Update peak memory
        for key, value in memory_info.items():
            if key not in self.peak_memory or value > self.peak_memory[key]:
                self.peak_memory[key] = value
        
        gpu_mem = memory_info.get('gpu_0_allocated_mb', 0)
        cpu_mem = memory_info['cpu_memory_mb']
        
        self.logger.info(f"Memory usage{' (' + step_name + ')' if step_name else ''}: "
                        f"GPU: {gpu_mem:.1f}MB, CPU: {cpu_mem:.1f}MB")
    
    @contextmanager
    def memory_scope(self, name: str = "operation"):
        """
        Context manager for memory-safe operations.
        
        Automatically cleans up memory before and after operation,
        and logs memory usage.
        """
        self.log_memory_usage(f"{name} start")
        
        try:
            yield self
        finally:
            self.cleanup_memory()
            self.log_memory_usage(f"{name} end")
    
    def create_safe_tensor(self, *args, **kwargs) -> torch.Tensor:
        """Create tensor with automatic tracking and memory management."""
        tensor = torch.tensor(*args, **kwargs)
        return self.track_tensor(tensor)
    
    def create_safe_zeros(self, *args, **kwargs) -> torch.Tensor:
        """Create zeros tensor with automatic tracking."""
        tensor = torch.zeros(*args, **kwargs)
        return self.track_tensor(tensor)
    
    def create_safe_ones(self, *args, **kwargs) -> torch.Tensor:
        """Create ones tensor with automatic tracking.""" 
        tensor = torch.ones(*args, **kwargs)
        return self.track_tensor(tensor)
    
    def safe_cat(self, tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Memory-safe tensor concatenation."""
        result = torch.cat(tensors, dim=dim)
        return self.track_tensor(result)
    
    def safe_stack(self, tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Memory-safe tensor stacking."""
        result = torch.stack(tensors, dim=dim)
        return self.track_tensor(result)
    
    def cleanup_and_exit(self):
        """Clean up resources before shutdown."""
        if self.auto_cleanup and self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)
        
        self.cleanup_memory(force=True)
        
        final_report = self.get_memory_report()
        self.logger.info(f"Final memory report:\n{final_report}")


class SpikingMemoryManager(MemoryManager):
    """
    Specialized memory manager for spiking neural networks.
    
    Handles memory patterns specific to temporal spiking computations,
    including spike recording accumulation and membrane potential states.
    """
    
    def __init__(self, max_spike_steps: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.max_spike_steps = max_spike_steps
        self._spike_accumulators = weakref.WeakSet()
    
    def create_spike_accumulator(self, shape: Tuple[int, ...], device: torch.device, 
                                dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Create memory-efficient spike accumulator tensor.
        
        Args:
            shape: Shape of the accumulator tensor (spike_steps, batch_size, ...)
            device: Device to create tensor on
            dtype: Data type for tensor
            
        Returns:
            Pre-allocated tensor for spike accumulation
        """
        accumulator = torch.zeros(shape, device=device, dtype=dtype)
        self._spike_accumulators.add(accumulator)
        return self.track_tensor(accumulator)
    
    def clear_spike_accumulators(self):
        """Clear all tracked spike accumulators."""
        with self._lock:
            for accumulator in list(self._spike_accumulators):
                if accumulator.is_cuda:
                    accumulator.zero_()
            self._spike_accumulators.clear()
    
    def cleanup_memory(self, force: bool = False):
        """Enhanced cleanup for spiking networks."""
        self.clear_spike_accumulators()
        super().cleanup_memory(force)
    
    @contextmanager
    def spike_processing_scope(self, batch_size: int, seq_len: int, 
                              num_features: int, spike_steps: int):
        """
        Context manager for spike processing operations.
        
        Pre-allocates necessary tensors and ensures cleanup.
        """
        with self.memory_scope("spike_processing"):
            # Pre-allocate common tensors
            accumulators = {
                'spikes': self.create_spike_accumulator(
                    (spike_steps, batch_size, seq_len, num_features),
                    device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
                )
            }
            
            try:
                yield accumulators
            finally:
                # Explicit cleanup of accumulators
                for tensor in accumulators.values():
                    del tensor


# Global memory manager instance
_global_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = SpikingMemoryManager()
    return _global_memory_manager

def set_memory_manager(manager: MemoryManager):
    """Set global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager:
        _global_memory_manager.cleanup_and_exit()
    _global_memory_manager = manager

def cleanup_global_memory():
    """Cleanup global memory manager."""
    global _global_memory_manager
    if _global_memory_manager:
        _global_memory_manager.cleanup_and_exit()
        _global_memory_manager = None


# Convenience functions for memory-safe operations
def safe_tensor(*args, **kwargs) -> torch.Tensor:
    """Create memory-safe tensor."""
    return get_memory_manager().create_safe_tensor(*args, **kwargs)

def safe_zeros(*args, **kwargs) -> torch.Tensor:
    """Create memory-safe zeros tensor."""
    return get_memory_manager().create_safe_zeros(*args, **kwargs)

def safe_ones(*args, **kwargs) -> torch.Tensor:
    """Create memory-safe ones tensor."""
    return get_memory_manager().create_safe_ones(*args, **kwargs)

def safe_cat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Memory-safe tensor concatenation."""
    return get_memory_manager().safe_cat(tensors, dim)

def safe_stack(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Memory-safe tensor stacking."""
    return get_memory_manager().safe_stack(tensors, dim)

@contextmanager
def memory_scope(name: str = "operation"):
    """Context manager for memory-safe operations."""
    with get_memory_manager().memory_scope(name):
        yield

def log_memory(step_name: str = ""):
    """Log current memory usage."""
    get_memory_manager().log_memory_usage(step_name)

def cleanup_memory(force: bool = False):
    """Cleanup memory."""
    get_memory_manager().cleanup_memory(force)
