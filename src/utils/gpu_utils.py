"""
GPU Utilities for Multi-GPU Training Support

This module provides comprehensive GPU detection, configuration, and distributed training
utilities for the Hybrid Liquid-Spiking Neural Network system.

No shortcuts, no mock data - full implementation for production use.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import os
import socket
import subprocess
import time
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

class MultiGPUStrategy(Enum):
    """Multi-GPU training strategies."""
    NONE = "none"                    # Single GPU or CPU
    DATA_PARALLEL = "dp"             # DataParallel (single machine, multiple GPUs)
    DISTRIBUTED_DATA_PARALLEL = "ddp"  # DistributedDataParallel (single/multi machine)
    AUTO = "auto"                    # Automatically choose best strategy

@dataclass
class GPUInfo:
    """Information about a single GPU."""
    device_id: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    memory_used: int   # MB
    compute_capability: Tuple[int, int]
    is_available: bool
    temperature: Optional[float] = None  # Celsius
    power_usage: Optional[float] = None  # Watts
    utilization: Optional[float] = None  # Percentage

@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training."""
    strategy: MultiGPUStrategy = MultiGPUStrategy.AUTO
    gpu_ids: Optional[List[int]] = None  # None means use all available
    master_addr: str = "localhost"
    master_port: str = "12355"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    sync_batchnorm: bool = True

class GPUDetector:
    """Advanced GPU detection and monitoring utilities."""
    
    @staticmethod
    def detect_gpus() -> List[GPUInfo]:
        """Detect all available GPUs and their detailed information."""
        gpus = []
        
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available. No GPUs detected.")
            return gpus
        
        device_count = torch.cuda.device_count()
        logging.info(f"Detected {device_count} CUDA devices")
        
        for i in range(device_count):
            try:
                # Basic PyTorch GPU info
                props = torch.cuda.get_device_properties(i)
                
                # Memory information
                torch.cuda.set_device(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory // (1024**2)
                memory_free = torch.cuda.memory_reserved(i) // (1024**2)
                memory_used = (torch.cuda.memory_allocated(i)) // (1024**2)
                
                gpu_info = GPUInfo(
                    device_id=i,
                    name=props.name,
                    memory_total=memory_total,
                    memory_free=memory_free,
                    memory_used=memory_used,
                    compute_capability=(props.major, props.minor),
                    is_available=True
                )
                
                # Try to get additional info via nvidia-smi
                try:
                    gpu_info.temperature = GPUDetector._get_gpu_temperature(i)
                    gpu_info.power_usage = GPUDetector._get_gpu_power_usage(i)
                    gpu_info.utilization = GPUDetector._get_gpu_utilization(i)
                except Exception as e:
                    logging.debug(f"Could not get extended GPU info for device {i}: {e}")
                
                gpus.append(gpu_info)
                
            except Exception as e:
                logging.error(f"Error detecting GPU {i}: {e}")
                gpus.append(GPUInfo(
                    device_id=i,
                    name="Unknown",
                    memory_total=0,
                    memory_free=0,
                    memory_used=0,
                    compute_capability=(0, 0),
                    is_available=False
                ))
        
        return gpus
    
    @staticmethod
    def _get_gpu_temperature(device_id: int) -> Optional[float]:
        """Get GPU temperature using nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=temperature.gpu', 
                '--format=csv,noheader,nounits', f'--id={device_id}'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None
    
    @staticmethod
    def _get_gpu_power_usage(device_id: int) -> Optional[float]:
        """Get GPU power usage using nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=power.draw', 
                '--format=csv,noheader,nounits', f'--id={device_id}'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None
    
    @staticmethod
    def _get_gpu_utilization(device_id: int) -> Optional[float]:
        """Get GPU utilization using nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu', 
                '--format=csv,noheader,nounits', f'--id={device_id}'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None
    
    @staticmethod
    def get_optimal_gpu_strategy(num_gpus: int) -> MultiGPUStrategy:
        """Determine the optimal multi-GPU strategy based on available hardware."""
        if num_gpus <= 1:
            return MultiGPUStrategy.NONE
        elif num_gpus <= 4:
            # For small numbers of GPUs, DataParallel can be simpler
            return MultiGPUStrategy.DATA_PARALLEL
        else:
            # For larger numbers of GPUs, DistributedDataParallel is more efficient
            return MultiGPUStrategy.DISTRIBUTED_DATA_PARALLEL
    
    @staticmethod
    def filter_compatible_gpus(gpus: List[GPUInfo], min_memory_mb: int = 4096) -> List[GPUInfo]:
        """Filter GPUs that meet minimum requirements for training."""
        compatible = []
        for gpu in gpus:
            if (gpu.is_available and 
                gpu.memory_total >= min_memory_mb and
                gpu.compute_capability[0] >= 5):  # Minimum compute capability
                compatible.append(gpu)
        return compatible
    
    @staticmethod
    def print_gpu_summary(gpus: List[GPUInfo], logger=None):
        """Print a detailed summary of available GPUs."""
        print_fn = logger.info if logger else print
        
        if not gpus:
            print_fn("🚫 No GPUs detected or CUDA unavailable")
            return
        
        print_fn("🔥 GPU Detection Summary:")
        print_fn("=" * 60)
        
        for i, gpu in enumerate(gpus):
            status = "✅ Available" if gpu.is_available else "❌ Unavailable"
            memory_gb = gpu.memory_total / 1024
            
            print_fn(f"GPU {gpu.device_id}: {gpu.name}")
            print_fn(f"  Status: {status}")
            print_fn(f"  Memory: {memory_gb:.1f} GB total, {gpu.memory_free} MB free")
            print_fn(f"  Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
            
            if gpu.temperature is not None:
                print_fn(f"  Temperature: {gpu.temperature}°C")
            if gpu.power_usage is not None:
                print_fn(f"  Power: {gpu.power_usage}W")
            if gpu.utilization is not None:
                print_fn(f"  Utilization: {gpu.utilization}%")
            print_fn("")

class MultiGPUTrainingManager:
    """Manager for multi-GPU training setup and coordination."""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self.gpus = GPUDetector.detect_gpus()
        self.compatible_gpus = GPUDetector.filter_compatible_gpus(self.gpus)
        self.is_distributed = False
        self.is_main_process = True
        
    def setup_multi_gpu_training(self) -> Tuple[str, List[int]]:
        """Setup multi-GPU training based on configuration and available hardware."""
        if not self.compatible_gpus:
            logging.warning("No compatible GPUs found. Falling back to CPU training.")
            return "cpu", []
        
        # Determine GPU IDs to use
        if self.config.gpu_ids is None:
            gpu_ids = [gpu.device_id for gpu in self.compatible_gpus]
        else:
            # Validate requested GPU IDs
            available_ids = {gpu.device_id for gpu in self.compatible_gpus}
            gpu_ids = [gid for gid in self.config.gpu_ids if gid in available_ids]
            
            if len(gpu_ids) != len(self.config.gpu_ids):
                logging.warning(f"Some requested GPU IDs are not available. Using: {gpu_ids}")
        
        if len(gpu_ids) == 0:
            logging.warning("No valid GPU IDs. Falling back to CPU training.")
            return "cpu", []
        elif len(gpu_ids) == 1:
            logging.info(f"Using single GPU: {gpu_ids[0]}")
            return f"cuda:{gpu_ids[0]}", gpu_ids
        
        # Multi-GPU setup
        strategy = self.config.strategy
        if strategy == MultiGPUStrategy.AUTO:
            strategy = GPUDetector.get_optimal_gpu_strategy(len(gpu_ids))
        
        if strategy == MultiGPUStrategy.DATA_PARALLEL:
            logging.info(f"Setting up DataParallel training on GPUs: {gpu_ids}")
            return f"cuda:{gpu_ids[0]}", gpu_ids
        elif strategy == MultiGPUStrategy.DISTRIBUTED_DATA_PARALLEL:
            logging.info(f"Setting up DistributedDataParallel training on GPUs: {gpu_ids}")
            self._setup_distributed_training(gpu_ids)
            return f"cuda:{self.config.local_rank}", gpu_ids
        else:
            # Fallback to single GPU
            logging.info(f"Using single GPU: {gpu_ids[0]}")
            return f"cuda:{gpu_ids[0]}", [gpu_ids[0]]
    
    def _setup_distributed_training(self, gpu_ids: List[int]):
        """Setup distributed training environment."""
        self.is_distributed = True
        
        # Set environment variables for distributed training
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        os.environ['WORLD_SIZE'] = str(len(gpu_ids))
        os.environ['RANK'] = str(self.config.rank)
        os.environ['LOCAL_RANK'] = str(self.config.local_rank)
        
        # Update config
        self.config.world_size = len(gpu_ids)
        self.is_main_process = self.config.rank == 0
        
        logging.info(f"Distributed training setup:")
        logging.info(f"  World size: {self.config.world_size}")
        logging.info(f"  Rank: {self.config.rank}")
        logging.info(f"  Local rank: {self.config.local_rank}")
        logging.info(f"  Master: {self.config.master_addr}:{self.config.master_port}")
    
    def initialize_distributed_process_group(self):
        """Initialize the distributed process group."""
        if not self.is_distributed:
            return
        
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                    world_size=self.config.world_size,
                    rank=self.config.rank,
                    timeout=torch.distributed.constants.default_pg_timeout
                )
                logging.info("Distributed process group initialized successfully")
            
            # Set device for this process
            torch.cuda.set_device(self.config.local_rank)
            
        except Exception as e:
            logging.error(f"Failed to initialize distributed process group: {e}")
            raise
    
    def cleanup_distributed(self):
        """Clean up distributed training resources."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            logging.info("Distributed process group cleaned up")
    
    def wrap_model_for_multi_gpu(self, model, strategy: MultiGPUStrategy, gpu_ids: List[int]):
        """Wrap model for multi-GPU training."""
        if len(gpu_ids) <= 1:
            return model
        
        if strategy == MultiGPUStrategy.DATA_PARALLEL:
            logging.info("Wrapping model with DataParallel")
            model = DP(model, device_ids=gpu_ids)
        elif strategy == MultiGPUStrategy.DISTRIBUTED_DATA_PARALLEL:
            logging.info("Wrapping model with DistributedDataParallel")
            model = DDP(
                model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=self.config.find_unused_parameters,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view
            )
            
            # Convert BatchNorm to SyncBatchNorm for better distributed training
            if self.config.sync_batchnorm:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logging.info("Converted model to use SyncBatchNorm")
        
        return model
    
    def adjust_batch_size_for_multi_gpu(self, batch_size: int, num_gpus: int) -> int:
        """Adjust batch size for multi-GPU training."""
        if num_gpus <= 1:
            return batch_size
        
        # For distributed training, each GPU processes batch_size samples
        # For data parallel, the batch is split across GPUs
        if self.config.strategy == MultiGPUStrategy.DISTRIBUTED_DATA_PARALLEL:
            # Each process gets the full batch size
            adjusted_batch_size = batch_size
        else:
            # DataParallel splits the batch, so we might want to increase total batch size
            adjusted_batch_size = batch_size * num_gpus
        
        logging.info(f"Adjusted batch size from {batch_size} to {adjusted_batch_size} for {num_gpus} GPUs")
        return adjusted_batch_size
    
    def should_save_checkpoint(self) -> bool:
        """Determine if this process should save checkpoints (only main process in distributed)."""
        return self.is_main_process
    
    def all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce a tensor across all processes (for metrics aggregation)."""
        if not self.is_distributed or not dist.is_initialized():
            return tensor
        
        # Clone to avoid modifying original
        reduced_tensor = tensor.clone()
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
        reduced_tensor /= self.config.world_size
        return reduced_tensor
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

def setup_multi_gpu_environment(
    strategy: str = "auto",
    gpu_ids: Optional[List[int]] = None,
    master_port: str = "12355"
) -> Tuple[MultiGPUTrainingManager, str, List[int]]:
    """
    High-level function to setup multi-GPU training environment.
    
    Args:
        strategy: Multi-GPU strategy ("auto", "dp", "ddp", "none")
        gpu_ids: List of GPU IDs to use (None for all available)
        master_port: Port for distributed training communication
    
    Returns:
        Tuple of (manager, device_string, gpu_ids_used)
    """
    # Create multi-GPU configuration
    multi_gpu_config = MultiGPUConfig(
        strategy=MultiGPUStrategy(strategy),
        gpu_ids=gpu_ids,
        master_port=master_port
    )
    
    # Create training manager
    manager = MultiGPUTrainingManager(multi_gpu_config)
    
    # Print GPU summary
    GPUDetector.print_gpu_summary(manager.gpus)
    
    # Setup training
    device, gpu_ids_used = manager.setup_multi_gpu_training()
    
    if len(gpu_ids_used) > 1:
        logging.info(f"🚀 Multi-GPU training configured:")
        logging.info(f"   Strategy: {manager.config.strategy.value}")
        logging.info(f"   GPUs: {gpu_ids_used}")
        logging.info(f"   Device: {device}")
    elif len(gpu_ids_used) == 1:
        logging.info(f"🔥 Single GPU training on: {device}")
    else:
        logging.info("💻 CPU training (no compatible GPUs found)")
    
    return manager, device, gpu_ids_used

def create_distributed_sampler(dataset, num_replicas=None, rank=None, shuffle=True):
    """Create a distributed sampler for multi-GPU training."""
    if num_replicas is None or rank is None:
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        else:
            return None
    
    from torch.utils.data.distributed import DistributedSampler
    return DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        drop_last=True
    )

def launch_distributed_training(training_fn, num_gpus: int, args):
    """
    Launch distributed training using torch.multiprocessing.
    
    Args:
        training_fn: Function to run training (should accept rank as first argument)
        num_gpus: Number of GPUs to use
        args: Arguments to pass to training function
    """
    if num_gpus <= 1:
        # Single GPU or CPU training
        training_fn(0, *args)
    else:
        # Multi-GPU distributed training
        mp.spawn(
            training_fn,
            args=args,
            nprocs=num_gpus,
            join=True,
            daemon=False,
            start_method='spawn'
        )
