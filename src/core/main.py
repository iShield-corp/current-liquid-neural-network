import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
import ncps
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP, FullyConnected
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import math
from collections import OrderedDict
import torchvision

# Import memory management utilities
try:
    from ..utils.memory_manager import (
        get_memory_manager, SpikingMemoryManager, memory_scope, 
        safe_zeros, safe_stack, cleanup_memory, log_memory
    )
except ImportError:
    # Fallback for relative imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.memory_manager import (
        get_memory_manager, SpikingMemoryManager, memory_scope, 
        safe_zeros, safe_stack, cleanup_memory, log_memory
    )
import torchvision.transforms as transforms
from torch.amp import autocast
from torch.amp import GradScaler
import hashlib
import time
from enum import Enum
import logging
from tqdm import tqdm

# Set up logger
logger = logging.getLogger(__name__)

# Text processing imports
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, Dataset as HFDataset
import requests
import gzip
import urllib.request

# GPU utilities import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.gpu_utils import (
    setup_multi_gpu_environment, MultiGPUTrainingManager, MultiGPUConfig,
    MultiGPUStrategy, create_distributed_sampler
)

# Advanced Surrogate Gradient Functions
class AdaptiveSurrogateGradient(torch.autograd.Function):
    """
    Temperature-adaptive surrogate gradient that adjusts based on training progress
    and membrane potential statistics for better convergence.
    """
    
    @staticmethod
    def forward(ctx, input, threshold, beta, temperature=1.0):
        ctx.save_for_backward(input, threshold, beta, torch.tensor(temperature))
        return (input >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, threshold, beta, temperature = ctx.saved_tensors
        
        # Multi-scale surrogate gradient
        # Primary: Fast sigmoid for main gradient
        primary_grad = beta / (1 + (beta * (input - threshold).abs()).pow(2))
        
        # Secondary: Gaussian for fine-tuning
        sigma = 0.5 / temperature
        secondary_grad = torch.exp(-((input - threshold) / sigma).pow(2)) / (sigma * np.sqrt(2 * np.pi))
        
        # Combine with learnable weight
        alpha = torch.sigmoid(input.mean() - threshold.mean())
        combined_grad = alpha * primary_grad + (1 - alpha) * secondary_grad
        
        # Temperature scaling
        scaled_grad = combined_grad / temperature
        
        return grad_output * scaled_grad, None, None, None

class MultiTimescaleSurrogate(nn.Module):
    """
    Multiple timescale surrogate gradients for different temporal dynamics.
    """
    
    def __init__(self, fast_beta=10.0, slow_beta=2.0):
        super().__init__()
        self.fast_beta = nn.Parameter(torch.tensor(fast_beta))
        self.slow_beta = nn.Parameter(torch.tensor(slow_beta))
        self.register_buffer('temperature', torch.tensor(1.0))
        
    def forward(self, membrane_potential, threshold):
        # Fast dynamics for rapid adaptation
        fast_spikes = AdaptiveSurrogateGradient.apply(
            membrane_potential, threshold, self.fast_beta, self.temperature
        )
        
        # Slow dynamics for stability
        slow_spikes = AdaptiveSurrogateGradient.apply(
            membrane_potential, threshold, self.slow_beta, self.temperature
        )
        
        # Combine based on membrane potential variance
        variance = membrane_potential.var()
        weight = torch.sigmoid(variance - 0.5)
        
        return weight * fast_spikes + (1 - weight) * slow_spikes
    
    def update_temperature(self, epoch, max_epochs):
        """Adaptive temperature scheduling."""
        # Start high, decrease to encourage precision
        self.temperature = 2.0 * (1 - epoch / max_epochs) + 0.5

# Temporal Credit Assignment Enhancement
class TemporalCreditAssignment(nn.Module):
    """
    Sophisticated temporal credit assignment using eligibility traces
    and multi-timescale STDP-inspired learning.
    """
    
    def __init__(self, hidden_dim, trace_decay=0.95, fast_decay=0.8, slow_decay=0.99):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trace_decay = trace_decay
        self.fast_decay = fast_decay
        self.slow_decay = slow_decay
        
        # Learnable trace parameters
        self.trace_weights = nn.Parameter(torch.ones(hidden_dim))
        self.temporal_kernel = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        
        # Multi-timescale eligibility traces
        self.register_buffer('fast_trace', torch.zeros(1, hidden_dim))
        self.register_buffer('slow_trace', torch.zeros(1, hidden_dim))
        
    def forward(self, spike_trains, liquid_states, targets):
        batch_size, seq_len, hidden_dim = spike_trains.shape
        
        # Initialize traces for batch
        fast_traces = torch.zeros(batch_size, seq_len, hidden_dim, device=spike_trains.device)
        slow_traces = torch.zeros(batch_size, seq_len, hidden_dim, device=spike_trains.device)
        
        credit_signals = torch.zeros_like(spike_trains)
        
        for t in range(seq_len):
            # Update eligibility traces
            if t > 0:
                fast_traces[:, t] = (self.fast_decay * fast_traces[:, t-1] + 
                                   spike_trains[:, t] * self.trace_weights)
                slow_traces[:, t] = (self.slow_decay * slow_traces[:, t-1] + 
                                   liquid_states[:, t] * self.trace_weights)
            else:
                fast_traces[:, t] = spike_trains[:, t] * self.trace_weights
                slow_traces[:, t] = liquid_states[:, t] * self.trace_weights
            
            # Compute prediction error
            if t < seq_len - 1:
                pred_error = targets[:, t+1] - liquid_states[:, t]
                
                # Assign credit based on traces
                fast_credit = fast_traces[:, t] * pred_error.unsqueeze(-1)
                slow_credit = slow_traces[:, t] * pred_error.unsqueeze(-1)
                
                # Combine with learned weights
                credit_signals[:, t] = 0.3 * fast_credit + 0.7 * slow_credit
        
        # Apply temporal smoothing
        credit_smooth = self.temporal_kernel(
            credit_signals.transpose(1, 2)
        ).transpose(1, 2)
        
        return credit_smooth


# Advanced Spike-to-Probability Conversion Mechanisms
class SpikeDecoder(nn.Module):
    """
    Advanced spike decoding mechanisms based on latest research.
    
    Supports multiple decoding strategies:
    - Rate coding: spike count over time window
    - Temporal coding: spike timing information
    - First-to-spike: temporal order encoding
    - Probabilistic: smooth conversion with uncertainty
    - Hybrid: combination of multiple methods
    """
    
    def __init__(self, input_dim, output_dim, decode_method='hybrid', 
                 temporal_window=10, temperature=1.0, use_uncertainty=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.decode_method = decode_method
        self.temporal_window = temporal_window
        self.temperature = temperature
        self.use_uncertainty = use_uncertainty
        
        # Rate coding components
        self.rate_projection = nn.Linear(input_dim, output_dim)
        self.rate_normalization = nn.LayerNorm(output_dim)
        
        # Temporal coding components
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim * temporal_window, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
        
        # First-to-spike decoder
        self.spike_order_embedding = nn.Embedding(temporal_window + 1, input_dim)
        self.order_projection = nn.Linear(input_dim, output_dim)
        
        # Probabilistic components with uncertainty estimation
        if use_uncertainty:
            self.uncertainty_head = nn.Linear(input_dim, output_dim)
            self.uncertainty_activation = nn.Softplus()
        
        # Hybrid fusion network
        if decode_method == 'hybrid':
            self.fusion_weights = nn.Parameter(torch.ones(3) / 3)  # Rate, temporal, order
            self.fusion_layer = nn.Sequential(
                nn.Linear(output_dim * 3, output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim)
            )
        
        # Temporal aggregation for smoothing
        self.temporal_aggregator = TemporalAggregator(output_dim, window_size=temporal_window)
        
    def rate_decode(self, spike_trains):
        """Rate coding: convert spike counts to probabilities."""
        # Sum spikes over temporal dimension
        spike_rates = spike_trains.sum(dim=-2) / self.temporal_window  # [batch, features]
        
        # Project to output space
        logits = self.rate_projection(spike_rates)
        logits = self.rate_normalization(logits)
        
        return logits / self.temperature
    
    def temporal_decode(self, spike_trains):
        """Temporal coding: encode spike timing information."""
        batch_size, seq_len, features = spike_trains.shape
        
        # Flatten temporal dimension to capture timing patterns
        temporal_pattern = spike_trains.view(batch_size, seq_len * features)
        
        # Encode temporal relationships
        logits = self.temporal_encoder(temporal_pattern)
        
        return logits / self.temperature
    
    def first_spike_decode(self, spike_trains):
        """First-to-spike coding: encode temporal order."""
        batch_size, seq_len, features = spike_trains.shape
        
        # Find first spike time for each feature
        spike_times = torch.zeros(batch_size, features, device=spike_trains.device)
        
        for t in range(seq_len):
            mask = (spike_times == 0) & (spike_trains[:, t, :] > 0.5)
            spike_times[mask] = t + 1  # 1-indexed, 0 = no spike
        
        # Embed spike timing information
        spike_times = spike_times.long().clamp(0, self.temporal_window)
        time_embeddings = self.spike_order_embedding(spike_times)  # [batch, features, embed_dim]
        
        # Aggregate across features
        aggregated = time_embeddings.mean(dim=1)  # [batch, embed_dim]
        
        # Project to output
        logits = self.order_projection(aggregated)
        
        return logits / self.temperature
    
    def probabilistic_decode(self, spike_trains, return_uncertainty=False):
        """Probabilistic decoding with uncertainty estimation."""
        # Use rate coding as base
        base_logits = self.rate_decode(spike_trains)
        
        if self.use_uncertainty:
            # Estimate uncertainty from spike variability
            spike_std = spike_trains.std(dim=-2)  # Temporal variance
            uncertainty = self.uncertainty_head(spike_std)
            uncertainty = self.uncertainty_activation(uncertainty)
            
            # Add uncertainty to logits (Bayesian approach)
            logits = base_logits + torch.randn_like(base_logits) * uncertainty
            
            if return_uncertainty:
                return logits, uncertainty
        
        return base_logits
    
    def hybrid_decode(self, spike_trains):
        """Hybrid decoding combining multiple methods."""
        # Get outputs from all methods
        rate_logits = self.rate_decode(spike_trains)
        temporal_logits = self.temporal_decode(spike_trains)
        order_logits = self.first_spike_decode(spike_trains)
        
        # Concatenate all outputs
        combined = torch.cat([rate_logits, temporal_logits, order_logits], dim=-1)
        
        # Fusion network
        fused_logits = self.fusion_layer(combined)
        
        # Apply learned fusion weights
        weights = torch.softmax(self.fusion_weights, dim=0)
        final_logits = (weights[0] * rate_logits + 
                       weights[1] * temporal_logits + 
                       weights[2] * order_logits + 
                       fused_logits) / 2
        
        return final_logits
    
    def forward(self, spike_trains, return_uncertainty=False):
        """Main forward pass with selected decoding method."""
        if self.decode_method == 'rate':
            logits = self.rate_decode(spike_trains)
        elif self.decode_method == 'temporal':
            logits = self.temporal_decode(spike_trains)
        elif self.decode_method == 'first_spike':
            logits = self.first_spike_decode(spike_trains)
        elif self.decode_method == 'probabilistic':
            if return_uncertainty:
                logits, uncertainty = self.probabilistic_decode(spike_trains, return_uncertainty=True)
                return logits, uncertainty
            else:
                logits = self.probabilistic_decode(spike_trains)
        elif self.decode_method == 'hybrid':
            logits = self.hybrid_decode(spike_trains)
        else:
            raise ValueError(f"Unknown decode method: {self.decode_method}")
        
        # Apply temporal aggregation for smoothing
        logits = self.temporal_aggregator(logits)
        
        return logits


class TemporalAggregator(nn.Module):
    """Temporal aggregation layer for smoothing spike-derived probabilities."""
    
    def __init__(self, dim, window_size=10, aggregation_method='exponential'):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.aggregation_method = aggregation_method
        
        if aggregation_method == 'learned':
            self.aggregation_weights = nn.Parameter(torch.ones(window_size))
        elif aggregation_method == 'exponential':
            # Exponential decay weights
            weights = torch.exp(-torch.arange(window_size, dtype=torch.float) * 0.1)
            self.register_buffer('aggregation_weights', weights / weights.sum())
        
        # Running buffer for temporal smoothing
        self.register_buffer('history_buffer', torch.zeros(1, window_size, dim))
        self.register_buffer('buffer_index', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x):
        """Apply temporal aggregation."""
        # Handle both list of tensors and single tensor input
        if isinstance(x, list):
            if not x:
                raise ValueError("Empty list provided to TemporalAggregator")
            
            # Convert list to tensor: [time_steps, batch, features]
            x_tensor = torch.stack(x, dim=0)  # [time, batch, features]
            
            # Average over time dimension for simplicity
            x = x_tensor.mean(dim=0)  # [batch, features]
        
        batch_size = x.shape[0]
        
        # Expand buffer if needed
        if self.history_buffer.shape[0] < batch_size:
            self.history_buffer = self.history_buffer.expand(batch_size, -1, -1).clone()
        
        # Update circular buffer
        current_idx = self.buffer_index % self.window_size
        self.history_buffer[:batch_size, current_idx] = x
        self.buffer_index += 1
        
        # Apply weighted aggregation
        if self.aggregation_method == 'learned':
            weights = torch.softmax(self.aggregation_weights, dim=0)
        else:
            weights = self.aggregation_weights
        
        # Weighted sum over history
        aggregated = torch.sum(self.history_buffer[:batch_size] * weights.view(1, -1, 1), dim=1)
        
        return aggregated


class PotentialBasedLayerNorm(nn.Module):
    """
    Potential-based Layer Normalization for Spiking Neural Networks.
    
    Based on Sun et al. (2022) "Solving the spike feature information vanishing 
    problem in spiking deep Q network with potential based normalization"
    """
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, membrane_potential):
        """
        Apply potential-based normalization to membrane potential.
        
        Args:
            membrane_potential: Tensor of shape [..., normalized_shape]
                               representing membrane potentials
        """
        # Calculate mean and variance across the last dimension
        mean = membrane_potential.mean(dim=-1, keepdim=True)
        var = membrane_potential.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (membrane_potential - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable affine transformation
        if self.elementwise_affine:
            normalized = normalized * self.weight + self.bias
        
        return normalized

class TaskType(Enum):
    LLM = "llm"
    VISION = "vision"
    ROBOTICS = "robotics"

@dataclass
class ModelConfig:
    # Core architecture parameters (required)
    task_type: TaskType
    input_dim: int
    hidden_dim: int
    output_dim: int
    
    # Liquid neural network parameters (required)
    liquid_units: int
    liquid_backbone: str  # 'cfc', 'ltc', or 'ncp'
    
    # Spiking neural network parameters (required)
    spiking_units: int
    spike_threshold: float
    beta: float  # Membrane potential decay factor
    
    # Network depth and structure (required)
    num_layers: int
    
    # Regularization parameters (required)
    dropout: float
    
    # Training parameters (required)
    sequence_length: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    mixed_precision: bool
    device: str
    seed: int
    
    # Optional parameters with defaults
    num_spike_steps: int = None  # Will default to sequence_length // 4 if None
    num_attention_heads: int = 8
    
    # Embedding parameters (for LLM tasks) - optional
    embedding_dim: int = None  # Will default to input_dim if None
    max_position_embeddings: int = None  # Will default to sequence_length if None
    vocab_size: int = None  # Will default to output_dim for LLM tasks
    
    # Convolutional parameters (for vision tasks) - optional
    conv_channels: List[int] = None  # [32, 64, 128] by default
    conv_kernel_sizes: List[int] = None  # [3, 3, 3] by default
    conv_strides: List[int] = None  # [1, 1, 1] by default
    conv_padding: List[int] = None  # [1, 1, 1] by default
    
    # Regularization parameters - optional
    attention_dropout: float = None  # Will default to dropout if None
    embedding_dropout: float = None  # Will default to dropout if None
    
    # Training parameters - optional
    num_epochs: int = 10
    
    # Advanced parameters - optional
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    
    # Multi-GPU parameters - optional
    multi_gpu_strategy: str = "auto"  # "auto", "dp", "ddp", "none"
    gpu_ids: Optional[List[int]] = None  # None means use all available
    distributed_backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    sync_batchnorm: bool = True  # Use synchronized batch normalization
    find_unused_parameters: bool = False  # For DDP unused parameter detection
    
    # STDP Configuration - optional
    use_stdp: bool = False
    stdp_type: str = 'homeostatic'  # 'classical', 'triplet', 'homeostatic', 'bcm'
    stdp_learning_rate: float = 0.01
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    stdp_target_rate: float = 0.1
    
    # Meta-Plasticity Configuration - optional
    use_meta_plasticity: bool = False
    meta_lr: float = 0.001
    meta_history_length: int = 100
    meta_hidden_dim: int = 128
    
    # Continual Learning Configuration - optional
    use_continual_learning: bool = False
    consolidation_strength: float = 1000.0
    plasticity_decay: float = 0.9
    use_experience_replay: bool = True
    replay_buffer_size: int = 1000
    replay_sampling_strategy: str = 'balanced'  # 'uniform', 'importance', 'balanced'
    
    # Integration flags - optional
    stdp_layers_to_enhance: Optional[List[str]] = None  # None = all
    compute_importance_interval: int = 1  # How often to compute Fisher information (epochs)
    
    # === Mamba Integration Configuration (NEW) ===
    use_mamba: bool = False  # Enable Mamba blocks
    integration_mode: str = 'bidirectional'  # 'sequential', 'parallel', 'bidirectional'
    
    # Mamba-specific parameters
    mamba_d_state: int = 16  # State space dimension
    mamba_d_conv: int = 4    # Convolution kernel size
    mamba_expand: int = 2    # Expansion factor
    
    # Communication parameters
    spike_to_mamba_method: str = 'temporal'  # 'rate', 'temporal', 'potential'
    spike_temporal_tau: float = 20.0  # Time constant for temporal coding
    
    # Gating parameters (for parallel mode)
    use_adaptive_gating: bool = True
    num_gate_heads: int = 4
    
    # Cross-attention parameters (for bidirectional mode)
    use_cross_attention: bool = False  # Only for bidirectional
    cross_attn_heads: int = 8
    
    def __post_init__(self):
        """Set default values for optional parameters."""
        if self.embedding_dim is None:
            self.embedding_dim = self.input_dim
        
        if self.max_position_embeddings is None:
            self.max_position_embeddings = self.sequence_length
            
        if self.vocab_size is None and self.task_type == TaskType.LLM:
            self.vocab_size = self.output_dim
            
        if self.num_spike_steps is None:
            self.num_spike_steps = max(1, self.sequence_length // 4)
            
        if self.attention_dropout is None:
            self.attention_dropout = self.dropout
            
        if self.embedding_dropout is None:
            self.embedding_dropout = self.dropout
            
        # Validate and adjust attention parameters for compatibility
        if self.hidden_dim % self.num_attention_heads != 0:
            # Try to adjust num_attention_heads first to a nearby divisor
            original_num_heads = self.num_attention_heads
            adjusted = False
            for candidate_heads in [self.num_attention_heads - 1, self.num_attention_heads + 1, 
                                  self.num_attention_heads - 2, self.num_attention_heads + 2,
                                  self.num_attention_heads - 3, self.num_attention_heads + 3]:
                if candidate_heads > 0 and self.hidden_dim % candidate_heads == 0:
                    self.num_attention_heads = candidate_heads
                    print(f"⚠️  Auto-adjusted num_attention_heads from {original_num_heads} to {self.num_attention_heads} "
                          f"to make it compatible with hidden_dim={self.hidden_dim}")
                    adjusted = True
                    break
            
            if not adjusted:
                # If no nearby divisor found, adjust hidden_dim to nearest compatible value
                original_hidden_dim = self.hidden_dim
                self.hidden_dim = (self.hidden_dim // self.num_attention_heads) * self.num_attention_heads
                print(f"⚠️  Auto-adjusted hidden_dim from {original_hidden_dim} to {self.hidden_dim} "
                      f"to make it compatible with num_attention_heads={self.num_attention_heads}")
            
        # Set default conv parameters for vision tasks
        if self.task_type == TaskType.VISION:
            if self.conv_channels is None:
                self.conv_channels = [32, 64, 128]
            if self.conv_kernel_sizes is None:
                self.conv_kernel_sizes = [3, 3, 3]
            if self.conv_strides is None:
                self.conv_strides = [1, 1, 1]
            if self.conv_padding is None:
                self.conv_padding = [1, 1, 1]
        
        # Validate continual learning configuration
        if self.use_continual_learning:
            if not self.use_stdp and not self.use_meta_plasticity:
                print("⚠️  Continual learning works best with STDP or meta-plasticity enabled")
        
        # Auto-enable features for optimal continual learning
        if self.use_continual_learning and not self.use_stdp:
            print("📌 Auto-enabling homeostatic STDP for continual learning")
            self.use_stdp = True
            self.stdp_type = 'homeostatic'
        
        # Validate Mamba settings
        if self.use_mamba:
            valid_modes = ['sequential', 'parallel', 'bidirectional']
            if self.integration_mode not in valid_modes:
                raise ValueError(
                    f"Invalid integration_mode: {self.integration_mode}. "
                    f"Must be one of {valid_modes}"
                )
            
            if self.integration_mode == 'bidirectional':
                self.use_cross_attention = True
            
            print(f"🔄 Mamba integration enabled: mode={self.integration_mode}")
            print(f"   d_state={self.mamba_d_state}, "
                  f"d_conv={self.mamba_d_conv}, expand={self.mamba_expand}")
    
    def to_dict(self):
        return {
            'task_type': self.task_type.value,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'liquid_units': self.liquid_units,
            'liquid_backbone': self.liquid_backbone,
            'spiking_units': self.spiking_units,
            'spike_threshold': self.spike_threshold,
            'beta': self.beta,
            'num_spike_steps': self.num_spike_steps,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'embedding_dim': self.embedding_dim,
            'max_position_embeddings': self.max_position_embeddings,
            'vocab_size': self.vocab_size,
            'conv_channels': self.conv_channels,
            'conv_kernel_sizes': self.conv_kernel_sizes,
            'conv_strides': self.conv_strides,
            'conv_padding': self.conv_padding,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'embedding_dropout': self.embedding_dropout,
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'gradient_clip': self.gradient_clip,
            'mixed_precision': self.mixed_precision,
            'device': self.device,
            'seed': self.seed,
            'num_epochs': self.num_epochs,
            'layer_norm_eps': self.layer_norm_eps,
            'initializer_range': self.initializer_range,
            'use_cache': self.use_cache,
            'multi_gpu_strategy': self.multi_gpu_strategy,
            'gpu_ids': self.gpu_ids,
            'distributed_backend': self.distributed_backend,
            'sync_batchnorm': self.sync_batchnorm,
            'find_unused_parameters': self.find_unused_parameters,
            'use_stdp': self.use_stdp,
            'stdp_type': self.stdp_type,
            'stdp_learning_rate': self.stdp_learning_rate,
            'stdp_tau_plus': self.stdp_tau_plus,
            'stdp_tau_minus': self.stdp_tau_minus,
            'stdp_target_rate': self.stdp_target_rate,
            'use_meta_plasticity': self.use_meta_plasticity,
            'meta_lr': self.meta_lr,
            'meta_history_length': self.meta_history_length,
            'meta_hidden_dim': self.meta_hidden_dim,
            'use_continual_learning': self.use_continual_learning,
            'consolidation_strength': self.consolidation_strength,
            'plasticity_decay': self.plasticity_decay,
            'use_experience_replay': self.use_experience_replay,
            'replay_buffer_size': self.replay_buffer_size,
            'replay_sampling_strategy': self.replay_sampling_strategy,
            'stdp_layers_to_enhance': self.stdp_layers_to_enhance,
            'compute_importance_interval': self.compute_importance_interval,
            # Mamba parameters
            'use_mamba': self.use_mamba,
            'integration_mode': self.integration_mode,
            'mamba_d_state': self.mamba_d_state,
            'mamba_d_conv': self.mamba_d_conv,
            'mamba_expand': self.mamba_expand,
            'spike_to_mamba_method': self.spike_to_mamba_method,
            'spike_temporal_tau': self.spike_temporal_tau,
            'use_adaptive_gating': self.use_adaptive_gating,
            'num_gate_heads': self.num_gate_heads,
            'use_cross_attention': self.use_cross_attention,
            'cross_attn_heads': self.cross_attn_heads
        }
    
    @classmethod
    def from_dict(cls, data):
        data['task_type'] = TaskType(data['task_type'])
        return cls(**data)

class SpikingEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_steps, beta=0.95):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_dim, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.dropout = nn.Dropout(0.2)
        
        # Track membrane potential for normalization
        self.membrane_potential = None
        
    def forward(self, x):
        # Handle different input shapes properly
        original_shape = x.shape
        
        # Ensure we have the right dimensions: [batch_size, seq_len, features] or [batch_size, features]
        if len(x.shape) == 2:  # [batch_size, features]
            batch_size, feature_dim = x.shape
            # Expand to include sequence dimension for processing
            x = x.unsqueeze(1).repeat(1, self.num_steps, 1)  # [batch_size, num_steps, features]
        elif len(x.shape) == 3:  # [batch_size, seq_len, features]
            batch_size, seq_len, feature_dim = x.shape
            # If sequence length is 1, expand it to num_steps
            if seq_len == 1:
                x = x.repeat(1, self.num_steps, 1)  # [batch_size, num_steps, features]
        elif len(x.shape) == 1:  # [features] - single sample
            feature_dim = x.shape[0]
            batch_size = 1
            x = x.unsqueeze(0).unsqueeze(0).repeat(1, self.num_steps, 1)  # [1, num_steps, features]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Now x should be [batch_size, num_steps, feature_dim]
        batch_size, actual_steps, feature_dim = x.shape
        
        # Initialize LIF states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Use memory manager for safe tensor allocation
        output_dim = self.fc2.out_features
        spike_output = safe_zeros((batch_size, actual_steps, output_dim), 
                                 device=x.device, dtype=x.dtype)
        
        # Track membrane potentials for normalization
        membrane_potentials = []
        
        # Process each time step
        for step in range(actual_steps):
            # Get input for this time step: [batch_size, feature_dim]
            step_input = x[:, step, :]
            
            # Forward through layers
            cur1 = self.fc1(step_input)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout(spk1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Store spike output for this step
            spike_output[:, step, :] = spk2
            
            # Store membrane potential for normalization
            membrane_potentials.append(mem2.clone().detach())
        
        # Store average membrane potential for potential-based normalization
        if membrane_potentials:
            self.membrane_potential = torch.stack(membrane_potentials, dim=1).mean(dim=1)  # [batch, output_dim]
        
        return spike_output

class LiquidCell(nn.Module):
    def __init__(self, input_dim, units, backbone='cfc'):
        super().__init__()
        self.input_dim = input_dim
        self.units = units
        self.backbone = backbone  # Store backbone for enhanced cell creation
        
        # Always create fallback layer for checkpoint compatibility
        self.fallback = nn.Linear(input_dim, units)
        
        if backbone == 'cfc':
            # For CfC, we need output units < total units - 2
            # So if we want 'units' outputs, we need at least units + 2 total units
            total_units = units + 4  # Give some extra buffer
            wiring = AutoNCP(total_units, units)
            self.cell = CfC(input_dim, wiring, mode="default")
        elif backbone == 'ltc':
            total_units = units + 4
            wiring = AutoNCP(total_units, units)
            self.cell = LTC(input_dim, wiring, return_sequences=True)
        else:
            wiring = FullyConnected(units)
            self.cell = CfC(input_dim, wiring, mode="default")
        
    def forward(self, x, h=None):
        batch_size = x.shape[0]
        
        try:
            # For NCP cells, handle state dimensions properly
            if h is None:
                # NCP cells are sensitive to state dimensions
                if batch_size == 1:
                    # For single batch, use 1D state
                    h = torch.zeros(self.units, device=x.device)
                else:
                    # For multiple batches, use 2D state
                    h = torch.zeros(batch_size, self.units, device=x.device)
            
            # Ensure input is properly shaped for NCP
            if x.dim() == 3 and x.size(1) == 1:
                # Remove singleton sequence dimension
                x_input = x.squeeze(1)
            else:
                x_input = x
                
            output, h_new = self.cell(x_input, h)
            
            # Ensure output has consistent dimensions for fusion
            if output.dim() == 1:
                # Single sample output, add batch and sequence dims
                output = output.unsqueeze(0).unsqueeze(1)
            elif output.dim() == 2 and batch_size > 1:
                # Batch output, add sequence dimension
                output = output.unsqueeze(1)
            elif output.dim() == 2 and batch_size == 1:
                # Single batch, keep as is but ensure correct shape
                if output.size(0) != 1:
                    output = output.unsqueeze(0)
                if output.dim() == 2:
                    output = output.unsqueeze(1)
                    
            return output, h_new
            
        except Exception as e:
            # Use fallback layer if NCP fails - ensure correct dimensions
            if x.dim() == 3 and x.size(1) == 1:
                x_fallback = x.squeeze(1)
            else:
                x_fallback = x
                
            output = self.fallback(x_fallback)
            
            # Ensure fallback output matches expected dimensions
            if output.dim() == 2:
                output = output.unsqueeze(1)
            elif output.dim() == 1:
                output = output.unsqueeze(0).unsqueeze(1)
                
            # Create appropriate h_new for consistency
            if h is None:
                if batch_size == 1:
                    h_new = torch.zeros(self.units, device=x.device)
                else:
                    h_new = torch.zeros(batch_size, self.units, device=x.device)
            else:
                h_new = h
                
            return output, h_new

class ResidualLiquidSpikingBlock(nn.Module):
    """
    Hybrid Liquid-Spiking Block with residual connections.
    
    Based on research by Kim et al. (2024) on addition-based skip connections
    in spiking neural networks and Karn et al. (2024) on liquid networks
    with residual connections.
    """
    
    def __init__(self, input_dim, liquid_units, spiking_units, spike_steps, beta=0.95, 
                 backbone='cfc', use_potential_norm=True, residual_type='addition',
                 stdp_rule=None):
        super().__init__()
        self.input_dim = input_dim
        self.liquid_units = liquid_units
        self.spiking_units = spiking_units
        self.residual_type = residual_type  # 'addition' or 'concatenation'
        self.stdp_rule = stdp_rule  # STDP learning rule (if enabled)
        
        # Core components
        self.spike_encoder = SpikingEncoder(input_dim, spiking_units, spike_steps, beta)
        self.liquid_cell = LiquidCell(spiking_units, liquid_units, backbone)
        
        # Potential-based normalization for spike vanishing prevention
        if use_potential_norm:
            self.potential_norm = PotentialBasedLayerNorm(spiking_units)
        else:
            self.potential_norm = None
        
        # Residual connection handling
        if residual_type == 'addition':
            # Addition-based residual: output must match input_dim
            self.fusion = nn.Sequential(
                nn.Linear(liquid_units + spiking_units, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            self.residual_projection = nn.Identity() if input_dim == input_dim else nn.Linear(input_dim, input_dim)
            
        elif residual_type == 'concatenation':
            # Concatenation-based residual
            self.fusion = nn.Sequential(
                nn.Linear(liquid_units + spiking_units, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            self.residual_fusion = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),  # Concatenated input + output
                nn.LayerNorm(input_dim),
                nn.GELU()
            )
        
        # Temporal gating for better information flow
        self.temporal_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.output_dim = input_dim
        
        # STDP spike tracking (if STDP is enabled)
        if self.stdp_rule is not None:
            self.register_buffer('pre_spike_history', torch.zeros(1, 100, spiking_units))
            self.register_buffer('post_spike_history', torch.zeros(1, 100, liquid_units))
            self.spike_history_idx = 0
            self.spike_history_length = 100
        
    def forward(self, x, h=None, return_internals=False):
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor
            h: Hidden state for liquid cell
            return_internals: Whether to return intermediate activations
        """
        residual = x  # Store original input for residual connection
        
        # Handle sequence input for LLM
        if len(x.shape) == 3:  # [batch, seq_len, features]
            batch_size, seq_len, features = x.shape
            
            output_tensor = safe_zeros((batch_size, seq_len, self.output_dim), 
                                      device=x.device, dtype=x.dtype)
            
            internals = {'spike_outputs': [], 'liquid_outputs': [], 'membrane_potentials': []} if return_internals else None
            
            for t in range(seq_len):
                token_input = x[:, t:t+1, :]  # [batch, 1, features]
                token_residual = residual[:, t, :]  # [batch, features]
                
                # Spike encoding with membrane potential tracking
                spike_out = self.spike_encoder(token_input)
                spike_features = spike_out.mean(dim=1)  # [batch, spiking_units]
                
                # Track spikes for STDP if enabled
                if self.stdp_rule is not None and self.training:
                    # Store spike history for STDP learning
                    idx = self.spike_history_idx % self.spike_history_length
                    if spike_features.size(0) == self.pre_spike_history.size(0):
                        self.pre_spike_history[:, idx, :] = spike_features.detach()
                
                # Apply potential-based normalization to prevent spike vanishing
                if self.potential_norm is not None:
                    # Get membrane potential from spike encoder (assume it's available)
                    membrane_potential = getattr(self.spike_encoder, 'membrane_potential', spike_features)
                    normalized_potential = self.potential_norm(membrane_potential)
                    spike_features = spike_features + 0.1 * normalized_potential  # Small residual influence
                
                # Liquid processing
                liquid_out, h = self.liquid_cell(spike_features, h)
                if liquid_out.dim() == 3:
                    liquid_out = liquid_out.squeeze(1)
                
                # Track liquid output spikes for STDP if enabled
                if self.stdp_rule is not None and self.training:
                    idx = self.spike_history_idx % self.spike_history_length
                    if liquid_out.size(0) == self.post_spike_history.size(0):
                        self.post_spike_history[:, idx, :] = liquid_out.detach()
                    self.spike_history_idx += 1
                
                # Fusion of spike and liquid features
                combined = torch.cat([liquid_out, spike_features], dim=-1)
                block_output = self.fusion(combined)
                
                # Apply residual connection
                if self.residual_type == 'addition':
                    # Addition-based residual (preferred for temporal coding)
                    gated_residual = self.temporal_gate(token_residual) * token_residual
                    output = block_output + gated_residual
                    
                elif self.residual_type == 'concatenation':
                    # Concatenation-based residual
                    concatenated = torch.cat([block_output, token_residual], dim=-1)
                    output = self.residual_fusion(concatenated)
                
                output_tensor[:, t, :] = output
                
                # Store internals for analysis
                if return_internals:
                    internals['spike_outputs'].append(spike_features.detach())
                    internals['liquid_outputs'].append(liquid_out.detach())
                    if self.potential_norm is not None:
                        internals['membrane_potentials'].append(normalized_potential.detach())
            
            if return_internals:
                return output_tensor, h, internals
            return output_tensor, h
            
        else:
            # Non-sequence processing
            spike_out = self.spike_encoder(x)
            spike_features = spike_out.mean(dim=1)
            
            # Apply potential normalization
            if self.potential_norm is not None:
                membrane_potential = getattr(self.spike_encoder, 'membrane_potential', spike_features)
                normalized_potential = self.potential_norm(membrane_potential)
                spike_features = spike_features + 0.1 * normalized_potential
            
            liquid_out, h_new = self.liquid_cell(spike_features, h)
            if liquid_out.dim() == 3:
                liquid_out = liquid_out.squeeze(1)
            
            combined = torch.cat([liquid_out, spike_features], dim=-1)
            block_output = self.fusion(combined)
            
            # Apply residual connection
            if self.residual_type == 'addition':
                gated_residual = self.temporal_gate(residual) * residual
                output = block_output + gated_residual
            elif self.residual_type == 'concatenation':
                concatenated = torch.cat([block_output, residual], dim=-1)
                output = self.residual_fusion(concatenated)
            
            if return_internals:
                internals = {
                    'spike_outputs': [spike_features.detach()],
                    'liquid_outputs': [liquid_out.detach()],
                }
                if self.potential_norm is not None:
                    internals['membrane_potentials'] = [normalized_potential.detach()]
                return output, h_new, internals
            
            return output, h_new
    
    def apply_stdp_update(self):
        """
        Apply STDP weight updates based on collected spike history.
        Should be called periodically during training (not every forward pass).
        """
        if self.stdp_rule is None or not self.training:
            return
        
        if self.spike_history_idx < 10:
            # Not enough history yet
            return
        
        try:
            # Get recent spike history
            history_len = min(self.spike_history_idx, self.spike_history_length)
            pre_spikes = self.pre_spike_history[:, :history_len, :]
            post_spikes = self.post_spike_history[:, :history_len, :]
            
            # Apply STDP to fusion layer weights (main learning location)
            if hasattr(self.fusion[0], 'weight'):
                weights = self.fusion[0].weight
                # Compute weight updates
                weight_update = self.stdp_rule.compute_weight_update(
                    pre_spikes, post_spikes, weights, dt=1.0
                )
                # Apply updates
                with torch.no_grad():
                    updated_weights = self.stdp_rule.apply_weight_update(
                        weights, weight_update
                    )
                    self.fusion[0].weight.copy_(updated_weights)
            
            # Reset spike history counter
            self.spike_history_idx = 0
            
        except Exception as e:
            logger.warning(f"STDP update failed: {e}")


class HybridLiquidSpikingBlock(nn.Module):
    def __init__(self, input_dim, liquid_units, spiking_units, spike_steps, beta=0.95, backbone='cfc'):
        super().__init__()
        self.spike_encoder = SpikingEncoder(input_dim, spiking_units, spike_steps, beta)
        
        # Store dimensions for proper initialization in enhanced_forward
        self.liquid_units = liquid_units
        self.spiking_units = spiking_units
        
        # Create liquid cell with the correct input dimension
        # In normal mode: takes spiking_units as input
        # In enhanced mode: will handle liquid_units input dynamically
        self.liquid_cell = LiquidCell(spiking_units, liquid_units, backbone)
        
        self.fusion = nn.Sequential(
            nn.Linear(liquid_units + spiking_units, input_dim),  # Output back to input_dim
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.output_dim = input_dim  # Match input dimension for residuals
        
    def forward(self, x, h=None):
        # Handle sequence input for LLM with improved memory management
        if len(x.shape) == 3:  # [batch, seq_len, features]
            batch_size, seq_len, features = x.shape
            
            # Use memory manager for safe tensor allocation
            output_tensor = safe_zeros((batch_size, seq_len, self.output_dim), 
                                      device=x.device, dtype=x.dtype)
            
            # Process each timestep
            for t in range(seq_len):
                token_input = x[:, t:t+1, :]  # [batch, 1, features]
                spike_out = self.spike_encoder(token_input)
                spike_features = spike_out.mean(dim=1)  # [batch, spiking_units]
                liquid_out, h = self.liquid_cell(spike_features, h)
                if liquid_out.dim() == 3:
                    liquid_out = liquid_out.squeeze(1)  # Remove seq dim if present
                combined = torch.cat([liquid_out, spike_features], dim=-1)
                output = self.fusion(combined)
                output_tensor[:, t, :] = output.squeeze(1) if output.dim() > 2 else output
            
            return output_tensor, h
        else:
            # Original processing for non-sequence input
            spike_out = self.spike_encoder(x)
            spike_features = spike_out.mean(dim=1)
            liquid_out, h_new = self.liquid_cell(spike_features, h)
            if liquid_out.dim() == 3:
                liquid_out = liquid_out.squeeze(1)
            combined = torch.cat([liquid_out, spike_features], dim=-1)
            output = self.fusion(combined)
            return output, h_new

class MultiHeadSpikingAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, spike_steps, beta=0.95):
        super().__init__()
        
        # hidden_dim should already be validated to be divisible by num_heads in ModelConfig
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.spike_steps = spike_steps
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.q_lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.k_lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.v_lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q_mem = self.q_lif.init_leaky()
        k_mem = self.k_lif.init_leaky()
        v_mem = self.v_lif.init_leaky()
        
        # Use memory manager for safe spike accumulator allocation
        q_spike_accumulator = safe_zeros((self.spike_steps, batch_size, self.num_heads, seq_len, self.head_dim), 
                                        device=x.device, dtype=x.dtype)
        k_spike_accumulator = safe_zeros((self.spike_steps, batch_size, self.num_heads, seq_len, self.head_dim), 
                                        device=x.device, dtype=x.dtype)
        v_spike_accumulator = safe_zeros((self.spike_steps, batch_size, self.num_heads, seq_len, self.head_dim), 
                                        device=x.device, dtype=x.dtype)
        
        q_reshaped = q.reshape(-1, self.head_dim)
        k_reshaped = k.reshape(-1, self.head_dim)
        v_reshaped = v.reshape(-1, self.head_dim)
        
        for step in range(self.spike_steps):
            q_spk, q_mem = self.q_lif(q_reshaped, q_mem)
            k_spk, k_mem = self.k_lif(k_reshaped, k_mem)
            v_spk, v_mem = self.v_lif(v_reshaped, v_mem)
            
            q_spike_accumulator[step] = q_spk.view(batch_size, self.num_heads, seq_len, self.head_dim)
            k_spike_accumulator[step] = k_spk.view(batch_size, self.num_heads, seq_len, self.head_dim)
            v_spike_accumulator[step] = v_spk.view(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Use mean instead of stack to reduce memory usage
        q_agg = q_spike_accumulator.mean(0)
        k_agg = k_spike_accumulator.mean(0)
        v_agg = v_spike_accumulator.mean(0)
        
        # Compute attention scores
        scores = torch.matmul(q_agg, k_agg.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_agg)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        return output

class LiquidSpikingNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.task_type = config.task_type
        
        # Task-specific input processing
        if self.task_type == TaskType.LLM:
            # Token embeddings for LLM using configurable parameters
            self.token_embedding = nn.Embedding(config.vocab_size or config.output_dim, config.embedding_dim)
            self.position_embedding = nn.Embedding(config.max_position_embeddings, config.embedding_dim)
            self.embedding_dropout = nn.Dropout(config.embedding_dropout)
            
        elif self.task_type == TaskType.VISION:
            # Build configurable convolutional encoder
            conv_layers = []
            in_channels = 3
            
            for i, (out_channels, kernel_size, stride, padding) in enumerate(zip(
                config.conv_channels, config.conv_kernel_sizes, config.conv_strides, config.conv_padding
            )):
                conv_layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2) if i < len(config.conv_channels) - 1 else nn.AdaptiveAvgPool2d((4, 4))
                ])
                in_channels = out_channels
            
            conv_layers.extend([
                nn.Flatten(),
                nn.Linear(config.conv_channels[-1] * 16, config.input_dim)
            ])
            
            self.conv_encoder = nn.Sequential(*conv_layers)
        
        # Input projection layer
        if self.task_type == TaskType.LLM:
            # For LLM, project from embedding dimension to hidden dimension
            self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        else:
            # For other tasks, project from input dimension to hidden dimension
            self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        self.hybrid_blocks = nn.ModuleList([
            ResidualLiquidSpikingBlock(
                config.hidden_dim if i > 0 else config.hidden_dim,
                config.liquid_units,
                config.spiking_units,
                config.num_spike_steps,
                config.beta,
                config.liquid_backbone,
                residual_type='addition',  # Use addition-based residual connections
                use_potential_norm=True   # Enable potential-based normalization
            )
            for i in range(config.num_layers)
        ])
        
        # Add spike decoder for probability conversion
        self.spike_decoder = SpikeDecoder(
            input_dim=config.spiking_units,
            output_dim=config.hidden_dim,
            decode_method='hybrid',  # Use hybrid decoding for best performance
            temporal_window=config.num_spike_steps,
            use_uncertainty=True
        )
        
        self.attention_layers = nn.ModuleList([
            MultiHeadSpikingAttention(
                config.hidden_dim,  # Use hidden_dim for consistency
                num_heads=config.num_attention_heads,
                spike_steps=max(1, config.sequence_length // config.num_attention_heads),
                beta=config.beta
            )
            for _ in range(config.num_layers // 2)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)  # Use hidden_dim after projection
            for _ in range(config.num_layers)
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        
        if self.task_type == TaskType.LLM:
            self.output_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                nn.GELU(),
                nn.LayerNorm(config.hidden_dim * 2),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 2, config.output_dim)
            )
        elif self.task_type == TaskType.VISION:
            self.output_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(config.hidden_dim),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim)
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(config.hidden_dim // 2, config.output_dim)
            )
        
        # STDP Integration (NEW)
        self.stdp_rule = None
        if config.use_stdp:
            # Import plasticity modules
            from .plasticity import (
                ClassicalSTDP, TripletSTDP, HomeostaticSTDP, BCMRule
            )
            
            # Create STDP rule based on config
            if config.stdp_type == 'classical':
                self.stdp_rule = ClassicalSTDP(
                    learning_rate=config.stdp_learning_rate,
                    tau_plus=config.stdp_tau_plus,
                    tau_minus=config.stdp_tau_minus
                )
            elif config.stdp_type == 'triplet':
                self.stdp_rule = TripletSTDP(
                    learning_rate=config.stdp_learning_rate,
                    tau_plus=config.stdp_tau_plus,
                    tau_minus=config.stdp_tau_minus
                )
            elif config.stdp_type == 'homeostatic':
                self.stdp_rule = HomeostaticSTDP(
                    learning_rate=config.stdp_learning_rate,
                    tau_plus=config.stdp_tau_plus,
                    tau_minus=config.stdp_tau_minus,
                    target_rate=config.stdp_target_rate
                )
            elif config.stdp_type == 'bcm':
                self.stdp_rule = BCMRule(
                    learning_rate=config.stdp_learning_rate
                )
            
            # Pass STDP rule to blocks that should use it
            if config.stdp_layers_to_enhance:
                for idx in config.stdp_layers_to_enhance:
                    if idx < len(self.hybrid_blocks):
                        self.hybrid_blocks[idx].stdp_rule = self.stdp_rule
                        # Re-initialize spike history buffers with correct batch size
                        self.hybrid_blocks[idx].register_buffer(
                            'pre_spike_history',
                            torch.zeros(1, 100, config.spiking_units)
                        )
                        self.hybrid_blocks[idx].register_buffer(
                            'post_spike_history',
                            torch.zeros(1, 100, config.liquid_units)
                        )
                        # Initialize tracking variables
                        self.hybrid_blocks[idx].spike_history_idx = 0
                        self.hybrid_blocks[idx].spike_history_length = 100
            else:
                # Apply to all blocks
                for block in self.hybrid_blocks:
                    block.stdp_rule = self.stdp_rule
                    block.register_buffer(
                        'pre_spike_history',
                        torch.zeros(1, 100, config.spiking_units)
                    )
                    block.register_buffer(
                        'post_spike_history',
                        torch.zeros(1, 100, config.liquid_units)
                    )
                    # Initialize tracking variables
                    block.spike_history_idx = 0
                    block.spike_history_length = 100
            
            logger.info(f"🧠 STDP enabled: {config.stdp_type}")
        
        # Meta-Plasticity Integration (NEW)
        self.meta_controller = None
        if config.use_meta_plasticity:
            from .plasticity import MetaPlasticityController
            
            self.meta_controller = MetaPlasticityController(
                num_layers=config.num_layers,
                hidden_dim=config.meta_hidden_dim,
                history_length=config.meta_history_length,
                meta_lr=config.meta_lr
            )
            logger.info("🧠 Meta-plasticity enabled")
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Advanced weight initialization for better convergence."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use different initializations based on layer purpose
                if 'output_head' in name or 'out_proj' in name:
                    # Output layers: smaller initialization for stability
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                    # Attention layers: Xavier with appropriate gain
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                elif 'fusion' in name or 'fallback' in name:
                    # Fusion layers: He initialization for better gradient flow
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                else:
                    # Standard layers: He initialization
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                
                if module.bias is not None:
                    # Small positive bias for better initial activation
                    nn.init.constant_(module.bias, 0.01)
                    
            elif isinstance(module, nn.Embedding):
                # Improved embedding initialization
                nn.init.normal_(module.weight, mean=0, std=0.02)
                # Zero out padding token if present (index 0)
                if hasattr(self, 'token_embedding') and module is self.token_embedding:
                    with torch.no_grad():
                        module.weight[0].fill_(0)
                        
            elif isinstance(module, nn.Conv2d):
                # He initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d)):
                # Better normalization initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
            elif hasattr(module, 'weight') and module.weight is not None:
                # Fallback for other modules with weights
                if module.weight.dim() >= 2:
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x):
        """Enhanced forward pass with robust input validation and configuration handling."""
        batch_size = x.shape[0]
        
        if self.task_type == TaskType.LLM:
            # Handle text input with embeddings - with comprehensive validation
            if x.dtype == torch.float:
                x = x.long()  # Convert to long for embedding lookup
            
            seq_len = x.shape[1]
            
            # Critical Fix 1: Validate and clamp token IDs to prevent CUDA assert
            if x.dtype in [torch.long, torch.int64, torch.int32]:
                # Token IDs - ensure they're within vocabulary bounds
                x = torch.clamp(x, 0, self.token_embedding.num_embeddings - 1)
                
                # Critical Fix 2: Handle sequence length vs position embedding mismatch
                max_pos_len = self.position_embedding.num_embeddings
                if seq_len > max_pos_len:
                    logger.warning(f"Sequence length {seq_len} exceeds max position embeddings {max_pos_len}. Truncating sequence.")
                    x = x[:, :max_pos_len]
                    seq_len = max_pos_len
                
                # Token embeddings with bounds checking
                try:
                    token_emb = self.token_embedding(x)  # [batch, seq_len, input_dim]
                except RuntimeError as e:
                    logger.error(f"Token embedding error: {e}")
                    logger.error(f"Token range: {x.min().item()} to {x.max().item()}")
                    logger.error(f"Vocab size: {self.token_embedding.num_embeddings}")
                    # Fallback: clamp more aggressively
                    x = torch.clamp(x, 0, min(self.token_embedding.num_embeddings - 1, 50256))
                    token_emb = self.token_embedding(x)
                
                # Position embeddings with proper range
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                positions = torch.clamp(positions, 0, self.position_embedding.num_embeddings - 1)
                
                try:
                    pos_emb = self.position_embedding(positions)
                except RuntimeError as e:
                    logger.error(f"Position embedding error: {e}")
                    logger.error(f"Position range: {positions.min().item()} to {positions.max().item()}")
                    logger.error(f"Max positions: {self.position_embedding.num_embeddings}")
                    # Fallback: use modulo to wrap positions
                    positions = positions % self.position_embedding.num_embeddings
                    pos_emb = self.position_embedding(positions)
                
                # Critical Fix 3: Ensure embedding dimensions match before addition
                if token_emb.shape != pos_emb.shape:
                    logger.warning(f"Embedding shape mismatch: token_emb {token_emb.shape}, pos_emb {pos_emb.shape}")
                    # Adjust pos_emb to match token_emb
                    if pos_emb.shape[1] != token_emb.shape[1]:
                        pos_emb = pos_emb[:, :token_emb.shape[1], :]
                    if pos_emb.shape[2] != token_emb.shape[2]:
                        # Project to correct dimension
                        pos_emb = nn.Linear(pos_emb.shape[2], token_emb.shape[2], device=x.device)(pos_emb)
                
                # Safe embedding addition
                x = token_emb + pos_emb
                x = self.embedding_dropout(x)
            else:
                # Handle non-token input (fallback)
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)
            
        elif self.task_type == TaskType.VISION and len(x.shape) == 4:
            x = self.conv_encoder(x)
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        hidden_states = [None] * self.config.num_layers
        
        # Process through hybrid liquid-spiking blocks with proper residual handling
        for i, (block, norm) in enumerate(zip(self.hybrid_blocks, self.layer_norms)):
            if self.task_type == TaskType.LLM:
                # For LLM, process the entire sequence at once
                residual = x
                result = block(x, hidden_states[i], return_internals=True)
                
                # Handle different return lengths gracefully
                if len(result) >= 3:
                    x = result[0]
                    hidden_states[i] = result[1] 
                    internals = result[2]
                elif len(result) >= 2:
                    x = result[0]
                    hidden_states[i] = result[1]
                    internals = {}
                else:
                    raise ValueError(f"Unexpected number of return values: {len(result)}")
                
                # Apply spike decoder for better probability conversion if we have spike outputs
                if 'spike_outputs' in internals and len(internals['spike_outputs']) > 0:
                    spike_features = torch.stack(internals['spike_outputs']).mean(dim=0)
                    
                    # Ensure spike_features has correct shape for decoder [batch, seq_len, features]
                    if spike_features.dim() == 2:  # [batch, features]
                        spike_features = spike_features.unsqueeze(1)  # [batch, 1, features]
                    
                    try:
                        decoded_probs = self.spike_decoder(spike_features)
                        # Enhance the output with decoded probabilities
                        if decoded_probs.shape == x.shape:
                            x = x + 0.1 * decoded_probs  # Small contribution to maintain stability
                    except Exception as decode_error:
                        # Skip spike enhancement if decoding fails
                        pass
                    
            else:
                residual = x.squeeze(1) if x.dim() == 3 else x
                result = block(x, hidden_states[i], return_internals=True)
                
                # Handle different return lengths gracefully for non-LLM tasks
                if len(result) >= 3:
                    x = result[0]
                    hidden_states[i] = result[1] 
                    internals = result[2]
                elif len(result) >= 2:
                    x = result[0]
                    hidden_states[i] = result[1]
                    internals = {}
                else:
                    raise ValueError(f"Unexpected number of return values: {len(result)}")
                
                # Apply spike decoder for non-LLM tasks as well
                if 'spike_outputs' in internals and len(internals['spike_outputs']) > 0:
                    spike_features = internals['spike_outputs'][0]  # Single timestep for non-sequence tasks
                    
                    # Ensure spike_features has correct shape for decoder [batch, seq_len, features]
                    if spike_features.dim() == 2:  # [batch, features]
                        spike_features = spike_features.unsqueeze(1)  # [batch, 1, features]
                    
                    try:
                        decoded_probs = self.spike_decoder(spike_features)
                        if decoded_probs.shape == x.shape:
                            x = x + 0.1 * decoded_probs
                    except Exception as decode_error:
                        # Skip spike enhancement if decoding fails
                        pass
            
            x = norm(x)
            x = self.dropout(x)
            
            # Apply attention
            if i % 2 == 1 and i // 2 < len(self.attention_layers):
                if self.task_type == TaskType.LLM:
                    x = x + self.attention_layers[i // 2](x)
                else:
                    attn_input = x.unsqueeze(1) if x.dim() == 2 else x
                    x = x + self.attention_layers[i // 2](attn_input).squeeze(1)
        
        output = self.output_head(x)
        return output
    
    def apply_stdp_to_all_blocks(self):
        """
        Apply STDP updates to all blocks that have STDP enabled.
        Should be called periodically during training (e.g., every N batches).
        """
        if self.stdp_rule is None:
            return
        
        for block in self.hybrid_blocks:
            if hasattr(block, 'stdp_rule') and block.stdp_rule is not None:
                block.apply_stdp_update()
    
    def update_meta_plasticity(self, performance: float, loss: float, 
                               layer_activities: List = None):
        """
        Update meta-plasticity controller based on current performance.
        
        Args:
            performance: Current task performance (accuracy, etc.)
            loss: Current loss value
            layer_activities: List of layer activations (optional)
        """
        if self.meta_controller is None:
            return None
        
        # Calculate average weight change
        weight_change = 0.0
        param_count = 0
        for param in self.parameters():
            if param.requires_grad and param.grad is not None:
                weight_change += param.grad.abs().mean().item()
                param_count += 1
        
        if param_count > 0:
            weight_change /= param_count
        
        # Update history
        self.meta_controller.update_history(
            performance=performance,
            loss=loss,
            weight_change=weight_change
        )
        
        # Get predicted plasticity parameters
        predicted_params = self.meta_controller.predict_plasticity_parameters()
        
        # Update STDP rules with meta-learned parameters
        if self.stdp_rule is not None:
            for i, block in enumerate(self.hybrid_blocks):
                if hasattr(block, 'stdp_rule') and block.stdp_rule is not None:
                    # Update STDP parameters from meta-controller
                    if i < predicted_params['learning_rate'].size(0):
                        block.stdp_rule.learning_rate = (
                            predicted_params['learning_rate'][i].item()
                        )
                        if hasattr(block.stdp_rule, 'tau_plus'):
                            block.stdp_rule.tau_plus = (
                                predicted_params['tau_plus'][i].item()
                            )
                        if hasattr(block.stdp_rule, 'tau_minus'):
                            block.stdp_rule.tau_minus = (
                                predicted_params['tau_minus'][i].item()
                            )
        
        # Compute meta-loss for meta-learning
        meta_loss = self.meta_controller.compute_meta_loss(
            predicted_params, performance, target_performance=1.0
        )
        
        return meta_loss


class OptimizationEnhancement(nn.Module):
    """
    Advanced optimization enhancement module for liquid spiking networks.
    Provides adaptive gradient clipping, regularization, and learning rate scheduling.
    """
    
    def __init__(self, model_parameters, base_lr=1e-4):
        super().__init__()
        self.model_parameters = list(model_parameters)
        self.base_lr = base_lr
        
        # Adaptive gradient clipping parameters
        self.register_buffer('grad_norm_history', torch.zeros(100))  # Last 100 gradient norms
        self.register_buffer('step_count', torch.tensor(0))
        
        # Regularization parameters
        self.l1_weight = nn.Parameter(torch.tensor(1e-5))
        self.l2_weight = nn.Parameter(torch.tensor(1e-4))
        
        # Learning rate adaptation
        self.register_buffer('loss_history', torch.zeros(50))  # Last 50 losses
        self.register_buffer('lr_adaptation_step', torch.tensor(0))
        
    def adaptive_gradient_clipping(self):
        """
        Compute adaptive gradient clipping value based on gradient norm history.
        Returns: (current_grad_norm, clip_value)
        """
        # Compute current gradient norm
        total_norm = 0
        for p in self.model_parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        current_grad_norm = total_norm ** (1. / 2)
        
        # Update gradient norm history
        if self.step_count < 100:
            self.grad_norm_history[self.step_count] = current_grad_norm
        else:
            # Shift history and add new value
            self.grad_norm_history[:-1] = self.grad_norm_history[1:]
            self.grad_norm_history[-1] = current_grad_norm
        
        self.step_count += 1
        
        # Compute adaptive clip value
        if self.step_count >= 10:  # Need some history
            valid_history = self.grad_norm_history[:min(self.step_count, 100)]
            mean_norm = valid_history.mean()
            std_norm = valid_history.std()
            
            # Clip at mean + 2*std, but with bounds
            clip_value = torch.clamp(mean_norm + 2 * std_norm, min=0.1, max=10.0)
        else:
            clip_value = torch.tensor(1.0)
        
        return current_grad_norm, clip_value.item()
    
    def compute_adaptive_regularization(self):
        """
        Compute adaptive L1 and L2 regularization loss.
        Returns: regularization_loss
        """
        l1_loss = 0
        l2_loss = 0
        
        for p in self.model_parameters:
            if p.requires_grad:
                l1_loss += torch.sum(torch.abs(p))
                l2_loss += torch.sum(p * p)
        
        # Adaptive weights based on parameter magnitudes
        total_params = sum(p.numel() for p in self.model_parameters if p.requires_grad)
        norm_factor = 1.0 / (total_params + 1e-8)
        
        reg_loss = (self.l1_weight * l1_loss + self.l2_weight * l2_loss) * norm_factor
        return reg_loss
    
    def adaptive_learning_rate(self, current_loss, epoch):
        """
        Compute adaptive learning rate based on loss history and training progress.
        Returns: new_learning_rate
        """
        # Update loss history
        if self.lr_adaptation_step < 50:
            self.loss_history[self.lr_adaptation_step] = current_loss
        else:
            # Shift history and add new value
            self.loss_history[:-1] = self.loss_history[1:]
            self.loss_history[-1] = current_loss
        
        self.lr_adaptation_step += 1
        
        if self.lr_adaptation_step >= 10:  # Need some history
            valid_history = self.loss_history[:min(self.lr_adaptation_step, 50)]
            
            # Check for improvement trend
            recent_losses = valid_history[-5:]  # Last 5 losses
            earlier_losses = valid_history[-10:-5] if len(valid_history) >= 10 else valid_history[:-5]
            
            if len(earlier_losses) > 0:
                recent_mean = recent_losses.mean()
                earlier_mean = earlier_losses.mean()
                
                # Adaptive learning rate adjustment
                if recent_mean < earlier_mean:
                    # Loss is improving, slightly increase LR
                    lr_multiplier = 1.02
                elif recent_mean > earlier_mean * 1.1:
                    # Loss is getting worse, decrease LR
                    lr_multiplier = 0.95
                else:
                    # Stable, keep current LR
                    lr_multiplier = 1.0
            else:
                lr_multiplier = 1.0
        else:
            lr_multiplier = 1.0
        
        # Apply epoch-based decay
        epoch_decay = 0.95 ** (epoch // 10)
        
        new_lr = self.base_lr * lr_multiplier * epoch_decay
        return max(new_lr, self.base_lr * 0.01)  # Don't go below 1% of base LR


class LiquidSpikingTrainer:
    def __init__(self, model, config: ModelConfig):
        self.model = model
        self.config = config
        
        # Setup multi-GPU training environment
        self.multi_gpu_manager, device, self.gpu_ids = setup_multi_gpu_environment(
            strategy=config.multi_gpu_strategy,
            gpu_ids=config.gpu_ids
        )
        
        # Override device from multi-GPU setup
        self.device = torch.device(device)
        self.config.device = device
        
        # Setup distributed training if needed
        if self.multi_gpu_manager.is_distributed:
            self.multi_gpu_manager.initialize_distributed_process_group()
        
        # Move model to device
        self.model.to(self.device)
        
        # Wrap model for multi-GPU training
        if len(self.gpu_ids) > 1:
            strategy = MultiGPUStrategy(config.multi_gpu_strategy)
            if strategy == MultiGPUStrategy.AUTO:
                strategy = self.multi_gpu_manager.config.strategy
            
            self.model = self.multi_gpu_manager.wrap_model_for_multi_gpu(
                self.model, strategy, self.gpu_ids
            )
            
            # Adjust batch size for multi-GPU
            original_batch_size = config.batch_size
            config.batch_size = self.multi_gpu_manager.adjust_batch_size_for_multi_gpu(
                config.batch_size, len(self.gpu_ids)
            )
            
            if config.batch_size != original_batch_size:
                logging.info(f"Adjusted batch size from {original_batch_size} to {config.batch_size} for multi-GPU training")
        
        # Advanced optimizer with better parameter settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),  # Optimized beta values
            eps=1e-8,
            amsgrad=True  # Better convergence for sparse gradients
        )
        
        # Advanced learning rate scheduling with warmup
        self.warmup_epochs = max(1, config.num_epochs // 20)  # 5% warmup
        self.total_epochs = getattr(config, 'num_epochs', 100)
        
        # Create combined scheduler: warmup + cosine annealing with restarts
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return epoch / self.warmup_epochs
            else:
                # Cosine annealing with restarts
                cycle_length = (self.total_epochs - self.warmup_epochs) // 3
                if cycle_length < 1:
                    cycle_length = self.total_epochs - self.warmup_epochs
                
                epoch_in_cycle = (epoch - self.warmup_epochs) % cycle_length
                return 0.5 * (1 + math.cos(math.pi * epoch_in_cycle / cycle_length))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Backup scheduler for plateau
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Task-specific loss functions with optimizations
        if config.task_type == TaskType.LLM:
            # Label smoothing for better generalization
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=0.1,
                reduction='mean'
            )
        elif config.task_type == TaskType.VISION:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        else:
            # Huber loss for robotics (more robust than MSE)
            self.criterion = nn.SmoothL1Loss(reduction='mean')
        
        # Enhanced mixed precision with better settings
        self.scaler = GradScaler(
            device='cuda' if 'cuda' in self.config.device else 'cpu',
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=100
        ) if config.mixed_precision else None
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 10
        
        # Gradient accumulation for effective larger batch sizes
        self.accumulation_steps = max(1, 32 // config.batch_size)
        
        # EMA for model weights (better generalization)
        self.ema_decay = 0.999
        self.ema_model = None
        self._init_ema()
        
        # Multi-GPU specific settings
        self.is_main_process = self.multi_gpu_manager.should_save_checkpoint()
        self.world_size = len(self.gpu_ids) if self.gpu_ids else 1
        
        # Initialize memory manager for leak prevention
        self.memory_manager = SpikingMemoryManager(
            cleanup_threshold_mb=1000.0,  # 1GB threshold
            auto_cleanup=True,
            max_spike_steps=getattr(config, 'num_spike_steps', 32)  # Default to 32 if not set
        )
        
        # Initialize optimization enhancements
        self.optimization_enhancement = OptimizationEnhancement(
            self.model.parameters(), 
            base_lr=config.learning_rate
        ).to(self.device)
        
        # Continual Learning Integration (NEW)
        self.continual_system = None
        self.replay_buffer = None
        
        if config.use_continual_learning:
            from .plasticity import ContinualLearningSTDP, TaskBuffer
            
            self.continual_system = ContinualLearningSTDP(
                model=self.model,
                consolidation_strength=config.consolidation_strength,
                plasticity_decay=config.plasticity_decay,
                use_meta_plasticity=config.use_meta_plasticity,
                meta_lr=config.meta_lr
            )
            
            if config.use_experience_replay:
                self.replay_buffer = TaskBuffer(
                    buffer_size=config.replay_buffer_size,
                    sampling_strategy=config.replay_sampling_strategy
                )
            
            # Track task performance for continual learning
            self.task_performance = {}
            self.compute_importance_every = config.compute_importance_interval
            
            logger.info("🧠 Continual learning enabled")
            logger.info(f"   Consolidation strength: {config.consolidation_strength}")
            if config.use_experience_replay:
                logger.info(f"   Experience replay: {config.replay_buffer_size} examples")
        
        # Add enhanced forward method to hybrid blocks
        for block in self.model.hybrid_blocks:
            if hasattr(block, 'enhanced_forward'):
                # Replace forward with enhanced version
                block.original_forward = block.forward
                
                def create_enhanced_wrapper(block_instance, memory_manager):
                    def wrapper(x, h=None):
                        return block_instance.enhanced_forward(x, memory_manager)
                    return wrapper
                
                block.forward = create_enhanced_wrapper(block, self.memory_manager)
        
        # Log initial memory state
        self.memory_manager.log_memory_usage("Trainer initialization with optimizations")
    
    def _init_ema(self):
        """Initialize Exponential Moving Average of model parameters."""
        self.ema_model = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_model[name] = param.data.clone()
    
    def _update_ema(self):
        """Update EMA parameters with memory leak prevention."""
        if not hasattr(self, 'ema_model') or self.ema_model is None:
            return
            
        with torch.no_grad():  # Prevent gradient tracking for memory efficiency
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema_model:
                    # In-place update to prevent memory accumulation
                    self.ema_model[name].mul_(self.ema_decay).add_(
                        param.data, alpha=(1 - self.ema_decay)
                    )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        gradient_norm_sum = 0
        
        # Only show progress bar on main process for distributed training
        show_progress = not self.multi_gpu_manager.is_distributed or self.is_main_process
        progress_bar = tqdm(train_loader, desc="Training", disable=not show_progress) if hasattr(train_loader, '__len__') else train_loader
        
        # Reset gradient accumulation
        accumulated_loss = 0
        
        # Memory leak prevention: Track and clear GPU memory periodically
        memory_cleanup_interval = 50  # Clean memory every 50 batches
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, targets = batch
            elif isinstance(batch, dict):
                # Handle dict-based batches
                data = batch.get('input_ids', batch.get('data'))
                targets = batch.get('labels', batch.get('targets', data))
            else:
                # Assume batch is the data itself
                data = batch
                targets = data
            
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision and self.scaler:
                with autocast('cuda'):
                    outputs = self.model(data)
                    task_loss = self._compute_loss(outputs, targets)
                    
                    # Add consolidation loss for continual learning (NEW)
                    if self.continual_system is not None:
                        consolidation_loss = self.continual_system.compute_consolidation_loss()
                        loss = task_loss + consolidation_loss
                    else:
                        loss = task_loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
                # Gradient accumulation step
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping with scaled gradients
                    if self.config.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.gradient_clip
                        )
                        gradient_norm_sum += grad_norm.item()
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update EMA
                    self._update_ema()
                    
                    total_loss += accumulated_loss
                    accumulated_loss = 0
                    num_batches += 1
            else:
                # Standard precision training
                outputs = self.model(data)
                task_loss = self._compute_loss(outputs, targets)
                
                # Add consolidation loss for continual learning (NEW)
                if self.continual_system is not None:
                    consolidation_loss = self.continual_system.compute_consolidation_loss()
                    loss = task_loss + consolidation_loss
                else:
                    loss = task_loss
                
                loss = loss / self.accumulation_steps
                
                loss.backward()
                accumulated_loss += loss.item()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Apply enhanced optimization techniques
                    if hasattr(self, 'optimization_enhancement'):
                        # Adaptive gradient clipping
                        grad_norm, clip_value = self.optimization_enhancement.adaptive_gradient_clipping()
                        gradient_norm_sum += grad_norm
                        
                        # Add adaptive regularization to loss
                        reg_loss = self.optimization_enhancement.compute_adaptive_regularization()
                        loss = loss + reg_loss
                    elif self.config.gradient_clip > 0:
                        # Fallback to standard gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.gradient_clip
                        )
                        gradient_norm_sum += grad_norm.item()
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._update_ema()
                    
                    # Apply STDP updates (NEW) - do this periodically
                    if self.config.use_stdp and batch_idx % 10 == 0:
                        if hasattr(self.model, 'apply_stdp_to_all_blocks'):
                            self.model.apply_stdp_to_all_blocks()
                    
                    # Experience replay for continual learning (NEW)
                    if self.replay_buffer is not None and len(self.replay_buffer.examples) > 0:
                        replay_batch_size = max(1, data.size(0) // 4)
                        replay_samples = self.replay_buffer.sample(batch_size=replay_batch_size)
                        
                        if len(replay_samples) > 0:
                            replay_data = torch.stack([s[0] for s in replay_samples]).to(self.device)
                            replay_targets = torch.stack([s[1] for s in replay_samples]).to(self.device)
                            
                            replay_outputs = self.model(replay_data)
                            replay_loss = self._compute_loss(replay_outputs, replay_targets)
                            replay_loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    
                    total_loss += accumulated_loss
                    accumulated_loss = 0
                    num_batches += 1
            
            # Memory cleanup to prevent leaks
            if batch_idx % memory_cleanup_interval == 0 and batch_idx > 0:
                # Use memory manager for cleanup
                self.memory_manager.cleanup_memory()
                # Explicit cleanup of intermediate tensors
                del data, targets, outputs, loss
                    
            # Update progress bar with detailed metrics (only on main process)
            if show_progress and hasattr(progress_bar, 'set_postfix') and num_batches > 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_grad_norm = gradient_norm_sum / num_batches if num_batches > 0 else 0
                progress_bar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'grad_norm': f'{avg_grad_norm:.3f}',
                    'gpus': len(self.gpu_ids) if self.gpu_ids else 0
                })
        
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
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self._update_ema()
            total_loss += accumulated_loss
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_grad_norm = gradient_norm_sum / max(num_batches, 1)
        
        # Aggregate metrics across all processes for distributed training
        if self.multi_gpu_manager.is_distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            grad_norm_tensor = torch.tensor(avg_grad_norm, device=self.device)
            
            avg_loss = self.multi_gpu_manager.all_reduce_tensor(loss_tensor).item()
            avg_grad_norm = self.multi_gpu_manager.all_reduce_tensor(grad_norm_tensor).item()
        
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        # End-of-epoch memory cleanup
        self.memory_manager.cleanup_memory()
        self.memory_manager.log_memory_usage("End of training epoch")
        
        return avg_loss, avg_grad_norm
    
    def _compute_loss(self, outputs, targets):
        """Compute loss with task-specific optimizations."""
        if self.config.task_type == TaskType.LLM:
            # Reshape for cross-entropy loss
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            
            # Ignore padding tokens
            valid_indices = targets != -100
            if valid_indices.any():
                outputs = outputs[valid_indices]
                targets = targets[valid_indices]
            else:
                # Fallback if all tokens are padding
                return torch.tensor(0.0, device=outputs.device, requires_grad=True)
            
            loss = self.criterion(outputs, targets)
            
            # Add regularization for stability
            if hasattr(self.model, 'token_embedding'):
                # Embedding regularization
                embed_reg = 0.01 * torch.norm(self.model.token_embedding.weight, p=2)
                loss = loss + embed_reg
                
        elif self.config.task_type == TaskType.VISION:
            loss = self.criterion(outputs, targets)
        else:
            # Robotics task
            loss = self.criterion(outputs, targets)
            
        return loss
    
    def validate(self, val_loader, use_ema=True):
        """Enhanced validation with EMA model option and distributed support."""
        # Temporarily apply EMA weights if requested
        original_state = None
        if use_ema and self.ema_model:
            original_state = {}
            for name, param in self.model.named_parameters():
                if name in self.ema_model:
                    original_state[name] = param.data.clone()
                    param.data.copy_(self.ema_model[name])
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        total_correct = 0
        total_samples = 0
        
        # Only show progress on main process for distributed training
        show_progress = not self.multi_gpu_manager.is_distributed or self.is_main_process
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", disable=not show_progress):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, targets = batch
                elif isinstance(batch, dict):
                    # Handle dict-based batches
                    data = batch.get('input_ids', batch.get('data'))
                    targets = batch.get('labels', batch.get('targets', data))
                else:
                    # Assume batch is the data itself
                    data = batch
                    targets = data
                    
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(data)
                loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate accuracy for classification tasks
                if self.config.task_type in [TaskType.VISION, TaskType.LLM]:
                    if self.config.task_type == TaskType.LLM:
                        # For LLM, only count non-padding tokens
                        batch_size, seq_len, vocab_size = outputs.shape
                        outputs_flat = outputs.reshape(-1, vocab_size)
                        targets_flat = targets.reshape(-1)
                        
                        valid_indices = targets_flat != -100
                        if valid_indices.any():
                            predictions = outputs_flat[valid_indices].argmax(dim=-1)
                            correct = (predictions == targets_flat[valid_indices]).sum().item()
                            total_correct += correct
                            total_samples += valid_indices.sum().item()
                    else:
                        # Vision classification
                        predictions = outputs.argmax(dim=-1)
                        correct = (predictions == targets).sum().item()
                        total_correct += correct
                        total_samples += targets.size(0)
        
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_samples, 1) if total_samples > 0 else 0.0
        
        # Aggregate metrics across all processes for distributed training
        if self.multi_gpu_manager.is_distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            correct_tensor = torch.tensor(total_correct, device=self.device, dtype=torch.float)
            samples_tensor = torch.tensor(total_samples, device=self.device, dtype=torch.float)
            
            avg_loss = self.multi_gpu_manager.all_reduce_tensor(loss_tensor).item()
            total_correct = self.multi_gpu_manager.all_reduce_tensor(correct_tensor).item()
            total_samples = self.multi_gpu_manager.all_reduce_tensor(samples_tensor).item()
            
            accuracy = total_correct / max(total_samples, 1) if total_samples > 0 else 0.0
        
        # Restore original weights if EMA was used
        if original_state:
            for name, param in self.model.named_parameters():
                if name in original_state:
                    param.data.copy_(original_state[name])
        
        self.val_losses.append(avg_loss)
        
        # Update best model tracking
        is_best = avg_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = avg_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return avg_loss, accuracy, is_best
    
    def train(self, train_loader, val_loader, num_epochs):
        """Enhanced training loop with multi-GPU support and advanced features."""
        self.total_epochs = num_epochs
        
        # Only print on main process for distributed training
        if self.is_main_process:
            print(f"\n🚀 Starting training for {num_epochs} epochs")
            print(f"   💡 Using {len(self.gpu_ids) if self.gpu_ids else 0} GPUs")
            print(f"   🎯 Strategy: {self.config.multi_gpu_strategy}")
            print(f"   📦 Batch size: {self.config.batch_size}")
            if len(self.gpu_ids) > 1:
                print(f"   ⚡ Multi-GPU acceleration enabled!")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Training phase
            train_loss, grad_norm = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_accuracy, is_best = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            self.plateau_scheduler.step(val_loss)
            
            # Apply enhanced optimization techniques
            if hasattr(self, 'optimization_enhancement'):
                # Adaptive learning rate based on loss and epoch
                new_lr = self.optimization_enhancement.adaptive_learning_rate(
                    self.optimizer, val_loss, epoch
                )
                
                # Update surrogate gradient temperatures in model
                for block in self.model.hybrid_blocks:
                    if hasattr(block, 'surrogate_grad') and hasattr(block.surrogate_grad, 'update_temperature'):
                        block.surrogate_grad.update_temperature(epoch, num_epochs)
            
            epoch_time = time.time() - epoch_start_time
            
            # Only log and save on main process
            if self.is_main_process:
                # Enhanced logging with multi-GPU info
                gpu_info = f" (GPUs: {len(self.gpu_ids)})" if len(self.gpu_ids) > 1 else ""
                print(f"Epoch {epoch+1}/{num_epochs}{gpu_info}:")
                print(f"  🔥 Train Loss: {train_loss:.4f}")
                print(f"  ✅ Val Loss: {val_loss:.4f}")
                if val_accuracy > 0:
                    print(f"  🎯 Val Accuracy: {val_accuracy:.2%}")
                print(f"  📈 Grad Norm: {grad_norm:.3f}")
                print(f"  ⏱️  Time: {epoch_time:.1f}s")
                print(f"  📚 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Save checkpoint every 5 epochs or if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    print(f"  🏆 New best model! Validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                
                if (epoch + 1) % 5 == 0 or is_best:
                    checkpoint_name = f"checkpoint_epoch_{epoch+1}.pt"
                    if is_best:
                        checkpoint_name = "best_model.pt"
                    self.save_checkpoint(checkpoint_name)
                
                # Early stopping check
                if self.patience_counter >= self.max_patience:
                    print(f"🛑 Early stopping triggered. No improvement for {self.max_patience} epochs.")
                    break
                
                print("-" * 50)
            
            # Synchronize processes
            if self.multi_gpu_manager.is_distributed:
                self.multi_gpu_manager.barrier()
        
        # Final statistics (only on main process)
        if self.is_main_process:
            total_time = time.time() - start_time
            print(f"\n🎊 Training completed!")
            print(f"⏱️  Total time: {total_time/60:.1f} minutes")
            print(f"📈 Best validation loss: {self.best_val_loss:.4f}")
            print(f"🚀 Multi-GPU speedup achieved with {len(self.gpu_ids)} GPUs" if len(self.gpu_ids) > 1 else "")
        
        return self.train_losses, self.val_losses
        print(f"📊 Warmup epochs: {self.warmup_epochs}")
        print(f"📈 Gradient accumulation steps: {self.accumulation_steps}")
        print(f"🔧 Mixed precision: {self.config.mixed_precision}")
        print(f"📝 EMA decay: {self.ema_decay}")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, grad_norm = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_accuracy, is_best = self.validate(val_loader, use_ema=True)
            
            # Learning rate scheduling
            self.scheduler.step()
            # Also update plateau scheduler
            self.plateau_scheduler.step(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Enhanced logging
            print(f"\n📊 Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"   🏃 Train Loss: {train_loss:.4f} | Grad Norm: {grad_norm:.3f}")
            print(f"   ✅ Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.3f}")
            print(f"   📈 LR: {current_lr:.2e} | Patience: {self.patience_counter}/{self.max_patience}")
            
            if is_best:
                print(f"   🎉 New best validation loss: {self.best_val_loss:.4f}")
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pt")
            
            # Early stopping check
            if self.patience_counter >= self.max_patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save periodic checkpoints
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        print(f"\n🎊 Training completed!")
        print(f"📈 Best validation loss: {self.best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, filename):
        """Enhanced checkpoint saving with EMA and additional metrics (main process only)."""
        if not self.is_main_process:
            return  # Only save on main process for distributed training
        
        # Get the raw model state dict (unwrap from DDP/DP if needed)
        model_state_dict = self.model.state_dict()
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state_dict,
            'ema_model': self.ema_model if self.ema_model else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'plateau_scheduler_state_dict': self.plateau_scheduler.state_dict(),
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'epoch': len(self.train_losses),
            'accumulation_steps': self.accumulation_steps,
            'ema_decay': self.ema_decay,
            'gpu_ids': self.gpu_ids,
            'world_size': self.world_size
        }
        torch.save(checkpoint, filename)
        print(f"📁 Enhanced multi-GPU checkpoint saved to {filename}")
        if len(self.gpu_ids) > 1:
            print(f"   🔧 Model unwrapped from {type(self.model).__name__} wrapper")
    
    def load_checkpoint(self, filename):
        """Enhanced checkpoint loading with backward compatibility."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load EMA if available
        if 'ema_model' in checkpoint and checkpoint['ema_model']:
            self.ema_model = checkpoint['ema_model']
        else:
            self._init_ema()  # Reinitialize EMA if not found
        
        # Load optimizer and scheduler states
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'plateau_scheduler_state_dict' in checkpoint:
            self.plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler_state_dict'])
        
        # Load training state
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        # Load scaler state
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load additional parameters
        self.accumulation_steps = checkpoint.get('accumulation_steps', self.accumulation_steps)
        self.ema_decay = checkpoint.get('ema_decay', self.ema_decay)
        
        epoch = checkpoint.get('epoch', 0)
        print(f"📁 Enhanced checkpoint loaded from {filename} (epoch {epoch})")
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load multi-GPU information
        self.gpu_ids = checkpoint.get('gpu_ids', self.gpu_ids)
        self.world_size = checkpoint.get('world_size', self.world_size)
        
        print(f"📁 Enhanced multi-GPU checkpoint loaded from {filename} (epoch {epoch})")
        if len(self.gpu_ids) > 1:
            print(f"   🔧 Checkpoint saved from {checkpoint.get('world_size', 1)} GPU training")
    
    def cleanup(self):
        """Clean up distributed training resources."""
        if hasattr(self, 'multi_gpu_manager'):
            self.multi_gpu_manager.cleanup_distributed()
    
    def train_on_task(self, task_id: int, train_loader, val_loader, 
                     num_epochs: int):
        """
        Train on a specific task with continual learning support.
        
        Args:
            task_id: Unique identifier for the task
            train_loader: DataLoader for task training data
            val_loader: DataLoader for task validation data
            num_epochs: Number of epochs to train
        
        Returns:
            final_accuracy: Validation accuracy after training
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"📚 Training on Task {task_id}")
        logger.info(f"{'='*50}")
        
        # Regular training for this task
        self.train(train_loader, val_loader, num_epochs)
        
        # Compute parameter importance for this task
        if self.continual_system is not None:
            logger.info(f"🔍 Computing parameter importance for Task {task_id}...")
            self.continual_system.compute_parameter_importance(
                train_loader, num_samples=1000
            )
            
            # Consolidate knowledge for this task
            logger.info(f"🔒 Consolidating knowledge for Task {task_id}...")
            self.continual_system.consolidate_task_knowledge()
            
            # Store examples for experience replay
            if self.replay_buffer is not None:
                self._store_task_examples(task_id, train_loader, max_examples=200)
        
        # Evaluate on validation set
        self.model.eval()
        val_accuracy = self.evaluate(val_loader)
        
        # Store task performance
        self.task_performance[task_id] = val_accuracy
        
        logger.info(f"✅ Task {task_id} completed: {val_accuracy:.3f} accuracy")
        
        return val_accuracy
    
    def _store_task_examples(self, task_id: int, dataloader, max_examples: int = 200):
        """
        Store examples from this task for experience replay.
        
        Args:
            task_id: Task identifier
            dataloader: DataLoader for the task
            max_examples: Maximum number of examples to store
        """
        examples = []
        importance_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if len(examples) >= max_examples:
                    break
                
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, targets = batch
                elif isinstance(batch, dict):
                    data = batch.get('input_ids', batch.get('data'))
                    targets = batch.get('labels', batch.get('targets', data))
                else:
                    data = batch
                    targets = data
                
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Get model outputs to compute importance scores
                outputs = self.model(data)
                loss = self._compute_loss(outputs, targets)
                
                # Store examples with their importance (based on loss)
                for i in range(min(data.size(0), max_examples - len(examples))):
                    examples.append((data[i].cpu(), targets[i].cpu()))
                    importance_scores.append(loss.item())
        
        # Add to replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.add_examples(examples, task_id, importance_scores)
            logger.info(f"📦 Stored {len(examples)} examples from Task {task_id}")
    
    def evaluate_all_tasks(self, task_dataloaders: dict):
        """
        Evaluate model on all previous tasks to measure forgetting.
        
        Args:
            task_dataloaders: Dict mapping task_id to validation DataLoader
        
        Returns:
            results: Dict of task_id -> accuracy
            avg_accuracy: Average accuracy across all tasks
            avg_forgetting: Average forgetting compared to initial performance
        """
        logger.info("\n" + "="*50)
        logger.info("📊 Evaluating on all tasks...")
        logger.info("="*50)
        
        results = {}
        forgetting_scores = []
        
        self.model.eval()
        for task_id, val_loader in task_dataloaders.items():
            accuracy = self.evaluate(val_loader)
            results[task_id] = accuracy
            
            # Calculate forgetting if we have initial performance
            if task_id in self.task_performance:
                initial_acc = self.task_performance[task_id]
                forgetting = max(0, initial_acc - accuracy)
                forgetting_scores.append(forgetting)
                logger.info(
                    f"Task {task_id}: {accuracy:.3f} "
                    f"(initial: {initial_acc:.3f}, "
                    f"forgetting: {forgetting:.3f})"
                )
            else:
                logger.info(f"Task {task_id}: {accuracy:.3f}")
        
        avg_accuracy = sum(results.values()) / len(results) if results else 0.0
        avg_forgetting = sum(forgetting_scores) / len(forgetting_scores) if forgetting_scores else 0.0
        
        logger.info(f"\n📈 Average Accuracy: {avg_accuracy:.3f}")
        logger.info(f"🧠 Average Forgetting: {avg_forgetting:.3f}")
        
        return results, avg_accuracy, avg_forgetting

class TextDataset(Dataset):
    """Real text dataset for LLM training with proper tokenization."""
    
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, seq_length: int, stride: int = None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride or max(1, seq_length // 4)  # Smaller stride for more examples
        
        # Tokenize all texts
        self.examples = []
        
        print(f"Processing {len(texts)} texts for training...")
        for text in tqdm(texts, desc="Tokenizing texts"):
            # Skip very short texts
            if len(text.strip()) < 20:
                continue
                
            # Tokenize the text
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Create sliding windows with smaller stride for more examples
            for i in range(0, max(1, len(encoded) - seq_length + 1), self.stride):
                if i + seq_length > len(encoded):
                    break
                    
                input_ids = encoded[i:i + seq_length]
                
                # For causal LM, targets are shifted by 1
                if i + seq_length < len(encoded):
                    target_ids = encoded[i + 1:i + seq_length + 1]
                else:
                    target_ids = encoded[i + 1:] + [self.tokenizer.eos_token_id]
                
                # Pad if necessary
                if len(input_ids) == seq_length and len(target_ids) == seq_length:
                    self.examples.append({
                        'input_ids': input_ids,
                        'target_ids': target_ids
                    })
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return (
            torch.tensor(example['input_ids'], dtype=torch.long),
            torch.tensor(example['target_ids'], dtype=torch.long)
        )

class WikiTextDataset:
    """Download and process WikiText-2 dataset for LLM training."""
    
    @staticmethod
    def load_wikitext2(split='train', cache_dir='./data'):
        """Load WikiText-2 dataset."""
        try:
            # Try to load from HuggingFace datasets
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, cache_dir=cache_dir)
            texts = [example['text'] for example in dataset if len(example['text'].strip()) > 50]
            return texts
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            return WikiTextDataset._download_wikitext2_manual(split, cache_dir)
    
    @staticmethod
    def _download_wikitext2_manual(split='train', cache_dir='./data'):
        """Manual download of WikiText-2 if HuggingFace fails."""
        os.makedirs(cache_dir, exist_ok=True)
        
        # URLs for WikiText-2
        urls = {
            'train': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
        }
        
        if split not in urls:
            print(f"Split {split} not available for manual download. Using sample text.")
            return WikiTextDataset._create_sample_text()
        
        try:
            import zipfile
            zip_path = os.path.join(cache_dir, 'wikitext-2.zip')
            
            if not os.path.exists(zip_path):
                print(f"Downloading WikiText-2...")
                urllib.request.urlretrieve(urls[split], zip_path)
            
            # Extract and read
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
            
            # Read the text file
            text_file = os.path.join(cache_dir, 'wikitext-2', f'wiki.{split}.tokens')
            if os.path.exists(text_file):
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into paragraphs and filter
                texts = [para.strip() for para in content.split('\n\n') if len(para.strip()) > 50]
                return texts
            else:
                print("Failed to find extracted file. Using sample text.")
                return WikiTextDataset._create_sample_text()
                
        except Exception as e:
            print(f"Manual download failed: {e}. Using sample text.")
            return WikiTextDataset._create_sample_text()
    
    @staticmethod
    def _create_sample_text():
        """Create sample text for training if all else fails."""
        sample_texts = [
            "The history of artificial intelligence began in antiquity with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen.",
            "Modern artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding.",
            "Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
            "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.",
            "A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data.",
            "Recurrent neural networks are a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.",
            "Convolutional neural networks are a class of deep neural networks, most commonly applied to analyzing visual imagery.",
            "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment.",
        ]
        
        # Repeat and expand sample texts
        expanded_texts = []
        for i in range(100):  # Create more samples
            for text in sample_texts:
                expanded_texts.append(f"{text} This is sample text number {i+1} for training purposes. " * 3)
        
        return expanded_texts

class DatasetFactory:
    @staticmethod
    def create_llm_dataset(vocab_size=100277, seq_length=128, num_samples=50000, tokenizer_name='gpt4', tokenizer_type=None):
        """Create LLM dataset with robust tokenizer validation and token ID safety."""
        
        # Handle both parameter names for backward compatibility
        tokenizer_name = tokenizer_type or tokenizer_name
        
        logger.info(f"Creating LLM dataset with {tokenizer_name} tokenizer")
        logger.info(f"Target vocab size: {vocab_size:,}, sequence length: {seq_length}")
        
        # Create tokenizer with validation
        if tokenizer_name in ['gpt4', 'gpt3', 'o200k']:
            try:
                from ..core.tokenizer_upgrade import AdvancedTokenizerManager
                tokenizer_manager = AdvancedTokenizerManager(tokenizer_name)
                tokenizer = tokenizer_manager.tokenizer
                actual_vocab_size = tokenizer_manager.get_vocab_size()
                
                # Validate vocab size consistency
                if actual_vocab_size != vocab_size:
                    logger.warning(f"Vocab size mismatch: expected {vocab_size:,}, got {actual_vocab_size:,}")
                    logger.info(f"Using actual tokenizer vocab size: {actual_vocab_size:,}")
                    vocab_size = actual_vocab_size
                
            except ImportError as e:
                logger.warning(f"Advanced tokenizer not available: {e}")
                logger.info("Falling back to GPT-2 tokenizer")
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                vocab_size = len(tokenizer)
                
        else:
            # Use HuggingFace tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name != 'gpt2' else 'gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            vocab_size = len(tokenizer)
        
        logger.info(f"Final tokenizer vocab size: {vocab_size:,}")
        
        # Load dataset with proper error handling
        try:
            from ..datasets.advanced_programming_datasets import ProgrammingDatasetFactory
            
            dataset = ProgrammingDatasetFactory.create_llm_programming_dataset(
                tokenizer=tokenizer,
                sequence_length=seq_length,
                total_samples=num_samples
            )
            
            logger.info(f"Successfully loaded programming dataset: {len(dataset):,} samples")
            
            # Add token validation to dataset
            class ValidatedDataset(Dataset):
                def __init__(self, base_dataset, vocab_size):
                    self.base_dataset = base_dataset
                    self.vocab_size = vocab_size
                
                def __len__(self):
                    return len(self.base_dataset)
                
                def __getitem__(self, idx):
                    item = self.base_dataset[idx]
                    
                    # Validate and clamp token IDs
                    if 'input_ids' in item:
                        input_ids = item['input_ids']
                        if isinstance(input_ids, torch.Tensor):
                            # Clamp token IDs to valid range
                            item['input_ids'] = torch.clamp(input_ids, 0, self.vocab_size - 1)
                        
                    if 'labels' in item:
                        labels = item['labels']
                        if isinstance(labels, torch.Tensor):
                            # Clamp label IDs to valid range  
                            item['labels'] = torch.clamp(labels, 0, self.vocab_size - 1)
                    
                    return item
            
            # Wrap dataset with validation
            validated_dataset = ValidatedDataset(dataset, vocab_size)
            
            return validated_dataset, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load programming dataset: {e}")
            logger.info("Creating fallback text dataset")
            
            # Fallback to simple text dataset
            texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Neural networks are inspired by biological neurons.",
                "Deep learning uses multiple layers to learn representations.",
                "Python is a popular programming language for AI."
            ] * (num_samples // 5)
            
            dataset = TextDataset(texts[:num_samples], tokenizer, seq_length)
            return dataset, tokenizer
    
    @staticmethod
    def _create_fallback_dataset(tokenizer, seq_length, num_samples):
        """Create fallback dataset if advanced datasets fail."""
        # Generate diverse programming and text examples
        examples = [
            # Python programming examples
            '''def fibonacci(n):
    """Calculate the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")''',
            
            '''import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x''',
            
            # JavaScript examples
            '''function quickSort(arr) {
    if (arr.length <= 1) {
        return arr;
    }
    
    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const right = arr.filter(x => x > pivot);
    const middle = arr.filter(x => x === pivot);
    
    return [...quickSort(left), ...middle, ...quickSort(right)];
}

// Example usage
const numbers = [64, 34, 25, 12, 22, 11, 90];
console.log('Original array:', numbers);
console.log('Sorted array:', quickSort(numbers));''',
            
            # General knowledge and conversation
            '''Artificial Intelligence has revolutionized many aspects of modern life. Machine learning algorithms 
can now recognize patterns in data that would be impossible for humans to detect manually. Deep learning 
networks, inspired by the structure of the human brain, have achieved remarkable success in tasks like 
image recognition, natural language processing, and game playing.

The development of large language models has particularly transformed how we interact with AI systems. 
These models can understand context, generate coherent text, and even write code. However, challenges 
remain in ensuring AI systems are safe, reliable, and beneficial for society.''',
            
            '''Climate change represents one of the most significant challenges of our time. Rising global 
temperatures are causing shifts in weather patterns, melting ice caps, and rising sea levels. Scientists 
around the world are working to understand these complex systems and develop solutions to mitigate the 
effects of climate change.

Renewable energy technologies like solar and wind power are becoming increasingly efficient and cost-effective. 
Electric vehicles are gaining popularity as battery technology improves. However, the transition to a 
sustainable future requires coordinated global action and significant changes in how we produce and consume energy.''',
        ]
        
        # Repeat examples to reach desired number of samples
        all_texts = []
        while len(all_texts) < num_samples:
            all_texts.extend(examples)
        
        # Truncate to exact number requested
        all_texts = all_texts[:num_samples]
        
        # Create dataset
        dataset = TextDataset(all_texts, tokenizer, seq_length)
        
        logger.info(f"Created fallback dataset with {len(dataset):,} samples")
        return dataset

# Update create_llm_config to use better tokenizer by default
def create_optimized_llm_config(tokenizer_type: str = "gpt4"):
    """Create optimized LLM configuration with advanced tokenizer."""
    return create_advanced_llm_config(tokenizer_type)

# Configuration creation functions
def create_llm_config(tokenizer_type: str = "gpt4"):
    """Create configuration for LLM task with advanced tokenizer support."""
    
    # Determine vocabulary size based on tokenizer type
    vocab_size_mapping = {
        "gpt4": 100277,      # cl100k_base (actual tiktoken size)
        "gpt3": 50281,       # p50k_base (actual tiktoken size)
        "o200k": 200019,     # o200k_base (actual tiktoken size)
        "codellama": 32000,  # CodeLlama vocab
        "llama2": 32000,     # Llama2 vocab
        "gpt2": 50257        # GPT-2 fallback (r50k_base)
    }
    
    vocab_size = vocab_size_mapping.get(tokenizer_type, 50257)
    
    logger.info(f"🔧 Creating LLM config for {tokenizer_type} tokenizer (vocab: {vocab_size:,})")
    
    return ModelConfig(
        task_type=TaskType.LLM,
        input_dim=768,
        hidden_dim=512,
        output_dim=vocab_size,  # Use tokenizer-specific vocab size
        vocab_size=vocab_size,  # Use tokenizer-specific vocab size
        num_layers=8,
        num_spike_steps=10,
        sequence_length=256,
        liquid_units=256,
        spiking_units=128,
        num_attention_heads=8,
        dropout=0.1,
        spike_threshold=1.0,
        beta=0.95,
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        gradient_clip=1.0,
        mixed_precision=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        liquid_backbone="cfc"
    )

def create_vision_config():
    """Create configuration for vision task."""
    return ModelConfig(
        task_type=TaskType.VISION,
        input_dim=3072,  # 32x32x3 flattened
        hidden_dim=256,
        output_dim=10,   # CIFAR-10 classes
        vocab_size=10,
        num_layers=6,
        num_spike_steps=20,
        sequence_length=1,
        liquid_units=128,
        spiking_units=64,
        num_attention_heads=4,
        dropout_rate=0.2,
        learning_rate=1e-3,
        beta=0.9,
        mixed_precision=True,
        gradient_accumulation_steps=1
    )

def create_robotics_config():
    """Create configuration for robotics task."""
    return ModelConfig(
        task_type=TaskType.ROBOTICS,
        input_dim=100,   # Sensor data dimension
        hidden_dim=256,
        output_dim=6,    # 6DOF control
        vocab_size=1000,
        num_layers=4,
        num_spike_steps=50,
        sequence_length=100,
        liquid_units=128,
        spiking_units=64,
        num_attention_heads=4,
        dropout_rate=0.1,
        learning_rate=5e-4,
        beta=0.95,
        mixed_precision=True,
        gradient_accumulation_steps=2
    )

def create_custom_config(task_type: str, **kwargs) -> ModelConfig:
    """Create custom configuration with overrides."""
    if task_type.lower() == "llm":
        config = create_llm_config()
    elif task_type.lower() == "vision":
        config = create_vision_config()
    elif task_type.lower() == "robotics":
        config = create_robotics_config()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def save_config(config: ModelConfig, filepath: str):
    """Save configuration to file."""
    config_dict = {}
    for field in config.__dataclass_fields__:
        value = getattr(config, field)
        if isinstance(value, Enum):
            config_dict[field] = value.value
        else:
            config_dict[field] = value
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config(filepath: str) -> ModelConfig:
    """Load configuration from file."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Convert task_type back to enum
    if 'task_type' in config_dict:
        config_dict['task_type'] = TaskType(config_dict['task_type'])
    
    return ModelConfig(**config_dict)

def print_config_summary(config: ModelConfig):
    """Print configuration summary."""
    print(f"Configuration Summary:")
    print(f"  Task Type: {config.task_type.value}")
    print(f"  Model Dimensions: {config.input_dim} → {config.hidden_dim} → {config.output_dim}")
    print(f"  Network: {config.num_layers} layers, {config.liquid_units} liquid units, {config.spiking_units} spiking units")
    print(f"  Sequence: {config.sequence_length} tokens, {config.num_spike_steps} spike steps")
    print(f"  Training: LR={config.learning_rate}, β={config.beta}, dropout={config.dropout}")
    if config.task_type == TaskType.LLM:
        print(f"  Vocabulary: {config.vocab_size} tokens")

def get_model_parameter_count(config: ModelConfig) -> dict:
    """Get model parameter count estimate."""
    # Create a temporary model to count parameters
    model = LiquidSpikingNetwork(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "memory_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
    }

def train_llm_model(tokenizer_type: str = "gpt4", num_epochs: int = 10, save_path: str = None):
    """Train LLM model with advanced tokenizer features."""
    print(f"🚀 Starting LLM Training with {tokenizer_type} tokenizer...")
    
    # Create configuration with correct tokenizer
    config = create_llm_config(tokenizer_type)
    
    # Update config with training parameters
    config.num_epochs = num_epochs
    
    # Create model
    model = LiquidSpikingNetwork(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"📊 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"📚 Vocabulary size: {config.vocab_size:,} tokens ({tokenizer_type} tokenizer)")
    print(f"🎯 Output dimension: {config.output_dim:,}")
    
    # Create dataset and dataloader with advanced tokenizer
    dataset, tokenizer = DatasetFactory.create_llm_dataset(
        vocab_size=config.vocab_size,
        seq_length=config.sequence_length,
        num_samples=50000,  # Can be made configurable
        tokenizer_name=tokenizer_type
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"📈 Training set: {train_size:,} samples")
    print(f"📊 Validation set: {val_size:,} samples")
    
    # Create trainer
    trainer = LiquidSpikingTrainer(model, config)
    
    # Train model
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    # Save model with tokenizer info
    if save_path is None:
        save_path = f"llm_final_model_{tokenizer_type}.pt"
    
    trainer.save_checkpoint(save_path)
    
    # Save tokenizer
    tokenizer_save_path = f"./llm_tokenizer_{tokenizer_type}"
    if hasattr(tokenizer, 'save_pretrained'):
        try:
            tokenizer.save_pretrained(tokenizer_save_path)
            print(f"📁 Tokenizer saved: {tokenizer_save_path}")
        except Exception as e:
            print(f"⚠️ Could not save tokenizer: {e}")
    
    print(f"✅ LLM training completed with {tokenizer_type} tokenizer!")
    print(f"📁 Model saved: {save_path}")
    
    return model, trainer
    
    # Save tokenizer
    if hasattr(tokenizer, 'save_pretrained'):
        tokenizer.save_pretrained('./llm_tokenizer')
    
    print("✅ LLM training completed!")
    return model, trainer

def train_vision_model():
    """Train vision model."""
    print("🚀 Starting Vision Training...")
    
    config = create_vision_config()
    model = LiquidSpikingNetwork(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create vision dataset
    dataset = DatasetFactory.create_vision_dataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    trainer = LiquidSpikingTrainer(model, config, device)
    trainer.train(train_loader, val_loader, num_epochs=10)
    
    trainer.save_checkpoint("vision_final_model.pt")
    print("✅ Vision training completed!")

def train_robotics_model():
    """Train robotics model."""
    print("🚀 Starting Robotics Training...")
    
    config = create_robotics_config()
    model = LiquidSpikingNetwork(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create robotics dataset
    dataset = DatasetFactory.create_robotics_dataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    trainer = LiquidSpikingTrainer(model, config, device)
    trainer.train(train_loader, val_loader, num_epochs=15)
    
    trainer.save_checkpoint("robotics_final_model.pt")
    print("✅ Robotics training completed!")

def load_model(model_path: str, task_type: TaskType) -> Tuple[LiquidSpikingNetwork, ModelConfig]:
    """Load model from checkpoint."""
    print(f"📂 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract configuration
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        if isinstance(config_dict, dict):
            # Convert task_type if it's a string
            if 'task_type' in config_dict and isinstance(config_dict['task_type'], str):
                config_dict['task_type'] = TaskType(config_dict['task_type'])
            config = ModelConfig(**config_dict)
        else:
            config = config_dict
    else:
        # Create default config based on task type
        if task_type == TaskType.LLM:
            config = create_llm_config()
        elif task_type == TaskType.VISION:
            config = create_vision_config()
        elif task_type == TaskType.ROBOTICS:
            config = create_robotics_config()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    # Create model
    model = LiquidSpikingNetwork(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint is the state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ Model loaded successfully!")
    
    return model, config

def generate_text(model: LiquidSpikingNetwork, config: ModelConfig, tokenizer, 
                 prompt: str, max_length: int = 50, temperature: float = 0.8, 
                 use_spike_enhancement: bool = True) -> str:
    """
    Generate text using the trained model with advanced spike-to-probability conversion.
    """
    model.eval()
    device = next(model.parameters()).device
    
    try:
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for step in range(max_length):
                # Get model output
                outputs = model(generated_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply spike enhancement if enabled
                if use_spike_enhancement and hasattr(model, 'spike_decoder'):
                    try:
                        # Create dummy spike features
                        dummy_spikes = torch.randn(
                            generated_ids.size(0), 
                            model.spike_decoder.temporal_window,
                            config.spiking_units, 
                            device=device
                        )
                        
                        # Get spike enhancement
                        spike_result = model.spike_decoder(dummy_spikes, return_uncertainty=False)
                        
                        if isinstance(spike_result, tuple):
                            spike_probs = spike_result[0]
                        else:
                            spike_probs = spike_result
                        
                        # Project to vocabulary space if needed
                        if spike_probs.size(-1) != next_token_logits.size(-1):
                            spike_proj = torch.nn.Linear(
                                spike_probs.size(-1), 
                                next_token_logits.size(-1),
                                device=device
                            )
                            spike_probs = spike_proj(spike_probs)
                        
                        # Enhance logits
                        alpha = 0.1
                        enhanced_logits = next_token_logits + alpha * spike_probs
                        next_token_logits = enhanced_logits
                        
                    except Exception as spike_error:
                        print(f"⚠️ Spike enhancement failed, using standard logits: {spike_error}")
                        pass
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Stop if EOS token
                if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                    break
                
                # Progress update
                if step % 10 == 0:
                    current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    print(f"🔄 Step {step}: {current_text[-50:]}")
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Return only the generated part
        if generated_text.startswith(prompt):
            result = generated_text[len(prompt):].strip()
        else:
            result = generated_text.strip()
        
        print(f"✅ Generated {len(result.split())} words using {'spike-enhanced' if use_spike_enhancement else 'standard'} generation")
        return result
            
    except Exception as e:
        print(f"❌ Text generation failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return ""

def evaluate_perplexity(model: LiquidSpikingNetwork, config: ModelConfig, 
                       tokenizer, test_texts: List[str]) -> float:
    """Evaluate model perplexity on test texts."""
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_tokens = 0
    
    try:
        with torch.no_grad():
            for text in test_texts:
                # Tokenize text
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                input_ids = inputs['input_ids'].to(device)
                
                if input_ids.size(1) < 2:
                    continue
                
                # Forward pass
                outputs = model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                               shift_labels.view(-1))
                
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            return perplexity
        else:
            return float('inf')
            
    except Exception as e:
        print(f"❌ Perplexity evaluation failed: {str(e)}")
        return float('inf')

def benchmark_model(model_path: str, task_type: TaskType) -> Dict[str, Any]:
    """Benchmark model performance."""
    model, config = load_model(model_path, task_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    results = {
        "model_path": model_path,
        "task_type": task_type.value,
        "device": str(device),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "memory_mb": sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
    }
    
    # Performance timing
    model.eval()
    with torch.no_grad():
        if task_type == TaskType.LLM:
            dummy_input = torch.randint(0, config.vocab_size, (1, 50), device=device)
        elif task_type == TaskType.VISION:
            dummy_input = torch.randn(1, 3, 32, 32, device=device)
        else:
            dummy_input = torch.randn(1, config.sequence_length, config.input_dim, device=device)
        
        # Warmup
        for _ in range(5):
            _ = model(dummy_input)
        
        # Timing
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        end_time = time.time()
        
        results["inference_time_ms"] = (end_time - start_time) * 10  # ms per inference
        results["throughput_fps"] = 100 / (end_time - start_time)
    
    return results

def export_onnx(model_path: str, task_type: TaskType, output_path: str = None):
    """Export model to ONNX format."""
    model, config = load_model(model_path, task_type)
    
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    # Create dummy input
    if task_type == TaskType.LLM:
        dummy_input = torch.randint(0, config.vocab_size, (1, 50))
    elif task_type == TaskType.VISION:
        dummy_input = torch.randn(1, 3, 32, 32)
    else:
        dummy_input = torch.randn(1, config.sequence_length, config.input_dim)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print(f"✅ Model exported to ONNX: {output_path}")

def inference_example(model_path: str, task_type: TaskType):
    """Run inference example."""
    model, config = load_model(model_path, task_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if task_type == TaskType.LLM:
        # Text generation example
        try:
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            prompt = "The future of artificial intelligence"
            generated = generate_text(model, config, tokenizer, prompt, max_length=30)
            print(f"Generated: {generated}")
        except Exception as e:
            print(f"Text generation failed: {e}")
    
    elif task_type == TaskType.VISION:
        # Vision classification example
        dummy_image = torch.randn(1, 3, 32, 32, device=device)
        output = model(dummy_image)
        predicted_class = output.argmax(dim=-1).item()
        print(f"Predicted class: {predicted_class}")
    
    else:
        # Robotics control example
        dummy_sensors = torch.randn(1, config.sequence_length, config.input_dim, device=device)
        control_output = model(dummy_sensors)
        print(f"Control output shape: {control_output.shape}")

# Tokenizer upgrade integration
try:
    from .tokenizer_upgrade import create_advanced_llm_config
except ImportError:
    try:
        from src.core.tokenizer_upgrade import create_advanced_llm_config
    except ImportError:
        def create_advanced_llm_config(tokenizer_type: str = "gpt4"):
            """Fallback when tokenizer upgrade is not available."""
            return create_llm_config()