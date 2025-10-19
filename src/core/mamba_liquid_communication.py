#!/usr/bin/env python3
"""
Communication Adapters for Mamba-Liquid-Spiking Integration

This module provides the communication mechanisms between:
- Spiking Neural Networks (event-driven, binary spikes)
- Liquid Neural Networks (continuous dynamics, adaptive time constants)
- Mamba SSM (selective state spaces, long-range dependencies)

Key components:
1. State Projection: Convert between different hidden state dimensions
2. Spike Adapters: Convert spike trains ↔ continuous representations
3. Gating: Learn dynamic routing between pathways
4. Cross-Attention: Allow mutual information exchange
5. Bidirectional Exchange: Modulate parameters between systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

import logging
logger = logging.getLogger(__name__)


class StateProjection(nn.Module):
    """
    Projects between different state space dimensions.
    Used for Liquid state ↔ Mamba state communication.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = (nn.LayerNorm(output_dim) if use_layer_norm
                     else nn.Identity())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(x))


class SpikeToMambaAdapter(nn.Module):
    """
    Converts spike trains to continuous Mamba-compatible representations.
    
    Methods:
    1. Rate coding: Average spike rate over time window
    2. Temporal coding: Weighted by spike timing (recent spikes matter more)
    3. Potential-based: Use membrane potential instead of spikes
    """
    
    def __init__(
        self,
        spiking_units: int,
        mamba_dim: int,
        method: str = 'temporal',
        tau: float = 20.0
    ):
        super().__init__()
        self.method = method
        self.tau = tau
        
        # Learnable projection
        self.spike_projection = nn.Linear(spiking_units, mamba_dim)
        
        # Temporal weighting (recent spikes matter more)
        if method == 'temporal':
            self.register_buffer(
                'temporal_kernel',
                self._create_temporal_kernel(tau)
            )
    
    def _create_temporal_kernel(
        self,
        tau: float,
        length: int = 50
    ) -> torch.Tensor:
        """Exponential decay kernel for temporal coding."""
        t = torch.arange(length, dtype=torch.float32)
        kernel = torch.exp(-t / tau)
        return kernel / kernel.sum()
    
    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_train: [batch, time, spiking_units] - Binary spikes
        Returns:
            continuous: [batch, time, mamba_dim] - Continuous repr
        """
        batch_size, time_steps, _ = spike_train.shape
        
        if self.method == 'rate':
            # Simple rate coding: average over time window
            window_size = min(10, time_steps)
            spike_rate = F.avg_pool1d(
                spike_train.transpose(1, 2),
                kernel_size=window_size,
                stride=1,
                padding=window_size // 2
            ).transpose(1, 2)
            
            # Ensure output has same length as input
            if spike_rate.size(1) != time_steps:
                spike_rate = spike_rate[:, :time_steps, :]
            
        elif self.method == 'temporal':
            # Temporal coding: weight by recency
            kernel_size = min(len(self.temporal_kernel), time_steps)
            kernel = self.temporal_kernel[:kernel_size].view(1, 1, -1)
            
            # Convolve spikes with temporal kernel
            spike_rate = F.conv1d(
                spike_train.transpose(1, 2),
                kernel.expand(spike_train.size(2), 1, -1),
                padding=kernel_size // 2,
                groups=spike_train.size(2)
            ).transpose(1, 2)
            
            # Ensure output has same length as input
            if spike_rate.size(1) != time_steps:
                spike_rate = spike_rate[:, :time_steps, :]
            
        else:  # 'potential'
            # Use raw input as continuous representation
            spike_rate = spike_train
        
        # Project to Mamba dimension
        continuous = self.spike_projection(spike_rate)
        
        return continuous


class MambaToSpikeAdapter(nn.Module):
    """
    Converts Mamba continuous outputs back to spike-compatible repr.
    Useful for feedback loops where Mamba output influences spiking.
    """
    
    def __init__(
        self,
        mamba_dim: int,
        spiking_units: int,
        spike_threshold: float = 0.5
    ):
        super().__init__()
        self.threshold = spike_threshold
        
        # Project to spiking dimension
        self.projection = nn.Linear(mamba_dim, spiking_units)
        
        # Learnable threshold (optional)
        self.adaptive_threshold = nn.Parameter(
            torch.ones(1) * spike_threshold
        )
    
    def forward(
        self,
        mamba_output: torch.Tensor,
        generate_spikes: bool = False
    ) -> torch.Tensor:
        """
        Args:
            mamba_output: [batch, time, mamba_dim]
            generate_spikes: If True, return binary spikes; else potential
        Returns:
            spike_representation: [batch, time, spiking_units]
        """
        # Project to spiking dimension
        potential = torch.sigmoid(self.projection(mamba_output))
        
        if generate_spikes:
            # Stochastic spiking based on potential
            spike_prob = potential
            spikes = (torch.rand_like(spike_prob) < spike_prob).float()
            return spikes
        else:
            # Return continuous potential for gradient flow
            return potential


class LiquidMambaGate(nn.Module):
    """
    Adaptive gating mechanism to balance Liquid and Mamba pathways.
    
    Learns when to rely on:
    - Liquid: Fast, adaptive, short-term dynamics
    - Mamba: Long-range dependencies, context
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head gating (different aspects use different pathways)
        self.liquid_query = nn.Linear(hidden_dim, hidden_dim)
        self.mamba_key = nn.Linear(hidden_dim, hidden_dim)
        self.gate_projection = nn.Linear(hidden_dim, num_heads)
        
        # Temperature for gating
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        liquid_output: torch.Tensor,
        mamba_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            liquid_output: [batch, time, hidden_dim]
            mamba_output: [batch, time, hidden_dim]
        Returns:
            fused_output: [batch, time, hidden_dim]
            gate_weights: [batch, time, num_heads] - for interpretability
        """
        batch_size, time_steps, hidden_dim = liquid_output.shape
        
        # Compute gating scores
        q = self.liquid_query(liquid_output)  # What liquid asks for
        k = self.mamba_key(mamba_output)      # What mamba provides
        
        # Attention-like gating
        gate_logits = self.gate_projection(q * k)
        gate_weights = torch.sigmoid(gate_logits / self.temperature)
        
        # Split into heads for fine-grained control
        liquid_heads = liquid_output.view(
            batch_size, time_steps, self.num_heads, self.head_dim
        )
        mamba_heads = mamba_output.view(
            batch_size, time_steps, self.num_heads, self.head_dim
        )
        
        # Apply per-head gating
        gate_expanded = gate_weights.unsqueeze(-1)
        fused_heads = (gate_expanded * liquid_heads +
                       (1 - gate_expanded) * mamba_heads)
        
        # Merge heads
        fused_output = fused_heads.view(
            batch_size, time_steps, hidden_dim
        )
        
        return fused_output, gate_weights


class BidirectionalStateExchange(nn.Module):
    """
    Enables bidirectional information flow between Liquid and Mamba.
    
    - Liquid state → influences Mamba's selective mechanism (B, C)
    - Mamba state → modulates Liquid's time constants (τ)
    """
    
    def __init__(
        self,
        liquid_units: int,
        mamba_d_state: int,
        mamba_d_model: int
    ):
        super().__init__()
        self.liquid_units = liquid_units
        self.mamba_d_state = mamba_d_state
        
        # Liquid → Mamba: Modulate Mamba's selectivity
        self.liquid_to_mamba_B = nn.Linear(liquid_units, mamba_d_state)
        self.liquid_to_mamba_C = nn.Linear(liquid_units, mamba_d_state)
        
        # Mamba → Liquid: Modulate Liquid's time constants
        self.mamba_to_liquid_tau = nn.Linear(mamba_d_model, liquid_units)
        
    def modulate_mamba(
        self,
        liquid_state: torch.Tensor,
        mamba_B: torch.Tensor,
        mamba_C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Liquid state influences Mamba's selective parameters B and C.
        
        Args:
            liquid_state: [batch, liquid_units]
            mamba_B, mamba_C: Mamba's input/output matrices
        Returns:
            modulated_B, modulated_C
        """
        # Adaptive modulation based on liquid dynamics
        B_modulation = self.liquid_to_mamba_B(liquid_state)
        C_modulation = self.liquid_to_mamba_C(liquid_state)
        
        # Add modulation (residual preserves original behavior)
        modulated_B = mamba_B + B_modulation.unsqueeze(-1)
        modulated_C = mamba_C + C_modulation.unsqueeze(1)
        
        return modulated_B, modulated_C
    
    def modulate_liquid(
        self,
        mamba_output: torch.Tensor,
        base_tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Mamba output influences Liquid's time constants.
        
        Args:
            mamba_output: [batch, time, mamba_d_model]
            base_tau: [liquid_units] - base time constants
        Returns:
            adaptive_tau: [batch, time, liquid_units]
        """
        # Mamba suggests adaptive time constants
        tau_modulation = torch.sigmoid(
            self.mamba_to_liquid_tau(mamba_output)
        )
        
        # Scale base time constants
        adaptive_tau = (base_tau.unsqueeze(0).unsqueeze(0) *
                        (0.5 + tau_modulation))
        
        return adaptive_tau


class CrossModalAttention(nn.Module):
    """
    Cross-attention between Liquid and Mamba representations.
    Allows each pathway to query information from the other.
    """
    
    def __init__(
        self,
        liquid_dim: int,
        mamba_dim: int,
        num_heads: int = 8
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = min(liquid_dim, mamba_dim) // num_heads
        
        # Liquid queries Mamba
        self.liquid_q = nn.Linear(
            liquid_dim, num_heads * self.head_dim
        )
        self.mamba_kv = nn.Linear(
            mamba_dim, 2 * num_heads * self.head_dim
        )
        
        # Mamba queries Liquid
        self.mamba_q = nn.Linear(
            mamba_dim, num_heads * self.head_dim
        )
        self.liquid_kv = nn.Linear(
            liquid_dim, 2 * num_heads * self.head_dim
        )
        
        # Output projections
        self.liquid_out = nn.Linear(
            num_heads * self.head_dim, liquid_dim
        )
        self.mamba_out = nn.Linear(
            num_heads * self.head_dim, mamba_dim
        )
        
    def forward(
        self,
        liquid_repr: torch.Tensor,
        mamba_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            liquid_repr: [batch, time, liquid_dim]
            mamba_repr: [batch, time, mamba_dim]
        Returns:
            enhanced_liquid: [batch, time, liquid_dim]
            enhanced_mamba: [batch, time, mamba_dim]
        """
        batch_size, time_steps, _ = liquid_repr.shape
        
        # Liquid attends to Mamba
        q_l = self.liquid_q(liquid_repr).view(
            batch_size, time_steps, self.num_heads, self.head_dim
        )
        kv_m = self.mamba_kv(mamba_repr).view(
            batch_size, time_steps, self.num_heads, 2 * self.head_dim
        )
        k_m, v_m = kv_m.chunk(2, dim=-1)
        
        attn_l = self._compute_attention(q_l, k_m, v_m)
        enhanced_liquid = liquid_repr + self.liquid_out(
            attn_l.view(batch_size, time_steps, -1)
        )
        
        # Mamba attends to Liquid
        q_m = self.mamba_q(mamba_repr).view(
            batch_size, time_steps, self.num_heads, self.head_dim
        )
        kv_l = self.liquid_kv(liquid_repr).view(
            batch_size, time_steps, self.num_heads, 2 * self.head_dim
        )
        k_l, v_l = kv_l.chunk(2, dim=-1)
        
        attn_m = self._compute_attention(q_m, k_l, v_l)
        enhanced_mamba = mamba_repr + self.mamba_out(
            attn_m.view(batch_size, time_steps, -1)
        )
        
        return enhanced_liquid, enhanced_mamba
    
    def _compute_attention(self, q, k, v):
        """Standard scaled dot-product attention."""
        scores = (torch.matmul(q, k.transpose(-2, -1)) /
                  math.sqrt(self.head_dim))
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)


def test_communication_adapters():
    """Test communication adapters."""
    print("Testing communication adapters...")
    
    batch_size = 2
    seq_len = 64
    spiking_units = 128
    liquid_units = 256
    mamba_dim = 512
    mamba_d_state = 16
    
    # Test SpikeToMambaAdapter
    print("\n1. Testing SpikeToMambaAdapter...")
    spike_adapter = SpikeToMambaAdapter(spiking_units, mamba_dim)
    spike_train = torch.rand(batch_size, seq_len, spiking_units) > 0.5
    spike_train = spike_train.float()
    continuous = spike_adapter(spike_train)
    print(f"Spike train: {spike_train.shape} -> Continuous: "
          f"{continuous.shape}")
    
    # Test LiquidMambaGate
    print("\n2. Testing LiquidMambaGate...")
    gate = LiquidMambaGate(mamba_dim, num_heads=4)
    liquid_out = torch.randn(batch_size, seq_len, mamba_dim)
    mamba_out = torch.randn(batch_size, seq_len, mamba_dim)
    fused, weights = gate(liquid_out, mamba_out)
    print(f"Fused output: {fused.shape}, Gate weights: {weights.shape}")
    
    # Test CrossModalAttention
    print("\n3. Testing CrossModalAttention...")
    cross_attn = CrossModalAttention(liquid_units, mamba_dim, num_heads=8)
    liquid_repr = torch.randn(batch_size, seq_len, liquid_units)
    mamba_repr = torch.randn(batch_size, seq_len, mamba_dim)
    enh_l, enh_m = cross_attn(liquid_repr, mamba_repr)
    print(f"Enhanced liquid: {enh_l.shape}, Enhanced mamba: "
          f"{enh_m.shape}")
    
    print("\n✅ All communication adapters tested successfully!")


if __name__ == "__main__":
    test_communication_adapters()
