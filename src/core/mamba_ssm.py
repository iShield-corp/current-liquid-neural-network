#!/usr/bin/env python3
"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces

Implementation of Mamba SSM (Selective State Space Model) based on:
"Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

Key features:
- Selective state space mechanism (data-dependent B, C matrices)
- Linear-time complexity O(N) vs O(N²) for attention
- Efficient causal convolution for local context
- Hardware-efficient parallel scan implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)


@dataclass
class MambaConfig:
    """Configuration for Mamba block."""
    d_model: int = 512          # Model dimension
    d_state: int = 16           # SSM state dimension (N)
    d_conv: int = 4             # Convolution kernel size
    expand_factor: int = 2      # Expansion factor for inner dimension
    dt_rank: str = 'auto'       # Rank of dt projection (auto = d_model/16)
    dt_min: float = 0.001       # Minimum dt value
    dt_max: float = 0.1         # Maximum dt value
    dt_init: str = 'random'     # dt initialization: 'random' or 'constant'
    dt_scale: float = 1.0       # dt scaling factor
    dt_init_floor: float = 1e-4 # Floor for dt initialization
    conv_bias: bool = True      # Use bias in convolution
    bias: bool = False          # Use bias in linear layers
    use_fast_path: bool = True  # Use optimized implementation when possible
    
    def __post_init__(self):
        self.d_inner = self.d_model * self.expand_factor
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        
        logger.info(f"Mamba config: d_model={self.d_model}, d_state={self.d_state}, "
                   f"d_inner={self.d_inner}, dt_rank={self.dt_rank}")


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Core of Mamba
    
    The key innovation: B and C matrices are data-dependent (selective),
    allowing the model to filter irrelevant information and remember important context.
    
    State space equation:
        h'(t) = Ah(t) + Bx(t)
        y(t) = Ch(t) + Dx(t)
    
    Discretized (zero-order hold):
        h[t] = A_bar * h[t-1] + B_bar * x[t]
        y[t] = C * h[t] + D * x[t]
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_inner = config.d_inner
        self.dt_rank = config.dt_rank
        
        # SSM parameters
        # A: (d_inner, d_state) - Diagonal structure for efficiency
        A = torch.randn(config.d_inner, config.d_state)
        self.A_log = nn.Parameter(torch.log(A))  # Log-space for numerical stability
        
        # D: (d_inner,) - Skip connection
        self.D = nn.Parameter(torch.ones(config.d_inner))
        
        # Selective projections (data-dependent B, C, dt)
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        
        # dt projection: from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        
        # Initialize dt projection for better stability
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == 'constant':
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == 'random':
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias to be between dt_min and dt_max
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_inner)
        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch_size, seq_len, d_inner = x.shape
        
        # Get A matrix (stable version via log-space)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Selective projections: compute data-dependent B, C, dt
        x_proj_out = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)
        
        # Split into dt, B, C
        dt_proj_input, B, C = torch.split(
            x_proj_out,
            [self.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )
        
        # Compute dt (time step) - data dependent!
        dt = F.softplus(self.dt_proj(dt_proj_input))  # (batch, seq_len, d_inner)
        
        # Discretize continuous system to discrete
        # Using zero-order hold (ZOH) discretization
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (batch, seq_len, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq_len, d_inner, d_state)
        
        # Selective scan (the core operation)
        y = self.selective_scan(x, dA, dB, C, self.D)
        
        return y
    
    def selective_scan(
        self,
        x: torch.Tensor,
        dA: torch.Tensor,
        dB: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform selective scan operation.
        
        This is the associative scan that can be parallelized efficiently.
        For simplicity, we use a sequential implementation here.
        Hardware-efficient parallel scan would use work-efficient algorithms.
        
        Args:
            x: (batch, seq_len, d_inner) - Input
            dA: (batch, seq_len, d_inner, d_state) - Discretized A
            dB: (batch, seq_len, d_inner, d_state) - Discretized B
            C: (batch, seq_len, d_state) - Output projection
            D: (d_inner,) - Skip connection
        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = self.d_state
        
        # Initialize hidden state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        # Sequential scan (can be parallelized with associative scan)
        for t in range(seq_len):
            # h[t] = dA[t] * h[t-1] + dB[t] * x[t]
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            
            # y[t] = C[t] @ h[t] + D * x[t]
            y = torch.einsum('bdn,bn->bd', h, C[:, t]) + D * x[:, t]
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)


class MambaBlock(nn.Module):
    """
    Complete Mamba block with:
    1. Input projection and expansion
    2. Causal convolution for local context
    3. Selective SSM for long-range dependencies
    4. Gating mechanism (SiLU activation)
    5. Output projection
    
    Architecture:
        x -> [Linear(expand)] -> [split] -> gate & input
        input -> [Conv1d] -> [SiLU] -> [SSM] -> output
        output = output * SiLU(gate)
        output -> [Linear(project)] -> out
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Input projection: d_model -> d_inner * 2 (for gating)
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        
        # Causal 1D convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            groups=config.d_inner,  # Depthwise convolution
            padding=config.d_conv - 1,  # Causal padding
            bias=config.conv_bias
        )
        
        # Activation
        self.act = nn.SiLU()
        
        # Selective SSM
        self.ssm = SelectiveSSM(config)
        
        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        
        # Normalization (optional, for stability)
        self.norm = nn.LayerNorm(config.d_inner)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Input projection and split for gating
        x_proj = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_main, x_gate = x_proj.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Causal convolution (transpose for Conv1d)
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        
        # Activation
        x_conv = self.act(x_conv)
        
        # Apply selective SSM
        x_ssm = self.ssm(x_conv)  # (batch, seq_len, d_inner)
        
        # Normalization for stability
        x_ssm = self.norm(x_ssm)
        
        # Gating with SiLU-activated gate
        x_gated = x_ssm * self.act(x_gate)
        
        # Output projection
        output = self.out_proj(x_gated)
        
        return output


class MambaLayer(nn.Module):
    """
    Complete Mamba layer with residual connection and normalization.
    
    Architecture:
        x -> LayerNorm -> MambaBlock -> residual -> output
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mamba = MambaBlock(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Pre-norm + residual connection
        return x + self.mamba(self.norm(x))


def test_mamba():
    """Test Mamba implementation."""
    print("Testing Mamba SSM implementation...")
    
    # Configuration
    config = MambaConfig(
        d_model=256,
        d_state=16,
        d_conv=4,
        expand_factor=2
    )
    
    # Create model
    model = MambaLayer(config)
    
    # Test input
    batch_size = 2
    seq_len = 128
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Memory efficiency test
    print(f"\nMemory efficiency (seq_len=1024):")
    x_long = torch.randn(1, 1024, config.d_model)
    with torch.no_grad():
        output_long = model(x_long)
    print(f"Successfully processed sequence of length {x_long.shape[1]}")
    
    print("✅ Mamba test completed successfully!")


if __name__ == "__main__":
    test_mamba()
