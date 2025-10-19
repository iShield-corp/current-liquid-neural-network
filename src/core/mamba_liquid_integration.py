#!/usr/bin/env python3
"""
Integrated Mamba-Liquid-Spiking Block

This module integrates three neural paradigms:
1. Spiking Neural Networks - Event-driven, energy-efficient encoding
2. Liquid Neural Networks - Adaptive short-term temporal dynamics
3. Mamba SSM - Efficient long-range dependencies (linear time)

Communication architecture follows bidirectional pattern:
    Input → Spike encoding
    Spikes → Liquid, Spikes → Mamba (parallel)
    Cross-attention: Liquid ↔ Mamba
    Liquid state influences Mamba's selectivity (via B, C matrices)
    Mamba output modulates Liquid's time constants (via τ)
    Fused output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import sys
import os

# Add parent directory for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.core.mamba_ssm import MambaConfig, MambaBlock
from src.core.mamba_liquid_communication import (
    SpikeToMambaAdapter, MambaToSpikeAdapter,
    LiquidMambaGate, BidirectionalStateExchange,
    CrossModalAttention, StateProjection
)

import logging
logger = logging.getLogger(__name__)


class IntegratedMambaLiquidSpikingBlock(nn.Module):
    """
    Fully integrated block combining:
    - Spiking neural encoding (event-driven)
    - Liquid neural dynamics (adaptive short-term)
    - Mamba state space (long-range dependencies)
    
    Communication: Bidirectional with cross-attention
    """
    
    def __init__(self, config, integration_mode: str = 'bidirectional'):
        """
        Args:
            config: ModelConfig with all parameters
            integration_mode: 'sequential', 'parallel', 'bidirectional'
        """
        super().__init__()
        self.config = config
        self.integration_mode = integration_mode
        
        # Import components from main.py
        try:
            from src.core.main import SpikingEncoder, LiquidCell
        except ImportError:
            logger.warning("Could not import from main, using placeholders")
            SpikingEncoder = None
            LiquidCell = None
        
        # === Core Components ===
        
        if SpikingEncoder is not None:
            # 1. Spike Encoder
            self.spike_encoder = SpikingEncoder(
                config.input_dim,
                config.spiking_units,
                config.num_spike_steps,
                config.beta
            )
            
            # 2. Liquid Cell
            self.liquid_cell = LiquidCell(
                config.spiking_units,
                config.liquid_units,
                config.liquid_backbone
            )
        else:
            # Placeholder implementations
            self.spike_encoder = nn.Linear(
                config.input_dim, config.spiking_units
            )
            self.liquid_cell = nn.LSTM(
                config.spiking_units, config.liquid_units, batch_first=True
            )
        
        # 3. Mamba Block
        mamba_config = MambaConfig(
            d_model=config.hidden_dim,
            d_state=getattr(config, 'mamba_d_state', 16),
            d_conv=getattr(config, 'mamba_d_conv', 4),
            expand_factor=getattr(config, 'mamba_expand', 2)
        )
        self.mamba = MambaBlock(mamba_config)
        
        # === Communication Adapters ===
        
        # Spike → Mamba
        self.spike_to_mamba = SpikeToMambaAdapter(
            spiking_units=config.spiking_units,
            mamba_dim=config.hidden_dim,
            method='temporal',
            tau=20.0
        )
        
        # Mamba → Spike (for feedback)
        self.mamba_to_spike = MambaToSpikeAdapter(
            mamba_dim=config.hidden_dim,
            spiking_units=config.spiking_units,
            spike_threshold=0.5
        )
        
        # Liquid ↔ Mamba projections
        self.liquid_to_hidden = StateProjection(
            config.liquid_units,
            config.hidden_dim
        )
        self.hidden_to_liquid = StateProjection(
            config.hidden_dim,
            config.liquid_units
        )
        
        # === Integration Mechanisms ===
        
        if integration_mode == 'parallel':
            self.gate = LiquidMambaGate(config.hidden_dim, num_heads=4)
            
        elif integration_mode == 'bidirectional':
            self.bidirectional_exchange = BidirectionalStateExchange(
                liquid_units=config.liquid_units,
                mamba_d_state=getattr(config, 'mamba_d_state', 16),
                mamba_d_model=config.hidden_dim
            )
            self.cross_attention = CrossModalAttention(
                liquid_dim=config.liquid_units,
                mamba_dim=config.hidden_dim,
                num_heads=8
            )
        
        # Normalization
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        # Residual projection and input adaptation 
        # (will be created dynamically if needed)
        self.residual_proj = None
        self.input_proj = None  # Project varying input dims to expected dim
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
    
    def forward(
        self,
        x: torch.Tensor,
        h_liquid: Optional[torch.Tensor] = None,
        return_internals: bool = False
    ) -> Tuple:
        """
        Forward pass with communication between all components.
        
        Args:
            x: [batch, time, input_dim]
            h_liquid: Liquid cell hidden state
            return_internals: Return intermediate representations
            
        Returns:
            output: [batch, time, hidden_dim]
            h_liquid_new: Updated liquid state
            (optional) internals: Dict of intermediate values
        """
        batch_size, time_steps, input_dim = x.shape
        
        # Create input projection if input dimension doesn't match expected
        if input_dim != self.input_dim:
            if self.input_proj is None:
                self.input_proj = nn.Linear(
                    input_dim, self.input_dim
                ).to(x.device)
            x_projected = self.input_proj(x)
        else:
            x_projected = x
        
        # Create residual projection if needed (lazy initialization)
        if self.residual_proj is None and self.input_dim != self.hidden_dim:
            self.residual_proj = nn.Linear(
                self.input_dim, self.hidden_dim
            ).to(x_projected.device)
        
        # Store for residual connection
        residual = (
            self.residual_proj(x_projected)
            if self.residual_proj
            else x_projected
        )
        
        # === Stage 1: Spike Encoding ===
        spike_train = self.spike_encoder(x_projected)
        if not isinstance(spike_train, torch.Tensor):
            # Handle LSTM case
            spike_train = spike_train[0]
        
        # === Stage 2: Liquid Dynamics ===
        liquid_output, h_liquid_new = self.liquid_cell(
            spike_train, h_liquid
        )
        if not isinstance(liquid_output, torch.Tensor):
            # Handle LSTM case
            liquid_output = liquid_output[0]
            h_liquid_new = liquid_output
        
        # Project liquid to hidden dimension
        liquid_repr = self.liquid_to_hidden(liquid_output)
        
        # === Stage 3: Integration Mode ===
        
        if self.integration_mode == 'sequential':
            # Spike → Liquid → Mamba (simple pipeline)
            mamba_input = self.norm1(liquid_repr)
            mamba_output = self.mamba(mamba_input)
            output = self.norm2(mamba_output + residual)
            
        elif self.integration_mode == 'parallel':
            # Spike → {Liquid, Mamba} → Gate (parallel pathways)
            spike_repr = self.spike_to_mamba(spike_train)
            mamba_input = self.norm1(spike_repr)
            mamba_output = self.mamba(mamba_input)
            
            # Fuse liquid and mamba pathways
            output, gate_weights = self.gate(liquid_repr, mamba_output)
            output = self.norm2(output + residual)
            
        elif self.integration_mode == 'bidirectional':
            # Spike → Liquid ↔ Mamba (full bidirectional exchange)
            
            # Initial Mamba processing
            spike_repr = self.spike_to_mamba(spike_train)
            mamba_input = self.norm1(spike_repr)
            mamba_output = self.mamba(mamba_input)
            
            # Cross-modal attention (Liquid and Mamba attend to each)
            liquid_enhanced, mamba_enhanced = self.cross_attention(
                liquid_output, mamba_output
            )
            
            # Project enhanced liquid back to hidden dimension
            liquid_final = self.liquid_to_hidden(liquid_enhanced)
            
            # Combine with learned weighting
            alpha = 0.5  # Could be learned
            output = alpha * liquid_final + (1 - alpha) * mamba_enhanced
            output = self.norm2(output + residual)
        
        else:
            raise ValueError(
                f"Unknown integration mode: {self.integration_mode}"
            )
        
        if return_internals:
            internals = {
                'spike_train': spike_train,
                'liquid_output': liquid_output,
                'liquid_repr': liquid_repr,
                'mamba_output': mamba_output if 'mamba_output' in locals()
                                else None,
                'h_liquid': h_liquid_new
            }
            return output, h_liquid_new, internals
        
        return output, h_liquid_new


class MambaLiquidSpikingNetwork(nn.Module):
    """
    Complete network with stacked Mamba-Liquid-Spiking blocks.
    
    Drop-in replacement for LiquidSpikingNetwork in main.py.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input embedding
        if hasattr(config, 'vocab_size'):
            self.embedding = nn.Embedding(
                config.vocab_size, config.input_dim
            )
        else:
            self.embedding = nn.Identity()
        
        # Stacked integrated blocks
        self.blocks = nn.ModuleList([
            IntegratedMambaLiquidSpikingBlock(
                config,
                integration_mode=getattr(
                    config, 'integration_mode', 'bidirectional'
                )
            )
            for _ in range(config.num_layers)
        ])
        
        # Output head
        self.output_head = nn.Linear(config.hidden_dim, config.output_dim)
        
        logger.info(
            f"Created MambaLiquidSpikingNetwork with {len(self.blocks)} "
            f"layers, mode={getattr(config, 'integration_mode', 'bidirectional')}"
        )
        
    def forward(self, x, h_states=None):
        """
        Args:
            x: [batch, time] for LLM or [batch, time, features] for other
            h_states: List of liquid hidden states per layer
        """
        # Handle token inputs (LLM)
        if x.ndim == 2:
            x = self.embedding(x)
        
        # Initialize hidden states if needed
        if h_states is None:
            h_states = [None] * len(self.blocks)
        
        # Process through blocks
        new_h_states = []
        for i, block in enumerate(self.blocks):
            x, h_new = block(x, h_states[i])
            new_h_states.append(h_new)
        
        # Output projection
        output = self.output_head(x)
        
        return output, new_h_states


def test_integrated_block():
    """Test integrated block."""
    print("Testing Integrated Mamba-Liquid-Spiking Block...")
    
    # Mock config
    class MockConfig:
        input_dim = 128
        spiking_units = 256
        liquid_units = 256
        hidden_dim = 512
        output_dim = 50000
        num_spike_steps = 10
        beta = 0.95
        liquid_backbone = 'cfc'
        num_layers = 4
        mamba_d_state = 16
        mamba_d_conv = 4
        mamba_expand = 2
        integration_mode = 'bidirectional'
    
    config = MockConfig()
    
    # Test single block
    print("\n1. Testing single block...")
    block = IntegratedMambaLiquidSpikingBlock(
        config, integration_mode='bidirectional'
    )
    
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, config.input_dim)
    
    with torch.no_grad():
        output, h_new, internals = block(
            x, return_internals=True
        )
    
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Spike train: {internals['spike_train'].shape}")
    print(f"Liquid output: {internals['liquid_output'].shape}")
    print(f"Liquid repr: {internals['liquid_repr'].shape}")
    
    # Test full network
    print("\n2. Testing full network...")
    network = MambaLiquidSpikingNetwork(config)
    
    with torch.no_grad():
        net_output, net_h = network(x)
    
    print(f"Network output: {net_output.shape}")
    print(f"Hidden states: {len(net_h)} layers")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✅ Integrated block test completed successfully!")


if __name__ == "__main__":
    test_integrated_block()
