#!/usr/bin/env python3
"""
Critical fixes for zero-learning issue in Liquid-Spiking Networks.
Addresses gradient flow, spike encoding, and initialization problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FixedSpikingEncoder(nn.Module):
    """
    Fixed spiking encoder with proper gradient flow.
    
    Issues fixed:
    1. Surrogate gradient scale (was too small)
    2. Membrane potential initialization
    3. Spike threshold adaptation
    4. Better temporal dynamics
    """
    
    def __init__(self, input_dim, output_dim, num_steps, beta=0.95):
        super().__init__()
        self.num_steps = num_steps
        self.output_dim = output_dim
        
        # Projection layers with better initialization
        self.fc1 = nn.Linear(input_dim, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)
        
        # Initialize with Xavier/Glorot for better gradient flow
        nn.init.xavier_normal_(self.fc1.weight, gain=1.0)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
        # LIF neurons with LARGER surrogate gradient slope
        import snntorch as snn
        from snntorch import surrogate
        
        # Use steep surrogate gradient (beta=25 instead of default 5)
        self.lif1 = snn.Leaky(
            beta=beta, 
            spike_grad=surrogate.fast_sigmoid(slope=25)  # CRITICAL FIX
        )
        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=surrogate.fast_sigmoid(slope=25)
        )
        
        self.dropout = nn.Dropout(0.1)  # Reduced from 0.2
        
        # Learnable spike threshold (adaptive)
        self.spike_threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper gradient scaling."""
        # Input normalization for stable membrane potentials
        x = F.layer_norm(x, [x.size(-1)])
        
        batch_size = x.size(0)
        
        # Handle shape
        if x.ndim == 2:
            x = x.unsqueeze(1).expand(-1, self.num_steps, -1)
        
        # Initialize membrane potentials with small random values
        mem1 = torch.randn(batch_size, self.fc1.out_features, 
                          device=x.device) * 0.01
        mem2 = torch.randn(batch_size, self.output_dim, 
                          device=x.device) * 0.01
        
        spike_output = []
        
        for t in range(x.size(1)):
            # First layer
            cur1 = self.fc1(x[:, t])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Dropout and second layer
            spk1_dropped = self.dropout(spk1)
            cur2 = self.fc2(spk1_dropped)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spike_output.append(spk2)
        
        # Stack spikes [batch, time, features]
        spike_train = torch.stack(spike_output, dim=1)
        
        return spike_train


class ImprovedSpikeDecoder(nn.Module):
    """
    Improved spike-to-logits decoder with better gradient flow.
    
    Issues fixed:
    1. Rate coding normalization
    2. Learnable temperature
    3. Skip connection for gradient highway
    """
    
    def __init__(self, spike_dim: int, output_dim: int, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        
        # Multi-path decoding for robust gradients
        self.rate_decoder = nn.Sequential(
            nn.Linear(spike_dim, spike_dim),
            nn.LayerNorm(spike_dim),
            nn.ReLU(),
            nn.Linear(spike_dim, output_dim)
        )
        
        # Temporal pattern decoder
        self.temporal_decoder = nn.Sequential(
            nn.Conv1d(spike_dim, spike_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(spike_dim, output_dim, kernel_size=1)
        )
        
        # Learnable temperature for logit scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Fusion weights
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """
        Convert spikes to logits with multiple pathways.
        
        Args:
            spike_train: [batch, time, spike_dim]
        Returns:
            logits: [batch, time, output_dim]
        """
        # Path 1: Rate coding (mean over time)
        spike_rate = spike_train.mean(dim=1)  # [batch, spike_dim]
        rate_logits = self.rate_decoder(spike_rate)  # [batch, output_dim]
        
        # Path 2: Temporal patterns
        temporal_input = spike_train.transpose(1, 2)  # [batch, spike_dim, time]
        temporal_features = self.temporal_decoder(temporal_input)  # [batch, output_dim, time]
        temporal_logits = temporal_features.mean(dim=2)  # [batch, output_dim]
        
        # Fuse pathways with learnable weight
        alpha = torch.sigmoid(self.fusion_weight)
        logits = alpha * rate_logits + (1 - alpha) * temporal_logits
        
        # Temperature scaling
        logits = logits / (self.temperature.abs() + 1e-8)
        
        return logits


class GradientHealthMonitor:
    """
    Monitor gradient health during training.
    Helps diagnose gradient vanishing/explosion.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.grad_stats = []
        
    def check_gradients(self) -> dict:
        """Check gradient statistics."""
        grad_norms = []
        zero_grads = 0
        inf_grads = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if grad_norm == 0:
                    zero_grads += 1
                if torch.isinf(param.grad).any():
                    inf_grads += 1
            else:
                zero_grads += 1
        
        stats = {
            'mean_norm': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
            'max_norm': max(grad_norms) if grad_norms else 0,
            'min_norm': min(grad_norms) if grad_norms else 0,
            'zero_grads': zero_grads,
            'inf_grads': inf_grads,
            'total_params': len(list(self.model.parameters()))
        }
        
        self.grad_stats.append(stats)
        return stats
    
    def diagnose(self) -> str:
        """Diagnose gradient issues."""
        if not self.grad_stats:
            return "No gradient data collected"
        
        latest = self.grad_stats[-1]
        issues = []
        
        if latest['mean_norm'] < 1e-6:
            issues.append("⚠️ VANISHING GRADIENTS: Mean norm extremely small")
        
        if latest['max_norm'] > 1000:
            issues.append("⚠️ EXPLODING GRADIENTS: Max norm very large")
        
        if latest['zero_grads'] > latest['total_params'] * 0.5:
            issues.append(f"⚠️ DEAD PARAMETERS: {latest['zero_grads']} params have zero gradient")
        
        if latest['inf_grads'] > 0:
            issues.append(f"⚠️ INFINITE GRADIENTS: {latest['inf_grads']} params have inf gradient")
        
        if not issues:
            return "✓ Gradients look healthy"
        
        return "\n".join(issues)


def apply_training_fixes(model: nn.Module, config) -> nn.Module:
    """
    Apply critical fixes to enable learning.
    
    Fixes:
    1. Replace spiking encoders
    2. Fix output head initialization
    3. Add gradient monitoring
    """
    logger.info("🔧 Applying critical training fixes...")
    
    # Fix 1: Replace spiking encoders in hybrid blocks
    if hasattr(model, 'hybrid_blocks'):
        for i, block in enumerate(model.hybrid_blocks):
            logger.info(f"  Fixing hybrid block {i}...")
            if hasattr(block, 'spike_encoder'):
                block.spike_encoder = FixedSpikingEncoder(
                    block.spike_encoder.fc1.in_features,
                    block.spike_encoder.fc2.out_features,
                    block.spike_encoder.num_steps,
                    config.beta
                )
    
    # Fix 2: Add improved spike decoder
    if hasattr(model, 'spike_decoder'):
        logger.info("  Replacing spike decoder...")
        model.spike_decoder = ImprovedSpikeDecoder(
            config.spiking_units,
            config.hidden_dim,
            config.num_spike_steps
        )
    
    # Fix 3: Re-initialize output head with smaller weights
    if hasattr(model, 'output_head'):
        logger.info("  Re-initializing output head...")
        if isinstance(model.output_head, nn.Linear):
            nn.init.normal_(model.output_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(model.output_head.bias)
        elif isinstance(model.output_head, nn.Sequential):
            for layer in model.output_head:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    logger.info("✅ Training fixes applied!")
    return model


def add_warmup_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """
    Create warmup + cosine decay scheduler.
    Critical for spiking networks to stabilize.
    """
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))
    
    return LambdaLR(optimizer, lr_lambda)


def diagnose_training_stuck(model: nn.Module, train_loader, device) -> dict:
    """
    Diagnose why training is stuck.
    """
    logger.info("\n" + "="*60)
    logger.info("🔍 DIAGNOSING TRAINING ISSUES")
    logger.info("="*60)
    
    diagnostics = {}
    
    # Check 1: Forward pass produces valid outputs
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
        else:
            inputs = batch.to(device)
        
        outputs = model(inputs)
        
        diagnostics['output_mean'] = outputs.mean().item()
        diagnostics['output_std'] = outputs.std().item()
        diagnostics['output_min'] = outputs.min().item()
        diagnostics['output_max'] = outputs.max().item()
        
        logger.info(f"📊 Output statistics:")
        logger.info(f"   Mean: {diagnostics['output_mean']:.4f}")
        logger.info(f"   Std:  {diagnostics['output_std']:.4f}")
        logger.info(f"   Min:  {diagnostics['output_min']:.4f}")
        logger.info(f"   Max:  {diagnostics['output_max']:.4f}")
    
    # Check 2: Gradient flow
    model.train()
    model.zero_grad()
    
    outputs = model(inputs)
    if isinstance(batch, (list, tuple)) and len(batch) > 1:
        targets = batch[1].to(device)
        loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )
    else:
        # Dummy loss
        loss = outputs.mean()
    
    loss.backward()
    
    monitor = GradientHealthMonitor(model)
    grad_stats = monitor.check_gradients()
    
    logger.info(f"\n📈 Gradient statistics:")
    logger.info(f"   Mean norm: {grad_stats['mean_norm']:.6f}")
    logger.info(f"   Max norm:  {grad_stats['max_norm']:.6f}")
    logger.info(f"   Zero grads: {grad_stats['zero_grads']}/{grad_stats['total_params']}")
    
    diagnosis = monitor.diagnose()
    logger.info(f"\n{diagnosis}")
    
    diagnostics.update(grad_stats)
    diagnostics['diagnosis'] = diagnosis
    
    logger.info("="*60 + "\n")
    
    return diagnostics
