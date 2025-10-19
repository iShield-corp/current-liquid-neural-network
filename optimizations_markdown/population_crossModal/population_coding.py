"""
Population Coding for Spiking Neural Networks
Implements robust encoding using multiple neurons with different thresholds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PopulationCoding(nn.Module):
    """
    Population coding where multiple neurons encode the same value with different thresholds.
    This provides robustness through redundancy and enables rate-based and temporal coding.
    
    Based on neuroscience research showing that populations of neurons with varying
    response properties provide robust and precise information encoding.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        population_size: int = 5,
        num_steps: int = 32,
        beta: float = 0.95,
        threshold_distribution: str = 'uniform',
        encoding_scheme: str = 'hybrid'
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            population_size: Number of neurons per output dimension
            num_steps: Number of time steps for spike encoding
            beta: Membrane potential decay factor
            threshold_distribution: How to distribute thresholds ('uniform', 'gaussian', 'log')
            encoding_scheme: 'rate', 'temporal', or 'hybrid'
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.population_size = population_size
        self.num_steps = num_steps
        self.beta = beta
        self.encoding_scheme = encoding_scheme
        
        # Total number of spiking neurons
        self.total_neurons = output_dim * population_size
        
        # Input projection to population
        self.input_projection = nn.Linear(input_dim, self.total_neurons)
        
        # Generate threshold distribution for population
        self.thresholds = self._generate_thresholds(threshold_distribution)
        
        # Create LIF neurons with different thresholds
        self.population_neurons = nn.ModuleList([
            snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=surrogate.fast_sigmoid(slope=25),
                init_hidden=True,
                reset_mechanism='subtract'
            )
            for threshold in self.thresholds
        ])
        
        # Decoder to aggregate population activity
        self.population_decoder = PopulationDecoder(
            population_size=population_size,
            output_dim=output_dim,
            encoding_scheme=encoding_scheme
        )
        
        # Optional: learnable weights for combining population responses
        self.population_weights = nn.Parameter(
            torch.ones(population_size) / population_size
        )
        
        logger.info(f"🧠 Population coding initialized:")
        logger.info(f"   Population size: {population_size} neurons per dimension")
        logger.info(f"   Total neurons: {self.total_neurons}")
        logger.info(f"   Encoding: {encoding_scheme}")
        logger.info(f"   Threshold range: [{self.thresholds.min():.3f}, {self.thresholds.max():.3f}]")
    
    def _generate_thresholds(self, distribution: str) -> torch.Tensor:
        """
        Generate threshold distribution for population neurons.
        
        Different distributions create different encoding properties:
        - uniform: Equal spacing, good for linear ranges
        - gaussian: More neurons near mean, good for natural distributions
        - log: More neurons for small values, good for Weber's law
        """
        if distribution == 'uniform':
            # Uniformly spaced thresholds from 0.5 to 2.0
            thresholds = torch.linspace(0.5, 2.0, self.population_size)
            
        elif distribution == 'gaussian':
            # Gaussian distribution centered at 1.0
            mean = 1.0
            std = 0.3
            thresholds = torch.randn(self.population_size) * std + mean
            thresholds = torch.clamp(thresholds, 0.3, 2.0)
            thresholds, _ = torch.sort(thresholds)
            
        elif distribution == 'log':
            # Logarithmic spacing (more neurons for small values)
            thresholds = torch.logspace(-0.5, 0.5, self.population_size)
            
        else:
            raise ValueError(f"Unknown threshold distribution: {distribution}")
        
        # Ensure thresholds are sorted
        thresholds, _ = torch.sort(thresholds)
        
        return thresholds
    
    def forward(
        self,
        x: torch.Tensor,
        return_population_activity: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Encode input using population coding.
        
        Args:
            x: Input tensor [batch, features] or [batch, seq_len, features]
            return_population_activity: Whether to return detailed population statistics
        
        Returns:
            decoded_output: Decoded population response [batch, output_dim] or [batch, seq_len, output_dim]
            population_stats: Optional dictionary with population activity statistics
        """
        original_shape = x.shape
        
        # Handle different input shapes
        if len(x.shape) == 2:
            batch_size, features = x.shape
            has_sequence = False
        elif len(x.shape) == 3:
            batch_size, seq_len, features = x.shape
            x = x.reshape(batch_size * seq_len, features)
            has_sequence = True
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Project input to population space
        population_input = self.input_projection(x)  # [batch, total_neurons]
        
        # Reshape to separate populations: [batch, output_dim, population_size]
        population_input = population_input.view(
            -1, self.output_dim, self.population_size
        )
        
        # Initialize population activity tracking
        population_spikes = []
        population_membranes = []
        
        # Process through time steps with population neurons
        for step in range(self.num_steps):
            step_spikes = []
            step_membranes = []
            
            # Each output dimension has its own population
            for dim_idx in range(self.output_dim):
                dim_spikes = []
                dim_membranes = []
                
                # Process each neuron in the population
                for pop_idx, neuron in enumerate(self.population_neurons):
                    # Get input for this specific neuron
                    neuron_input = population_input[:, dim_idx, pop_idx]
                    
                    # Get spike and membrane potential
                    if step == 0:
                        # Initialize hidden state
                        mem = neuron.init_leaky()
                    else:
                        # Use stored membrane potential
                        mem = step_membranes[-1] if dim_membranes else neuron.init_leaky()
                    
                    spk, mem = neuron(neuron_input, mem)
                    
                    dim_spikes.append(spk)
                    dim_membranes.append(mem)
                
                step_spikes.append(torch.stack(dim_spikes, dim=-1))
                step_membranes.append(torch.stack(dim_membranes, dim=-1))
            
            # Stack across output dimensions
            population_spikes.append(torch.stack(step_spikes, dim=1))
            population_membranes.append(torch.stack(step_membranes, dim=1))
        
        # Stack across time: [num_steps, batch, output_dim, population_size]
        population_spikes = torch.stack(population_spikes, dim=0)
        population_membranes = torch.stack(population_membranes, dim=0)
        
        # Decode population activity
        decoded_output = self.population_decoder(
            population_spikes,
            population_membranes,
            self.population_weights
        )
        
        # Restore original shape if needed
        if has_sequence:
            decoded_output = decoded_output.view(batch_size, seq_len, self.output_dim)
        
        # Compute population statistics if requested
        population_stats = None
        if return_population_activity:
            population_stats = self._compute_population_statistics(
                population_spikes,
                population_membranes
            )
        
        return decoded_output, population_stats
    
    def _compute_population_statistics(
        self,
        spikes: torch.Tensor,
        membranes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute detailed statistics about population activity."""
        # spikes: [num_steps, batch, output_dim, population_size]
        
        # Firing rates for each neuron in population
        firing_rates = spikes.float().mean(dim=0)  # [batch, output_dim, population_size]
        
        # Population synchrony (correlation between neurons)
        # High synchrony might indicate redundancy, low synchrony indicates diversity
        spikes_flat = spikes.permute(1, 2, 0, 3)  # [batch, output_dim, num_steps, population_size]
        population_correlation = self._compute_correlation(
            spikes_flat.reshape(-1, self.num_steps, self.population_size)
        )
        
        # Temporal precision (spike time variance within population)
        spike_times = torch.argmax(spikes.float(), dim=0)  # First spike time
        temporal_variance = spike_times.float().var(dim=-1)  # Variance across population
        
        # Population sparsity
        sparsity = (spikes.sum() / spikes.numel()).item()
        
        return {
            'firing_rates': firing_rates,
            'population_correlation': population_correlation,
            'temporal_variance': temporal_variance,
            'sparsity': sparsity,
            'mean_firing_rate': firing_rates.mean().item(),
            'max_firing_rate': firing_rates.max().item()
        }
    
    def _compute_correlation(self, spikes: torch.Tensor) -> torch.Tensor:
        """Compute average pairwise correlation within populations."""
        # spikes: [batch*output_dim, num_steps, population_size]
        spikes_centered = spikes - spikes.mean(dim=1, keepdim=True)
        
        # Compute correlation matrix for each batch
        correlations = []
        for i in range(spikes_centered.shape[0]):
            cov = torch.matmul(
                spikes_centered[i].T,
                spikes_centered[i]
            ) / self.num_steps
            
            std = spikes_centered[i].std(dim=0, keepdim=True)
            corr = cov / (std.T @ std + 1e-8)
            
            # Average off-diagonal correlations
            mask = ~torch.eye(self.population_size, dtype=bool, device=corr.device)
            avg_corr = corr[mask].mean()
            correlations.append(avg_corr)
        
        return torch.stack(correlations).mean()


class PopulationDecoder(nn.Module):
    """
    Decoder that aggregates population activity into meaningful output.
    Supports multiple decoding strategies.
    """
    
    def __init__(
        self,
        population_size: int,
        output_dim: int,
        encoding_scheme: str = 'hybrid'
    ):
        super().__init__()
        
        self.population_size = population_size
        self.output_dim = output_dim
        self.encoding_scheme = encoding_scheme
        
        if encoding_scheme == 'rate':
            # Rate coding: decode from average firing rate
            self.decoder = nn.Linear(population_size, 1)
            
        elif encoding_scheme == 'temporal':
            # Temporal coding: decode from spike timing
            self.temporal_conv = nn.Conv1d(population_size, population_size, kernel_size=3, padding=1)
            self.decoder = nn.Linear(population_size, 1)
            
        elif encoding_scheme == 'hybrid':
            # Hybrid: combine rate and temporal information
            self.rate_decoder = nn.Linear(population_size, 1)
            self.temporal_decoder = nn.Sequential(
                nn.Conv1d(population_size, population_size // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(population_size // 2, 1, kernel_size=1)
            )
            self.fusion = nn.Parameter(torch.tensor([0.6, 0.4]))  # Learnable fusion weights
        
        else:
            raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")
    
    def forward(
        self,
        spikes: torch.Tensor,
        membranes: torch.Tensor,
        population_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode population activity.
        
        Args:
            spikes: [num_steps, batch, output_dim, population_size]
            membranes: [num_steps, batch, output_dim, population_size]
            population_weights: [population_size] learnable weights
        
        Returns:
            decoded: [batch, output_dim]
        """
        num_steps, batch_size, output_dim, pop_size = spikes.shape
        
        if self.encoding_scheme == 'rate':
            # Rate-based decoding: average spike count
            firing_rates = spikes.float().mean(dim=0)  # [batch, output_dim, population_size]
            
            # Apply learnable weights
            weighted_rates = firing_rates * population_weights.unsqueeze(0).unsqueeze(0)
            
            # Decode to single value per output dimension
            decoded = self.decoder(weighted_rates).squeeze(-1)  # [batch, output_dim]
            
        elif self.encoding_scheme == 'temporal':
            # Temporal decoding: consider spike timing
            # Reshape for temporal processing
            spikes_reshaped = spikes.permute(1, 2, 3, 0)  # [batch, output_dim, population_size, num_steps]
            spikes_flat = spikes_reshaped.reshape(-1, pop_size, num_steps)  # [batch*output_dim, pop_size, num_steps]
            
            # Apply temporal convolution
            temporal_features = self.temporal_conv(spikes_flat.float())
            temporal_features = temporal_features.mean(dim=-1)  # Average over time
            
            # Decode
            decoded = self.decoder(temporal_features)  # [batch*output_dim, 1]
            decoded = decoded.view(batch_size, output_dim)
            
        elif self.encoding_scheme == 'hybrid':
            # Combine rate and temporal information
            
            # Rate component
            firing_rates = spikes.float().mean(dim=0)  # [batch, output_dim, population_size]
            rate_decoded = self.rate_decoder(firing_rates).squeeze(-1)  # [batch, output_dim]
            
            # Temporal component
            spikes_reshaped = spikes.permute(1, 2, 3, 0)  # [batch, output_dim, population_size, num_steps]
            spikes_flat = spikes_reshaped.reshape(-1, pop_size, num_steps)
            temporal_features = self.temporal_decoder(spikes_flat.float())  # [batch*output_dim, 1, num_steps]
            temporal_decoded = temporal_features.mean(dim=-1).view(batch_size, output_dim)
            
            # Fuse with learnable weights
            fusion_weights = F.softmax(self.fusion, dim=0)
            decoded = (fusion_weights[0] * rate_decoded + 
                      fusion_weights[1] * temporal_decoded)
        
        return decoded


class PopulationSpikingEncoder(nn.Module):
    """
    Enhanced spiking encoder using population coding.
    Drop-in replacement for the standard SpikingEncoder.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_steps: int,
        beta: float = 0.95,
        population_size: int = 5,
        use_population_coding: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.use_population_coding = use_population_coding
        
        if use_population_coding:
            # Use population coding
            self.encoder = PopulationCoding(
                input_dim=input_dim,
                output_dim=output_dim,
                population_size=population_size,
                num_steps=num_steps,
                beta=beta,
                encoding_scheme='hybrid'
            )
        else:
            # Fallback to standard encoding
            self.fc1 = nn.Linear(input_dim, output_dim * 2)
            self.fc2 = nn.Linear(output_dim * 2, output_dim)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with population coding.
        
        Args:
            x: Input tensor
        
        Returns:
            Encoded output tensor
        """
        if self.use_population_coding:
            output, _ = self.encoder(x, return_population_activity=False)
            return output
        else:
            # Standard encoding fallback
            # ... (implement standard path as needed)
            pass


def integrate_population_coding(model, population_size: int = 5):
    """
    Replace standard spiking encoders with population-coded versions.
    
    Args:
        model: The LiquidSpikingNetwork model
        population_size: Number of neurons per population
    """
    # Find and replace spike encoders in hybrid blocks
    for name, module in model.named_modules():
        if isinstance(module, nn.Module) and hasattr(module, 'spike_encoder'):
            old_encoder = module.spike_encoder
            
            # Create new population-coded encoder
            new_encoder = PopulationSpikingEncoder(
                input_dim=old_encoder.fc1.in_features,
                output_dim=old_encoder.fc2.out_features,
                num_steps=old_encoder.num_steps,
                beta=old_encoder.lif1.beta if hasattr(old_encoder.lif1, 'beta') else 0.95,
                population_size=population_size,
                use_population_coding=True
            )
            
            # Replace encoder
            module.spike_encoder = new_encoder
            
            logger.info(f"✓ Replaced spike encoder in {name} with population coding (size={population_size})")
    
    logger.info(f"🧠 Population coding integration complete")


# Example usage and testing
if __name__ == "__main__":
    # Test population coding
    print("Testing Population Coding...")
    
    # Create population encoder
    encoder = PopulationCoding(
        input_dim=128,
        output_dim=64,
        population_size=5,
        num_steps=32,
        encoding_scheme='hybrid'
    )
    
    # Test forward pass
    x = torch.randn(4, 128)  # Batch of 4, 128 features
    output, stats = encoder(x, return_population_activity=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Population statistics:")
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {value}")