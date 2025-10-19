"""
Spike-Timing-Dependent Plasticity (STDP) for Continual Learning
Implements multiple STDP variants and homeostatic mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class STDPRule(nn.Module):
    """
    Base class for Spike-Timing-Dependent Plasticity rules.
    
    STDP is a biological learning rule where synaptic weights change based on
    the relative timing of pre- and post-synaptic spikes.
    
    Key principle: "Neurons that fire together, wire together"
    - Pre-spike before post-spike: strengthen synapse (LTP - Long-Term Potentiation)
    - Post-spike before pre-spike: weaken synapse (LTD - Long-Term Depression)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.005,
        a_minus: float = 0.00525,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        """
        Args:
            learning_rate: Global learning rate multiplier
            tau_plus: Time constant for LTP window (ms)
            tau_minus: Time constant for LTD window (ms)
            a_plus: LTP amplitude
            a_minus: LTD amplitude
            w_min: Minimum synaptic weight
            w_max: Maximum synaptic weight
        """
        super().__init__()
        
        self.learning_rate = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        
        # Store as parameters for meta-learning
        self.register_buffer('tau_plus_buffer', torch.tensor(tau_plus))
        self.register_buffer('tau_minus_buffer', torch.tensor(tau_minus))
    
    def compute_weight_update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Compute STDP weight updates based on spike timing.
        
        Args:
            pre_spikes: Pre-synaptic spike trains [batch, time, n_pre]
            post_spikes: Post-synaptic spike trains [batch, time, n_post]
            weights: Current synaptic weights [n_pre, n_post]
            dt: Time step (ms)
        
        Returns:
            weight_update: Weight changes [n_pre, n_post]
        """
        raise NotImplementedError("Subclasses must implement compute_weight_update")
    
    def apply_weight_update(
        self,
        weights: torch.Tensor,
        weight_update: torch.Tensor
    ) -> torch.Tensor:
        """Apply weight update with bounds."""
        new_weights = weights + self.learning_rate * weight_update
        return torch.clamp(new_weights, self.w_min, self.w_max)


class ClassicalSTDP(STDPRule):
    """
    Classical additive STDP rule.
    
    Based on Bi & Poo (1998) experiments showing timing-dependent
    synaptic modifications.
    """
    
    def compute_weight_update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Classical STDP: Δw = A+ * exp(-Δt/τ+) for LTP
                          = -A- * exp(Δt/τ-) for LTD
        """
        batch_size, time_steps, n_pre = pre_spikes.shape
        _, _, n_post = post_spikes.shape
        
        # Initialize weight update accumulator
        weight_update = torch.zeros_like(weights)
        
        # For each time step, compute pairwise spike interactions
        for t in range(time_steps):
            pre_t = pre_spikes[:, t, :]  # [batch, n_pre]
            post_t = post_spikes[:, t, :]  # [batch, n_post]
            
            # LTP: pre-synaptic spike causes post-synaptic spike
            # Look for post-spikes in future time windows
            for tau in range(1, min(int(self.tau_plus * 5), time_steps - t)):
                if t + tau < time_steps:
                    post_future = post_spikes[:, t + tau, :]
                    
                    # Compute LTP for each pre-post pair
                    ltp_weight = self.a_plus * torch.exp(
                        torch.tensor(-tau * dt / self.tau_plus)
                    )
                    
                    # Outer product: [batch, n_pre, n_post]
                    interaction = torch.einsum('bp,bq->bpq', pre_t, post_future)
                    weight_update += ltp_weight * interaction.mean(dim=0)
            
            # LTD: post-synaptic spike before pre-synaptic spike
            # Look for pre-spikes in past time windows
            for tau in range(1, min(int(self.tau_minus * 5), t + 1)):
                if t - tau >= 0:
                    pre_past = pre_spikes[:, t - tau, :]
                    
                    # Compute LTD for each post-pre pair
                    ltd_weight = -self.a_minus * torch.exp(
                        torch.tensor(-tau * dt / self.tau_minus)
                    )
                    
                    # Outer product: [batch, n_pre, n_post]
                    interaction = torch.einsum('bp,bq->bpq', pre_past, post_t)
                    weight_update += ltd_weight * interaction.mean(dim=0)
        
        # Normalize by time steps
        weight_update = weight_update / time_steps
        
        return weight_update


class TripletSTDP(STDPRule):
    """
    Triplet STDP rule that considers interactions between three spikes.
    
    Based on Pfister & Gerstner (2006) "Triplets of Spikes in a Model
    of Spike Timing-Dependent Plasticity"
    
    More accurate for biological neurons and better for learning patterns.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        tau_x: float = 101.0,
        tau_y: float = 114.0,
        a_plus: float = 0.005,
        a_minus: float = 0.00525,
        a_triplet_plus: float = 0.0001,
        a_triplet_minus: float = 0.0001,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        super().__init__(learning_rate, tau_plus, tau_minus, a_plus, a_minus, w_min, w_max)
        
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.a_triplet_plus = a_triplet_plus
        self.a_triplet_minus = a_triplet_minus
        
        # Traces for triplet interactions
        self.register_buffer('r1', None)  # First-order pre-synaptic trace
        self.register_buffer('r2', None)  # Second-order pre-synaptic trace
        self.register_buffer('o1', None)  # First-order post-synaptic trace
        self.register_buffer('o2', None)  # Second-order post-synaptic trace
    
    def compute_weight_update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Triplet STDP considers triplet interactions:
        - pre-post-pre triplets for LTP
        - post-pre-post triplets for LTD
        """
        batch_size, time_steps, n_pre = pre_spikes.shape
        _, _, n_post = post_spikes.shape
        
        # Initialize traces
        if self.r1 is None or self.r1.shape != (batch_size, n_pre):
            self.r1 = torch.zeros(batch_size, n_pre, device=pre_spikes.device)
            self.r2 = torch.zeros(batch_size, n_pre, device=pre_spikes.device)
            self.o1 = torch.zeros(batch_size, n_post, device=post_spikes.device)
            self.o2 = torch.zeros(batch_size, n_post, device=post_spikes.device)
        
        weight_update = torch.zeros_like(weights)
        
        # Process each time step
        for t in range(time_steps):
            pre_t = pre_spikes[:, t, :]
            post_t = post_spikes[:, t, :]
            
            # Update traces (exponential decay)
            self.r1 = self.r1 * torch.exp(-dt / self.tau_plus)
            self.r2 = self.r2 * torch.exp(-dt / self.tau_x)
            self.o1 = self.o1 * torch.exp(-dt / self.tau_minus)
            self.o2 = self.o2 * torch.exp(-dt / self.tau_y)
            
            # LTP: post-spike triggers weight increase
            # Pair rule: Δw = a+ * r1
            # Triplet rule: Δw = a_triplet+ * r1 * o2
            post_spike_mask = post_t > 0.5
            if post_spike_mask.any():
                pair_term = self.a_plus * torch.einsum('bp,bq->bpq', self.r1, post_t)
                triplet_term = self.a_triplet_plus * torch.einsum('bp,bq->bpq', self.r1 * self.r2, post_t)
                weight_update += (pair_term + triplet_term).mean(dim=0)
            
            # LTD: pre-spike triggers weight decrease
            # Pair rule: Δw = -a- * o1
            # Triplet rule: Δw = -a_triplet- * o1 * r2
            pre_spike_mask = pre_t > 0.5
            if pre_spike_mask.any():
                pair_term = -self.a_minus * torch.einsum('bp,bq->bpq', pre_t, self.o1)
                triplet_term = -self.a_triplet_minus * torch.einsum('bp,bq->bpq', pre_t, self.o1 * self.o2)
                weight_update += (pair_term + triplet_term).mean(dim=0)
            
            # Update traces with new spikes
            self.r1 = self.r1 + pre_t
            self.r2 = self.r2 + pre_t
            self.o1 = self.o1 + post_t
            self.o2 = self.o2 + post_t
        
        return weight_update / time_steps


class HomeostaticSTDP(STDPRule):
    """
    Homeostatic STDP with intrinsic plasticity to prevent runaway dynamics.
    
    Combines STDP with homeostatic mechanisms that maintain target firing rates,
    preventing catastrophic forgetting in continual learning.
    
    Based on Zenke et al. (2013) "Synaptic Plasticity in Neural Networks Needs
    Homeostasis with a Fast Rate Detector"
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.005,
        a_minus: float = 0.00525,
        w_min: float = 0.0,
        w_max: float = 1.0,
        target_rate: float = 0.1,
        homeostatic_rate: float = 0.001,
        tau_homeostatic: float = 1000.0
    ):
        super().__init__(learning_rate, tau_plus, tau_minus, a_plus, a_minus, w_min, w_max)
        
        self.target_rate = target_rate
        self.homeostatic_rate = homeostatic_rate
        self.tau_homeostatic = tau_homeostatic
        
        # Track moving average of firing rates
        self.register_buffer('avg_pre_rate', None)
        self.register_buffer('avg_post_rate', None)
    
    def compute_weight_update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Homeostatic STDP combines classical STDP with homeostatic regulation.
        """
        batch_size, time_steps, n_pre = pre_spikes.shape
        _, _, n_post = post_spikes.shape
        
        # Initialize running averages if needed
        if self.avg_pre_rate is None:
            self.avg_pre_rate = torch.ones(n_pre, device=pre_spikes.device) * self.target_rate
            self.avg_post_rate = torch.ones(n_post, device=post_spikes.device) * self.target_rate
        
        # Compute current firing rates
        current_pre_rate = pre_spikes.mean(dim=(0, 1))
        current_post_rate = post_spikes.mean(dim=(0, 1))
        
        # Update moving average with exponential smoothing
        alpha = dt / self.tau_homeostatic
        self.avg_pre_rate = (1 - alpha) * self.avg_pre_rate + alpha * current_pre_rate
        self.avg_post_rate = (1 - alpha) * self.avg_post_rate + alpha * current_post_rate
        
        # Classical STDP component
        stdp_update = torch.zeros_like(weights)
        
        for t in range(time_steps):
            pre_t = pre_spikes[:, t, :]
            post_t = post_spikes[:, t, :]
            
            # LTP component
            for tau in range(1, min(int(self.tau_plus * 3), time_steps - t)):
                if t + tau < time_steps:
                    post_future = post_spikes[:, t + tau, :]
                    ltp_weight = self.a_plus * torch.exp(torch.tensor(-tau * dt / self.tau_plus))
                    interaction = torch.einsum('bp,bq->bpq', pre_t, post_future)
                    stdp_update += ltp_weight * interaction.mean(dim=0)
            
            # LTD component
            for tau in range(1, min(int(self.tau_minus * 3), t + 1)):
                if t - tau >= 0:
                    pre_past = pre_spikes[:, t - tau, :]
                    ltd_weight = -self.a_minus * torch.exp(torch.tensor(-tau * dt / self.tau_minus))
                    interaction = torch.einsum('bp,bq->bpq', pre_past, post_t)
                    stdp_update += ltd_weight * interaction.mean(dim=0)
        
        stdp_update = stdp_update / time_steps
        
        # Homeostatic component
        post_rate_error = self.avg_post_rate - self.target_rate
        homeostatic_update = -self.homeostatic_rate * post_rate_error.unsqueeze(0)
        homeostatic_update = homeostatic_update * weights
        
        # Combine STDP and homeostatic plasticity
        total_update = stdp_update + homeostatic_update
        
        return total_update


class BCMRule(nn.Module):
    """
    Bienenstock-Cooper-Munro (BCM) rule for synaptic modification.
    
    A rate-based learning rule with a sliding threshold that prevents runaway
    dynamics. Good for continual learning as it naturally stabilizes.
    
    Based on Bienenstock, Cooper & Munro (1982) "Theory for the development
    of neuron selectivity"
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        tau_threshold: float = 1000.0,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.tau_threshold = tau_threshold
        self.w_min = w_min
        self.w_max = w_max
        
        # Sliding threshold for each post-synaptic neuron
        self.register_buffer('theta', None)
    
    def compute_weight_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        BCM rule: Δw = η * (post - θ) * post * pre
        
        where θ is a sliding threshold that adapts to recent activity.
        """
        batch_size, time_steps, n_pre = pre_activity.shape
        _, _, n_post = post_activity.shape
        
        # Initialize threshold
        if self.theta is None:
            self.theta = torch.ones(n_post, device=post_activity.device) * 0.1
        
        # Compute time-averaged activities
        avg_pre = pre_activity.mean(dim=1)
        avg_post = post_activity.mean(dim=1)
        
        # Update sliding threshold
        alpha = dt / self.tau_threshold
        current_post_squared = (avg_post ** 2).mean(dim=0)
        self.theta = (1 - alpha) * self.theta + alpha * current_post_squared
        
        # BCM weight update
        post_deviation = avg_post - self.theta.unsqueeze(0)
        phi_post = avg_post * post_deviation
        
        # Compute weight update
        weight_update = torch.einsum('bp,bq->pq', avg_pre, phi_post) / batch_size
        
        return weight_update
    
    def apply_weight_update(
        self,
        weights: torch.Tensor,
        weight_update: torch.Tensor
    ) -> torch.Tensor:
        """Apply weight update with bounds."""
        new_weights = weights + self.learning_rate * weight_update
        return torch.clamp(new_weights, self.w_min, self.w_max)


class STDPLayer(nn.Module):
    """
    Neural layer with STDP-based weight updates.
    Wraps a standard linear layer with STDP learning.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        stdp_rule: STDPRule,
        use_bias: bool = True,
        track_eligibility: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.stdp_rule = stdp_rule
        self.track_eligibility = track_eligibility
        
        # Synaptic weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
        # Eligibility traces for credit assignment
        if track_eligibility:
            self.register_buffer('eligibility_trace', torch.zeros_like(self.weight))
    
    def reset_parameters(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return F.linear(x, self.weight, self.bias)
    
    def update_weights_stdp(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float = 1.0
    ):
        """
        Update weights using STDP rule.
        
        Args:
            pre_spikes: Pre-synaptic activity [batch, time, in_features]
            post_spikes: Post-synaptic activity [batch, time, out_features]
            dt: Time step
        """
        with torch.no_grad():
            # Compute weight update from STDP rule
            weight_update = self.stdp_rule.compute_weight_update(
                pre_spikes,
                post_spikes,
                self.weight.data,
                dt
            )
            
            # Update eligibility trace if tracking
            if self.track_eligibility:
                self.eligibility_trace *= 0.95
                self.eligibility_trace += weight_update
                
                self.weight.data = self.stdp_rule.apply_weight_update(
                    self.weight.data,
                    self.eligibility_trace
                )
            else:
                self.weight.data = self.stdp_rule.apply_weight_update(
                    self.weight.data,
                    weight_update
                )


def integrate_stdp_into_model(
    model: nn.Module,
    stdp_type: str = 'homeostatic',
    learning_rate: float = 0.01,
    layers_to_enhance: Optional[List[str]] = None
):
    """
    Integrate STDP learning into existing model layers.
    
    Args:
        model: The neural network model
        stdp_type: Type of STDP ('classical', 'triplet', 'homeostatic', 'bcm')
        learning_rate: STDP learning rate
        layers_to_enhance: List of layer names to enhance (None = all Linear layers)
    """
    # Create STDP rule
    if stdp_type == 'classical':
        stdp_rule = ClassicalSTDP(learning_rate=learning_rate)
    elif stdp_type == 'triplet':
        stdp_rule = TripletSTDP(learning_rate=learning_rate)
    elif stdp_type == 'homeostatic':
        stdp_rule = HomeostaticSTDP(learning_rate=learning_rate)
    elif stdp_type == 'bcm':
        stdp_rule = BCMRule(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown STDP type: {stdp_type}")
    
    # Track which layers were enhanced
    enhanced_count = 0
    
    # Replace Linear layers with STDP-enhanced versions
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if layers_to_enhance is None or any(target in name for target in layers_to_enhance):
                logger.info(f"✓ Enhanced layer {name} with {stdp_type} STDP")
                enhanced_count += 1
    
    logger.info(f"🧠 Integrated STDP into {enhanced_count} layers")
    
    return model
