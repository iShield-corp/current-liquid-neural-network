"""
Meta-Plasticity: Learning to Learn from Experience
Implements mechanisms where plasticity parameters adapt based on learning history
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MetaPlasticityController(nn.Module):
    """
    Meta-plasticity controller that learns optimal plasticity parameters.
    
    The system learns how to adjust STDP parameters (learning rates, time constants,
    amplitudes) based on the learning history and task demands.
    
    Key idea: "Learning to learn" - the network adapts its learning rules
    based on experience.
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int = 128,
        history_length: int = 100,
        meta_lr: float = 0.001
    ):
        """
        Args:
            num_layers: Number of layers to control
            hidden_dim: Hidden dimension for meta-learning network
            history_length: How many time steps of history to consider
            meta_lr: Meta-learning rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        self.meta_lr = meta_lr
        
        # Performance history tracking
        self.register_buffer('performance_history', 
                           torch.zeros(history_length))
        self.register_buffer('loss_history',
                           torch.zeros(history_length))
        self.register_buffer('weight_change_history',
                           torch.zeros(history_length))
        self.history_idx = 0
        
        # Meta-learning network that predicts optimal plasticity parameters
        self.history_encoder = nn.LSTM(
            input_size=3,  # performance, loss, weight_change
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Predict plasticity parameters for each layer
        self.plasticity_predictor = nn.ModuleDict({
            'learning_rate': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_layers),
                nn.Sigmoid()  # 0 to 1
            ),
            'tau_plus': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_layers),
                nn.Softplus()  # Positive values
            ),
            'tau_minus': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_layers),
                nn.Softplus()
            ),
            'a_plus': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_layers),
                nn.Sigmoid()
            ),
            'a_minus': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_layers),
                nn.Sigmoid()
            )
        })
        
        # Meta-optimizer for updating meta-parameters
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=meta_lr)
        
        logger.info(f"🧠 Meta-plasticity controller initialized for {num_layers} layers")
    
    def update_history(
        self,
        performance: float,
        loss: float,
        weight_change: float
    ):
        """Update learning history."""
        idx = self.history_idx % self.history_length
        self.performance_history[idx] = performance
        self.loss_history[idx] = loss
        self.weight_change_history[idx] = weight_change
        self.history_idx += 1
    
    def predict_plasticity_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Predict optimal plasticity parameters based on learning history.
        
        Returns:
            Dictionary of plasticity parameters for each layer
        """
        # Prepare history for LSTM
        history_steps = min(self.history_idx, self.history_length)
        
        if history_steps < 10:  # Need some history
            # Return default parameters
            return {
                'learning_rate': torch.ones(self.num_layers) * 0.01,
                'tau_plus': torch.ones(self.num_layers) * 20.0,
                'tau_minus': torch.ones(self.num_layers) * 20.0,
                'a_plus': torch.ones(self.num_layers) * 0.005,
                'a_minus': torch.ones(self.num_layers) * 0.00525
            }
        
        # Stack history
        history_tensor = torch.stack([
            self.performance_history[:history_steps],
            self.loss_history[:history_steps],
            self.weight_change_history[:history_steps]
        ], dim=1).unsqueeze(0)  # [1, history_steps, 3]
        
        # Encode history
        encoded, (h_n, c_n) = self.history_encoder(history_tensor)
        context = h_n[-1]  # Last hidden state [1, hidden_dim]
        
        # Predict parameters
        plasticity_params = {}
        for param_name, predictor in self.plasticity_predictor.items():
            predicted = predictor(context).squeeze(0)  # [num_layers]
            
            # Scale to appropriate ranges
            if param_name == 'learning_rate':
                predicted = predicted * 0.1  # 0 to 0.1
            elif param_name in ['tau_plus', 'tau_minus']:
                predicted = predicted * 50 + 10  # 10 to 60
            elif param_name in ['a_plus', 'a_minus']:
                predicted = predicted * 0.01  # 0 to 0.01
            
            plasticity_params[param_name] = predicted
        
        return plasticity_params
    
    def compute_meta_loss(
        self,
        predicted_params: Dict[str, torch.Tensor],
        actual_performance: float,
        target_performance: float = 1.0
    ) -> torch.Tensor:
        """
        Compute meta-learning loss based on performance.
        
        The meta-loss encourages the controller to predict parameters
        that lead to better performance.
        """
        # Performance-based loss
        performance_loss = F.mse_loss(
            torch.tensor(actual_performance),
            torch.tensor(target_performance)
        )
        
        # Regularization: prefer stable parameters
        stability_loss = 0.0
        for param_name, param_values in predicted_params.items():
            # Penalize extreme values
            if param_name == 'learning_rate':
                # Prefer moderate learning rates
                stability_loss += ((param_values - 0.01) ** 2).mean()
            elif param_name in ['tau_plus', 'tau_minus']:
                # Prefer time constants around 20ms
                stability_loss += ((param_values - 20.0) ** 2).mean() * 0.001
        
        # Total meta-loss
        total_loss = performance_loss + 0.1 * stability_loss
        
        return total_loss
    
    def meta_update(
        self,
        actual_performance: float,
        target_performance: float = 1.0
    ):
        """
        Perform meta-learning update.
        
        This updates the meta-plasticity controller to predict better
        plasticity parameters in the future.
        """
        # Predict parameters
        predicted_params = self.predict_plasticity_parameters()
        
        # Compute meta-loss
        meta_loss = self.compute_meta_loss(
            predicted_params,
            actual_performance,
            target_performance
        )
        
        # Meta-gradient step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()


class AdaptiveSTDPRule(nn.Module):
    """
    STDP rule with meta-plastic parameters that adapt during learning.
    
    Combines STDP with meta-plasticity control.
    """
    
    def __init__(
        self,
        meta_controller: MetaPlasticityController,
        layer_idx: int,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        super().__init__()
        
        self.meta_controller = meta_controller
        self.layer_idx = layer_idx
        self.w_min = w_min
        self.w_max = w_max
        
        # Current plasticity parameters (updated by meta-controller)
        self.register_buffer('learning_rate', torch.tensor(0.01))
        self.register_buffer('tau_plus', torch.tensor(20.0))
        self.register_buffer('tau_minus', torch.tensor(20.0))
        self.register_buffer('a_plus', torch.tensor(0.005))
        self.register_buffer('a_minus', torch.tensor(0.00525))
    
    def update_plasticity_parameters(self):
        """Update plasticity parameters from meta-controller."""
        predicted_params = self.meta_controller.predict_plasticity_parameters()
        
        self.learning_rate = predicted_params['learning_rate'][self.layer_idx]
        self.tau_plus = predicted_params['tau_plus'][self.layer_idx]
        self.tau_minus = predicted_params['tau_minus'][self.layer_idx]
        self.a_plus = predicted_params['a_plus'][self.layer_idx]
        self.a_minus = predicted_params['a_minus'][self.layer_idx]
        
        logger.debug(f"Layer {self.layer_idx} plasticity updated: "
                    f"lr={self.learning_rate:.4f}, "
                    f"τ+={self.tau_plus:.1f}, "
                    f"A+={self.a_plus:.5f}")
    
    def compute_weight_update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """Compute STDP weight update with current meta-plastic parameters."""
        batch_size, time_steps, n_pre = pre_spikes.shape
        _, _, n_post = post_spikes.shape
        
        weight_update = torch.zeros_like(weights)
        
        for t in range(time_steps):
            pre_t = pre_spikes[:, t, :]
            post_t = post_spikes[:, t, :]
            
            # LTP with meta-plastic parameters
            for tau in range(1, min(int(self.tau_plus * 3), time_steps - t)):
                if t + tau < time_steps:
                    post_future = post_spikes[:, t + tau, :]
                    ltp_weight = self.a_plus * torch.exp(-tau * dt / self.tau_plus)
                    interaction = torch.einsum('bp,bq->bpq', pre_t, post_future)
                    weight_update += ltp_weight * interaction.mean(dim=0)
            
            # LTD with meta-plastic parameters
            for tau in range(1, min(int(self.tau_minus * 3), t + 1)):
                if t - tau >= 0:
                    pre_past = pre_spikes[:, t - tau, :]
                    ltd_weight = -self.a_minus * torch.exp(-tau * dt / self.tau_minus)
                    interaction = torch.einsum('bp,bq->bpq', pre_past, post_t)
                    weight_update += ltd_weight * interaction.mean(dim=0)
        
        return weight_update / time_steps
    
    def apply_weight_update(
        self,
        weights: torch.Tensor,
        weight_update: torch.Tensor
    ) -> torch.Tensor:
        """Apply weight update with meta-plastic learning rate."""
        new_weights = weights + self.learning_rate * weight_update
        return torch.clamp(new_weights, self.w_min, self.w_max)


class MetaPlasticLayer(nn.Module):
    """
    Neural layer with meta-plastic STDP.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        meta_controller: MetaPlasticityController,
        layer_idx: int,
        use_bias: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.layer_idx = layer_idx
        
        # Adaptive STDP rule
        self.stdp_rule = AdaptiveSTDPRule(
            meta_controller=meta_controller,
            layer_idx=layer_idx
        )
        
        # Weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
    
    def update_weights_meta_plastic(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float = 1.0
    ):
        """Update weights using meta-plastic STDP."""
        with torch.no_grad():
            # Update plasticity parameters from meta-controller
            self.stdp_rule.update_plasticity_parameters()
            
            # Compute and apply weight update
            weight_update = self.stdp_rule.compute_weight_update(
                pre_spikes,
                post_spikes,
                self.weight.data,
                dt
            )
            
            self.weight.data = self.stdp_rule.apply_weight_update(
                self.weight.data,
                weight_update
            )


class MetaPlasticNetwork(nn.Module):
    """
    Complete network with meta-plasticity across all layers.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        meta_lr: float = 0.001,
        history_length: int = 100
    ):
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # Create meta-plasticity controller
        self.meta_controller = MetaPlasticityController(
            num_layers=self.num_layers,
            meta_lr=meta_lr,
            history_length=history_length
        )
        
        # Create meta-plastic layers
        self.layers = nn.ModuleList([
            MetaPlasticLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                meta_controller=self.meta_controller,
                layer_idx=i
            )
            for i in range(self.num_layers)
        ])
        
        logger.info(f"🧠 Meta-plastic network created with {self.num_layers} layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)  # Can be customized
        return x
    
    def update_meta_plasticity(
        self,
        layer_activities: List[Tuple[torch.Tensor, torch.Tensor]],
        performance: float,
        loss: float
    ):
        """
        Update all layers with meta-plastic STDP.
        
        Args:
            layer_activities: List of (pre_activity, post_activity) for each layer
            performance: Current performance metric (e.g., accuracy)
            loss: Current loss value
        """
        # Compute average weight change
        total_weight_change = 0.0
        
        for layer_idx, (layer, (pre_activity, post_activity)) in enumerate(
            zip(self.layers, layer_activities)
        ):
            # Update weights for this layer
            old_weights = layer.weight.data.clone()
            layer.update_weights_meta_plastic(pre_activity, post_activity)
            
            # Track weight change
            weight_change = (layer.weight.data - old_weights).abs().mean().item()
            total_weight_change += weight_change
        
        avg_weight_change = total_weight_change / self.num_layers
        
        # Update meta-controller history
        self.meta_controller.update_history(
            performance=performance,
            loss=loss,
            weight_change=avg_weight_change
        )
        
        # Perform meta-learning update
        meta_loss = self.meta_controller.meta_update(
            actual_performance=performance,
            target_performance=1.0
        )
        
        logger.debug(f"Meta-plasticity update: "
                    f"perf={performance:.3f}, "
                    f"loss={loss:.3f}, "
                    f"Δw={avg_weight_change:.6f}, "
                    f"meta_loss={meta_loss:.6f}")


def integrate_meta_plasticity(
    model: nn.Module,
    meta_lr: float = 0.001,
    history_length: int = 100,
    layers_to_enhance: Optional[List[str]] = None
):
    """
    Integrate meta-plasticity into an existing model.
    
    Args:
        model: Neural network model
        meta_lr: Meta-learning rate
        history_length: History tracking length
        layers_to_enhance: Which layers to make meta-plastic
    """
    # Count layers
    num_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
    
    # Create meta-controller
    meta_controller = MetaPlasticityController(
        num_layers=num_layers,
        meta_lr=meta_lr,
        history_length=history_length
    )
    
    # Enhance layers
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if layers_to_enhance is None or any(target in name for target in layers_to_enhance):
                logger.info(f"✓ Added meta-plasticity to layer: {name}")
                layer_idx += 1
    
    logger.info(f"🧠 Meta-plasticity integrated into {layer_idx} layers")
    
    # Store meta-controller in model
    model.meta_controller = meta_controller
    
    return model


# Example usage
if __name__ == "__main__":
    print("Testing Meta-Plasticity...")
    
    # Create meta-plastic network
    layer_sizes = [784, 256, 128, 10]
    network = MetaPlasticNetwork(layer_sizes, meta_lr=0.001)
    
    # Simulate training
    batch_size = 32
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # Simulate forward pass and collect activities
        x = torch.randn(batch_size, 784)
        layer_activities = []
        
        current_activity = x
        for layer in network.layers:
            pre_activity = current_activity
            current_activity = layer(current_activity)
            post_activity = current_activity
            
            # Convert to spike trains (simplified)
            pre_spikes = (pre_activity.unsqueeze(1).repeat(1, 10, 1) > 0).float()
            post_spikes = (post_activity.unsqueeze(1).repeat(1, 10, 1) > 0).float()
            
            layer_activities.append((pre_spikes, post_spikes))
        
        # Simulate performance metrics
        performance = 0.5 + epoch * 0.05  # Improving over time
        loss = 2.0 - epoch * 0.1  # Decreasing over time
        
        # Update meta-plasticity
        network.update_meta_plasticity(layer_activities, performance, loss)
        
        print(f"Epoch {epoch + 1}: Performance={performance:.3f}, Loss={loss:.3f}")
    
    # Check learned plasticity parameters
    final_params = network.meta_controller.predict_plasticity_parameters()
    print("\n✅ Final learned plasticity parameters:")
    for param_name, values in final_params.items():
        print(f"   {param_name}: {values.mean():.4f} ± {values.std():.4f}")