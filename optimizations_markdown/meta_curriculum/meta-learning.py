"""
Meta-Learning Module for Hybrid Liquid-Spiking Neural Networks
Implements MAML (Model-Agnostic Meta-Learning) adapted for liquid neural networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import copy
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class LiquidMAML:
    """
    MAML implementation specifically designed for liquid-spiking networks.
    Focuses on adapting liquid time constants and spike dynamics quickly.
    """
    
    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-2,
        num_inner_steps: int = 5,
        first_order: bool = False,
        adapt_liquid_only: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            model: The hybrid liquid-spiking network
            meta_lr: Meta-learning rate (outer loop)
            inner_lr: Task adaptation learning rate (inner loop)
            num_inner_steps: Number of gradient steps for task adaptation
            first_order: Use first-order approximation (faster, less accurate)
            adapt_liquid_only: Only adapt liquid parameters during inner loop
            device: Device for computation
        """
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        self.adapt_liquid_only = adapt_liquid_only
        self.device = device
        
        # Meta-optimizer operates on original model parameters
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
        # Track which parameters to adapt in inner loop
        self.adaptable_params = self._identify_adaptable_parameters()
        
        logger.info(f"🧠 MAML initialized with {len(self.adaptable_params)} adaptable parameters")
    
    def _identify_adaptable_parameters(self) -> List[str]:
        """Identify which parameters should be adapted in the inner loop."""
        adaptable = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if self.adapt_liquid_only:
                # Focus on liquid components and time constants
                if any(keyword in name.lower() for keyword in 
                       ['liquid', 'time_constant', 'tau', 'cfc', 'ltc', 'ncp']):
                    adaptable.append(name)
            else:
                # Adapt all parameters
                adaptable.append(name)
        
        return adaptable
    
    def clone_model_with_params(self, params: OrderedDict) -> nn.Module:
        """Create a functional clone of the model with given parameters."""
        # Create a deep copy of the model
        cloned_model = copy.deepcopy(self.model)
        
        # Replace parameters
        state_dict = cloned_model.state_dict()
        for name, param in params.items():
            if name in state_dict:
                state_dict[name] = param
        
        cloned_model.load_state_dict(state_dict)
        return cloned_model
    
    def inner_loop_adaptation(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        criterion: nn.Module
    ) -> Tuple[OrderedDict, float]:
        """
        Perform inner loop adaptation on support set.
        
        Returns:
            adapted_params: Adapted parameter dictionary
            adaptation_loss: Final loss after adaptation
        """
        # Get current parameters
        current_params = OrderedDict(self.model.named_parameters())
        adapted_params = OrderedDict()
        
        # Initialize adapted params with current values
        for name, param in current_params.items():
            if name in self.adaptable_params:
                adapted_params[name] = param.clone()
        
        # Inner loop optimization
        for step in range(self.num_inner_steps):
            # Forward pass with current adapted parameters
            # We need to manually compute gradients for adapted params
            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            
            # Temporarily replace model parameters
            original_state = {name: p.data.clone() 
                            for name, p in self.model.named_parameters() 
                            if name in adapted_params}
            
            for name, param in adapted_params.items():
                self.model.get_parameter(name).data = param.data
            
            # Forward pass
            self.model.train()
            outputs = self.model(support_data)
            loss = criterion(outputs, support_labels)
            
            # Compute gradients w.r.t. adapted parameters
            grads = torch.autograd.grad(
                loss,
                [self.model.get_parameter(name) for name in adapted_params.keys()],
                create_graph=not self.first_order,
                allow_unused=True
            )
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
            
            # Restore original parameters
            for name, original_data in original_state.items():
                self.model.get_parameter(name).data = original_data
        
        # Compute final loss with adapted parameters
        for name, param in adapted_params.items():
            self.model.get_parameter(name).data = param.data
        
        with torch.no_grad():
            outputs = self.model(support_data)
            adaptation_loss = criterion(outputs, support_labels).item()
        
        return adapted_params, adaptation_loss
    
    def meta_train_step(
        self,
        task_batch: List[Dict[str, torch.Tensor]],
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Perform one meta-training step on a batch of tasks.
        
        Args:
            task_batch: List of tasks, each containing:
                - support_data: Training data for inner loop
                - support_labels: Training labels for inner loop
                - query_data: Test data for outer loop
                - query_labels: Test labels for outer loop
        
        Returns:
            Dictionary with meta-training metrics
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        task_losses = []
        adaptation_losses = []
        
        for task in task_batch:
            support_data = task['support_data']
            support_labels = task['support_labels']
            query_data = task['query_data']
            query_labels = task['query_labels']
            
            # Inner loop: adapt to support set
            adapted_params, adapt_loss = self.inner_loop_adaptation(
                support_data, support_labels, criterion
            )
            adaptation_losses.append(adapt_loss)
            
            # Outer loop: evaluate on query set with adapted parameters
            # Temporarily set adapted parameters
            original_state = {name: p.data.clone() 
                            for name, p in self.model.named_parameters() 
                            if name in adapted_params}
            
            for name, param in adapted_params.items():
                self.model.get_parameter(name).data = param.data
            
            # Forward pass on query set
            query_data = query_data.to(self.device)
            query_labels = query_labels.to(self.device)
            
            outputs = self.model(query_data)
            task_loss = criterion(outputs, query_labels)
            
            # Accumulate meta-loss
            meta_loss = meta_loss + task_loss / len(task_batch)
            task_losses.append(task_loss.item())
            
            # Restore original parameters before next task
            for name, original_data in original_state.items():
                self.model.get_parameter(name).data = original_data
        
        # Meta-gradient step
        meta_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'avg_task_loss': sum(task_losses) / len(task_losses),
            'avg_adaptation_loss': sum(adaptation_losses) / len(adaptation_losses),
            'num_tasks': len(task_batch)
        }
    
    def save_meta_checkpoint(self, filepath: str, epoch: int, metrics: Dict):
        """Save meta-learning checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'meta_lr': self.meta_lr,
            'inner_lr': self.inner_lr,
            'num_inner_steps': self.num_inner_steps,
            'metrics': metrics,
            'adaptable_params': self.adaptable_params
        }
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Meta-learning checkpoint saved to {filepath}")
    
    def load_meta_checkpoint(self, filepath: str):
        """Load meta-learning checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        logger.info(f"📂 Meta-learning checkpoint loaded from {filepath}")
        return checkpoint['epoch'], checkpoint['metrics']


class TaskSampler:
    """Sample tasks for meta-learning from a dataset."""
    
    def __init__(
        self,
        dataset,
        n_way: int = 5,
        k_shot: int = 5,
        query_size: int = 15,
        task_batch_size: int = 4
    ):
        """
        Args:
            dataset: Dataset to sample from
            n_way: Number of classes per task
            k_shot: Number of examples per class in support set
            query_size: Number of examples per class in query set
            task_batch_size: Number of tasks per meta-batch
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.task_batch_size = task_batch_size
    
    def sample_task_batch(self) -> List[Dict[str, torch.Tensor]]:
        """Sample a batch of tasks."""
        # This is a simplified version - adapt based on your dataset structure
        tasks = []
        
        for _ in range(self.task_batch_size):
            # Sample classes for this task
            # Sample support and query sets
            # Package into task dictionary
            
            task = {
                'support_data': torch.randn(self.k_shot * self.n_way, 3, 32, 32),
                'support_labels': torch.randint(0, self.n_way, (self.k_shot * self.n_way,)),
                'query_data': torch.randn(self.query_size * self.n_way, 3, 32, 32),
                'query_labels': torch.randint(0, self.n_way, (self.query_size * self.n_way,))
            }
            tasks.append(task)
        
        return tasks


def integrate_maml_with_trainer(trainer, enable_maml: bool = False):
    """
    Integrate MAML into existing Trainer class.
    Call this function to enable meta-learning in your training pipeline.
    """
    if not enable_maml:
        return
    
    # Create MAML wrapper
    trainer.maml = LiquidMAML(
        model=trainer.model,
        meta_lr=trainer.config.learning_rate,
        inner_lr=trainer.config.learning_rate * 10,  # Higher for fast adaptation
        num_inner_steps=5,
        device=trainer.device
    )
    
    # Create task sampler
    trainer.task_sampler = TaskSampler(
        dataset=trainer.train_dataset,
        n_way=5,
        k_shot=5,
        query_size=15,
        task_batch_size=4
    )
    
    logger.info("🧠 MAML meta-learning enabled for trainer")


# Example usage in training loop
def meta_learning_training_example():
    """Example of how to use meta-learning in training."""
    
    # Assume you have a trainer instance
    # trainer = Trainer(model, config)
    
    # Enable MAML
    # integrate_maml_with_trainer(trainer, enable_maml=True)
    
    # Meta-training loop
    # for meta_epoch in range(num_meta_epochs):
    #     # Sample batch of tasks
    #     task_batch = trainer.task_sampler.sample_task_batch()
    #     
    #     # Meta-training step
    #     metrics = trainer.maml.meta_train_step(task_batch, trainer.criterion)
    #     
    #     print(f"Meta-Epoch {meta_epoch}: {metrics}")
    
    pass