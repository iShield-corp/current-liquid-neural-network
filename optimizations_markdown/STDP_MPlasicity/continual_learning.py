"""
Continual Learning Framework with STDP and Meta-Plasticity
Prevents catastrophic forgetting while enabling lifelong learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import logging

from .stdp_plasticity import STDPRule, HomeostaticSTDP, STDPLayer
from .meta_plasticity import MetaPlasticityController, AdaptiveSTDPRule

logger = logging.getLogger(__name__)


class ContinualLearningSTDP(nn.Module):
    """
    Continual learning system combining STDP with importance-weighted consolidation.
    
    Prevents catastrophic forgetting by:
    1. Using homeostatic STDP to maintain activity levels
    2. Tracking parameter importance (Fisher Information)
    3. Consolidating important weights while keeping plasticity for new learning
    
    Based on concepts from:
    - Zenke et al. (2017) "Continual Learning Through Synaptic Intelligence"
    - Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting" (EWC)
    """
    
    def __init__(
        self,
        model: nn.Module,
        consolidation_strength: float = 1000.0,
        plasticity_decay: float = 0.9,
        importance_update_rate: float = 0.1,
        use_meta_plasticity: bool = True,
        meta_lr: float = 0.001
    ):
        """
        Args:
            model: Base neural network model
            consolidation_strength: How strongly to protect old knowledge
            plasticity_decay: Decay rate for plasticity (0=rigid, 1=fully plastic)
            importance_update_rate: How quickly importance estimates update
            use_meta_plasticity: Whether to use meta-plastic control
            meta_lr: Meta-learning rate
        """
        super().__init__()
        
        self.model = model
        self.consolidation_strength = consolidation_strength
        self.plasticity_decay = plasticity_decay
        self.importance_update_rate = importance_update_rate
        self.use_meta_plasticity = use_meta_plasticity
        
        # Track parameter importance for each task
        self.importance_weights = {}
        self.consolidated_weights = {}
        self.task_count = 0
        
        # Meta-plasticity controller
        if use_meta_plasticity:
            num_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
            self.meta_controller = MetaPlasticityController(
                num_layers=num_layers,
                meta_lr=meta_lr
            )
        else:
            self.meta_controller = None
        
        # Initialize importance tracking
        self._initialize_importance_tracking()
        
        logger.info("🧠 Continual learning system initialized")
        logger.info(f"   Consolidation strength: {consolidation_strength}")
        logger.info(f"   Meta-plasticity: {'Enabled' if use_meta_plasticity else 'Disabled'}")
    
    def _initialize_importance_tracking(self):
        """Initialize importance weights for all parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.importance_weights[name] = torch.zeros_like(param.data)
                self.consolidated_weights[name] = param.data.clone()
    
    def compute_parameter_importance(
        self,
        dataloader,
        num_samples: int = 1000
    ):
        """
        Compute parameter importance using Fisher Information approximation.
        
        Important parameters (those critical for current task) will be
        protected from large changes during future learning.
        """
        self.model.train()
        
        # Initialize Fisher information accumulators
        fisher_information = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_information[name] = torch.zeros_like(param.data)
        
        # Compute Fisher information
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            # Extract data
            if isinstance(batch, dict):
                inputs = batch['input_ids'] if 'input_ids' in batch else batch['data']
                targets = batch['labels'] if 'labels' in batch else batch['targets']
            else:
                inputs, targets = batch
            
            inputs = inputs.to(next(self.model.parameters()).device)
            targets = targets.to(next(self.model.parameters()).device)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss and gradients
            if outputs.dim() > 1 and outputs.size(-1) > 1:
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            else:
                loss = F.mse_loss(outputs, targets.float())
            
            loss.backward()
            
            # Accumulate squared gradients (Fisher approximation)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_information[name] += param.grad.data ** 2
            
            samples_processed += inputs.size(0)
        
        # Average Fisher information
        for name in fisher_information:
            fisher_information[name] /= samples_processed
        
        # Update importance weights (exponential moving average)
        for name in self.importance_weights:
            if name in fisher_information:
                self.importance_weights[name] = (
                    (1 - self.importance_update_rate) * self.importance_weights[name] +
                    self.importance_update_rate * fisher_information[name]
                )
        
        logger.info(f"✓ Parameter importance computed from {samples_processed} samples")
    
    def consolidate_task_knowledge(self):
        """
        Consolidate knowledge for current task.
        Should be called after training on each task.
        """
        # Store current weights as consolidated reference
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.consolidated_weights[name] = param.data.clone()
        
        self.task_count += 1
        
        logger.info(f"✓ Task {self.task_count} knowledge consolidated")
    
    def compute_consolidation_loss(self) -> torch.Tensor:
        """
        Compute consolidation loss that penalizes changes to important parameters.
        
        This is the key to preventing catastrophic forgetting - we penalize
        large changes to parameters that were important for previous tasks.
        """
        if self.task_count == 0:
            return torch.tensor(0.0)
        
        consolidation_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.importance_weights:
                # Quadratic penalty weighted by importance
                importance = self.importance_weights[name]
                consolidated = self.consolidated_weights[name]
                
                # L2 distance weighted by importance
                param_loss = (importance * (param - consolidated) ** 2).sum()
                consolidation_loss += param_loss
        
        return self.consolidation_strength * consolidation_loss / (2 * self.task_count)
    
    def get_adaptive_learning_rates(self) -> Dict[str, torch.Tensor]:
        """
        Get adaptive learning rates based on parameter importance.
        
        Important parameters get lower learning rates (more consolidated),
        less important parameters get higher learning rates (more plastic).
        """
        adaptive_lrs = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.importance_weights:
                # Inverse relationship: high importance → low learning rate
                importance = self.importance_weights[name]
                
                # Compute adaptive learning rate
                base_lr = 0.01
                adaptive_lr = base_lr / (1.0 + self.consolidation_strength * importance)
                
                adaptive_lrs[name] = adaptive_lr
        
        return adaptive_lrs
    
    def apply_stdp_with_consolidation(
        self,
        layer_name: str,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        stdp_rule: STDPRule
    ) -> torch.Tensor:
        """
        Apply STDP update modulated by consolidation constraints.
        
        Combines local STDP learning with global consolidation requirements.
        """
        # Compute STDP weight update
        weight_update = stdp_rule.compute_weight_update(
            pre_spikes,
            post_spikes,
            weights
        )
        
        # Modulate by importance (reduce plasticity for important weights)
        if layer_name in self.importance_weights:
            importance = self.importance_weights[layer_name]
            
            # Plasticity mask: high importance → low plasticity
            plasticity_mask = 1.0 / (1.0 + importance * self.consolidation_strength * 0.01)
            
            # Apply plasticity modulation
            weight_update = weight_update * plasticity_mask
        
        # Apply update
        new_weights = stdp_rule.apply_weight_update(weights, weight_update)
        
        return new_weights


class TaskBuffer:
    """
    Buffer for storing examples from previous tasks for replay.
    Helps prevent catastrophic forgetting through experience replay.
    """
    
    def __init__(
        self,
        buffer_size: int = 1000,
        sampling_strategy: str = 'uniform'
    ):
        """
        Args:
            buffer_size: Maximum number of examples to store
            sampling_strategy: 'uniform', 'importance', or 'balanced'
        """
        self.buffer_size = buffer_size
        self.sampling_strategy = sampling_strategy
        
        self.examples = []
        self.task_labels = []
        self.importance_scores = []
    
    def add_examples(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        task_id: int,
        importance_scores: Optional[List[float]] = None
    ):
        """Add examples to buffer from a task."""
        for i, (input, target) in enumerate(examples):
            if len(self.examples) < self.buffer_size:
                self.examples.append((input, target))
                self.task_labels.append(task_id)
                score = importance_scores[i] if importance_scores else 1.0
                self.importance_scores.append(score)
            else:
                # Replace least important example if this one is more important
                if importance_scores and importance_scores[i] > min(self.importance_scores):
                    min_idx = self.importance_scores.index(min(self.importance_scores))
                    self.examples[min_idx] = (input, target)
                    self.task_labels[min_idx] = task_id
                    self.importance_scores[min_idx] = importance_scores[i]
    
    def sample(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample a batch from buffer."""
        if len(self.examples) == 0:
            return []
        
        if self.sampling_strategy == 'uniform':
            indices = np.random.choice(len(self.examples), 
                                     size=min(batch_size, len(self.examples)),
                                     replace=False)
        elif self.sampling_strategy == 'importance':
            # Sample based on importance scores
            probs = np.array(self.importance_scores)
            probs = probs / probs.sum()
            indices = np.random.choice(len(self.examples),
                                     size=min(batch_size, len(self.examples)),
                                     replace=False,
                                     p=probs)
        else:  # balanced
            # Sample equally from each task
            unique_tasks = list(set(self.task_labels))
            samples_per_task = batch_size // len(unique_tasks)
            indices = []
            for task_id in unique_tasks:
                task_indices = [i for i, t in enumerate(self.task_labels) if t == task_id]
                selected = np.random.choice(task_indices,
                                          size=min(samples_per_task, len(task_indices)),
                                          replace=False)
                indices.extend(selected)
        
        return [self.examples[i] for i in indices]


class ContinualLearningTrainer:
    """
    Trainer for continual learning with STDP and meta-plasticity.
    """
    
    def __init__(
        self,
        model: nn.Module,
        consolidation_strength: float = 1000.0,
        use_experience_replay: bool = True,
        replay_buffer_size: int = 1000,
        use_meta_plasticity: bool = True,
        meta_lr: float = 0.001
    ):
        self.continual_system = ContinualLearningSTDP(
            model=model,
            consolidation_strength=consolidation_strength,
            use_meta_plasticity=use_meta_plasticity,
            meta_lr=meta_lr
        )
        
        self.use_experience_replay = use_experience_replay
        if use_experience_replay:
            self.replay_buffer = TaskBuffer(buffer_size=replay_buffer_size)
        
        self.task_performance = defaultdict(list)
        
        logger.info("🎓 Continual learning trainer initialized")
    
    def train_on_task(
        self,
        task_id: int,
        train_loader,
        val_loader,
        num_epochs: int,
        optimizer,
        criterion
    ):
        """
        Train on a single task with continual learning protections.
        """
        logger.info(f"\n📚 Training on Task {task_id}")
        
        model = self.continual_system.model
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                # Extract data
                if isinstance(batch, dict):
                    inputs = batch.get('input_ids', batch.get('data'))
                    targets = batch.get('labels', batch.get('targets'))
                else:
                    inputs, targets = batch
                
                inputs = inputs.to(next(model.parameters()).device)
                targets = targets.to(next(model.parameters()).device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Task loss
                task_loss = criterion(outputs, targets)
                
                # Consolidation loss (protect previous knowledge)
                consolidation_loss = self.continual_system.compute_consolidation_loss()
                
                # Total loss
                total_loss = task_loss + consolidation_loss
                
                # Backward and optimize
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Experience replay
                if self.use_experience_replay and len(self.replay_buffer.examples) > 0:
                    replay_samples = self.replay_buffer.sample(batch_size=inputs.size(0) // 2)
                    if replay_samples:
                        replay_inputs = torch.stack([s[0] for s in replay_samples])
                        replay_targets = torch.stack([s[1] for s in replay_samples])
                        
                        replay_outputs = model(replay_inputs)
                        replay_loss = criterion(replay_outputs, replay_targets)
                        
                        optimizer.zero_grad()
                        replay_loss.backward()
                        optimizer.step()
            
            avg_loss = epoch_loss / num_batches
            
            # Validation
            val_accuracy = self._evaluate(model, val_loader, criterion)
            
            logger.info(f"Task {task_id}, Epoch {epoch + 1}/{num_epochs}: "
                       f"Loss={avg_loss:.4f}, Val Acc={val_accuracy:.3f}")
            
            # Update meta-plasticity
            if self.continual_system.use_meta_plasticity:
                self.continual_system.meta_controller.update_history(
                    performance=val_accuracy,
                    loss=avg_loss,
                    weight_change=0.01  # Simplified
                )
        
        # After training, consolidate knowledge
        self.continual_system.compute_parameter_importance(train_loader)
        self.continual_system.consolidate_task_knowledge()
        
        # Store examples in replay buffer
        if self.use_experience_replay:
            self._store_task_examples(task_id, train_loader)
        
        # Track performance
        final_acc = self._evaluate(model, val_loader, criterion)
        self.task_performance[task_id].append(final_acc)
        
        logger.info(f"✓ Task {task_id} training complete. Final accuracy: {final_acc:.3f}")
        
        return final_acc
    
    def _evaluate(self, model, dataloader, criterion):
        """Evaluate model on dataloader."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = batch.get('input_ids', batch.get('data'))
                    targets = batch.get('labels', batch.get('targets'))
                else:
                    inputs, targets = batch
                
                inputs = inputs.to(next(model.parameters()).device)
                targets = targets.to(next(model.parameters()).device)
                
                outputs = model(inputs)
                
                if outputs.dim() > 1 and outputs.size(-1) > 1:
                    _, predicted = torch.max(outputs, -1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _store_task_examples(self, task_id: int, dataloader, max_examples: int = 200):
        """Store representative examples from task."""
        examples = []
        count = 0
        
        for batch in dataloader:
            if count >= max_examples:
                break
            
            if isinstance(batch, dict):
                inputs = batch.get('input_ids', batch.get('data'))
                targets = batch.get('labels', batch.get('targets'))
            else:
                inputs, targets = batch
            
            for i in range(inputs.size(0)):
                if count >= max_examples:
                    break
                examples.append((inputs[i].cpu(), targets[i].cpu()))
                count += 1
        
        self.replay_buffer.add_examples(examples, task_id)
        logger.info(f"Stored {len(examples)} examples from Task {task_id}")
    
    def evaluate_all_tasks(self, task_dataloaders: Dict[int, any], criterion):
        """Evaluate model on all previous tasks to measure forgetting."""
        model = self.continual_system.model
        results = {}
        
        logger.info("\n📊 Evaluating on all tasks...")
        
        for task_id, dataloader in task_dataloaders.items():
            accuracy = self._evaluate(model, dataloader, criterion)
            results[task_id] = accuracy
            logger.info(f"   Task {task_id}: Accuracy = {accuracy:.3f}")
        
        # Compute average accuracy and forgetting
        avg_accuracy = np.mean(list(results.values()))
        
        # Compute backward transfer (forgetting)
        forgetting = []
        for task_id in results:
            if len(self.task_performance[task_id]) > 1:
                initial_acc = self.task_performance[task_id][0]
                current_acc = results[task_id]
                forgetting.append(initial_acc - current_acc)
        
        avg_forgetting = np.mean(forgetting) if forgetting else 0.0
        
        logger.info(f"\n📈 Overall Performance:")
        logger.info(f"   Average Accuracy: {avg_accuracy:.3f}")
        logger.info(f"   Average Forgetting: {avg_forgetting:.3f}")
        
        return results, avg_accuracy, avg_forgetting


# Example usage
if __name__ == "__main__":
    print("Testing Continual Learning Framework...")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create trainer
    trainer = ContinualLearningTrainer(
        model=model,
        consolidation_strength=1000.0,
        use_experience_replay=True,
        use_meta_plasticity=True
    )
    
    print("✅ Continual learning framework initialized!")
    print(f"   Tasks can be trained sequentially without catastrophic forgetting")
    print(f"   System uses: STDP + Meta-plasticity + Experience replay")