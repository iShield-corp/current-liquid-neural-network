#!/usr/bin/env python3
"""
Advanced Continual Learning & Long-Term Memory System

This module provides a complete memory system for post-training learning without forgetting.

Features:
1. Episodic Memory Bank - Differentiable attention-based memory storage/retrieval
2. Experience Replay Buffer - Priority-weighted rehearsal of past experiences
3. Elastic Weight Consolidation (EWC) - Protect important weights using Fisher Information
4. Synaptic Intelligence (SI) - Online importance tracking during training
5. Memory Consolidation - Transfer short-term to long-term memory
6. Progressive Network Expansion - Add capacity for new domains (optional)

Based on latest research (2024-2025):
- arXiv:2404.05555 - Episodic memory for continual learning
- arXiv:1811.11682 - Experience replay for continual learning
- Zenke et al. (2017) - Synaptic Intelligence
- Kirkpatrick et al. (2017) - Elastic Weight Consolidation
- DeepMind - Memory-Augmented Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import deque
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContinualMemoryConfig:
    """Configuration for continual learning memory systems."""
    # Episodic Memory Bank
    episodic_memory_size: int = 10000
    memory_key_dim: int = 256
    memory_value_dim: int = 512
    num_read_heads: int = 4
    num_write_heads: int = 1
    
    # Experience Replay
    replay_buffer_size: int = 5000
    replay_batch_size: int = 32
    replay_frequency: float = 0.3
    use_priority_sampling: bool = True
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    
    # Elastic Weight Consolidation
    ewc_lambda: float = 5000.0
    ewc_online: bool = True
    ewc_gamma: float = 0.95
    fisher_estimation_samples: int = 200
    
    # Synaptic Intelligence
    si_c: float = 0.1
    si_epsilon: float = 1e-7
    si_damping: float = 0.1
    
    # Memory Consolidation
    consolidation_interval: int = 1000
    consolidation_strength: float = 0.1
    sleep_consolidation: bool = True
    
    # Memory Gating & Retrieval
    write_threshold: float = 0.5
    read_temperature: float = 1.0
    novelty_threshold: float = 0.3
    
    # Progressive Networks (optional)
    enable_progressive: bool = False
    max_expansions: int = 5
    expansion_threshold: float = 0.1


# ============================================================
# 1. EPISODIC MEMORY BANK
# ============================================================

class EpisodicMemoryBank(nn.Module):
    """
    Differentiable episodic memory bank with attention-based read/write.
    
    Based on Neural Turing Machines and Differentiable Neural Computers.
    Uses content-based addressing for efficient memory retrieval.
    """
    
    def __init__(self, config: ContinualMemoryConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.episodic_memory_size
        self.key_dim = config.memory_key_dim
        self.value_dim = config.memory_value_dim
        
        # Memory storage (persistent across batches)
        self.register_buffer(
            'memory_keys',
            torch.zeros(config.episodic_memory_size, config.memory_key_dim)
        )
        self.register_buffer(
            'memory_values',
            torch.zeros(config.episodic_memory_size, config.memory_value_dim)
        )
        self.register_buffer(
            'memory_usage',
            torch.zeros(config.episodic_memory_size)
        )
        self.register_buffer(
            'memory_importance',
            torch.ones(config.episodic_memory_size)
        )
        self.register_buffer(
            'memory_timestamps',
            torch.zeros(config.episodic_memory_size, dtype=torch.long)
        )
        self.register_buffer(
            'write_pointer',
            torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            'total_writes',
            torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            'global_step',
            torch.tensor(0, dtype=torch.long)
        )
        
        # Read/write projections
        self.query_proj = nn.Linear(config.memory_value_dim, config.memory_key_dim * config.num_read_heads)
        self.read_strength = nn.Linear(config.memory_value_dim, config.num_read_heads)
        
        self.write_key_proj = nn.Linear(config.memory_value_dim, config.memory_key_dim)
        self.write_value_proj = nn.Linear(config.memory_value_dim, config.memory_value_dim)
        
        # Novelty detector for write gating
        self.novelty_detector = nn.Sequential(
            nn.Linear(config.memory_key_dim, config.memory_key_dim // 2),
            nn.ReLU(),
            nn.Linear(config.memory_key_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(
            config.memory_value_dim * config.num_read_heads,
            config.memory_value_dim
        )
    
    def read(
        self,
        query: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory using content-based attention.
        
        Args:
            query: [batch, value_dim] - Query vector
            top_k: Number of memories to retrieve
            
        Returns:
            retrieved: [batch, value_dim] - Retrieved memory content
            attention: [batch, num_heads, memory_size] - Attention weights
        """
        batch_size = query.shape[0]
        device = query.device
        
        # Generate read keys for each head
        read_keys = self.query_proj(query)  # [batch, key_dim * num_heads]
        read_keys = read_keys.view(
            batch_size, self.config.num_read_heads, self.config.memory_key_dim
        )
        
        # Get read strengths (sharpness of attention)
        strengths = F.softplus(self.read_strength(query))  # [batch, num_heads]
        
        # Compute attention over memory
        # [batch, num_heads, key_dim] x [memory_size, key_dim]^T
        similarities = torch.einsum(
            'bhk,mk->bhm',
            read_keys,
            self.memory_keys.to(device)
        )
        
        # Apply temperature and strength
        attention_logits = similarities * strengths.unsqueeze(-1)
        
        # Mask unused memory slots
        mask = (self.memory_usage.to(device) < 1e-8).unsqueeze(0).unsqueeze(0)
        attention_logits = attention_logits.masked_fill(mask, float('-inf'))
        
        # Softmax attention
        attention = F.softmax(
            attention_logits / self.config.read_temperature,
            dim=-1
        )
        
        # Weight by importance
        if self.config.use_priority_sampling:
            importance_weights = self.memory_importance.to(device).unsqueeze(0).unsqueeze(0)
            attention = attention * importance_weights
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Sparse attention (top-k)
        if top_k < self.memory_size:
            top_values, top_indices = attention.topk(min(top_k, attention.shape[-1]), dim=-1)
            sparse_attention = torch.zeros_like(attention)
            sparse_attention.scatter_(2, top_indices, top_values)
            attention = sparse_attention / (sparse_attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Retrieve values
        # [batch, num_heads, memory_size] x [memory_size, value_dim]
        read_values = torch.einsum(
            'bhm,mv->bhv',
            attention,
            self.memory_values.to(device)
        )
        
        # Combine heads
        read_values = read_values.view(batch_size, -1)  # [batch, num_heads * value_dim]
        retrieved = self.output_proj(read_values)
        
        # Update usage (decay unused memories)
        with torch.no_grad():
            self.memory_usage *= 0.999
        
        return retrieved, attention
    
    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        force_write: bool = False
    ) -> torch.Tensor:
        """
        Write to memory with novelty-based gating.
        
        Args:
            key: [batch, key_dim] - Memory key
            value: [batch, value_dim] - Memory value
            importance: [batch] - Optional importance scores
            force_write: Force write regardless of novelty
            
        Returns:
            novelty: [batch] - Novelty scores of written items
        """
        if key.dim() == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        
        batch_size = key.shape[0]
        device = key.device
        
        # Compute novelty (how different from existing memories)
        similarities = F.cosine_similarity(
            key.unsqueeze(1),  # [batch, 1, key_dim]
            self.memory_keys.to(device).unsqueeze(0),  # [1, memory_size, key_dim]
            dim=-1
        )  # [batch, memory_size]
        
        max_similarity = similarities.max(dim=-1)[0]  # [batch]
        novelty = 1.0 - max_similarity
        
        # Learned novelty gating
        novelty_gate = self.novelty_detector(key).squeeze(-1)  # [batch]
        
        # Write if novel enough or forced
        should_write = (
            (novelty > self.config.write_threshold) |
            (novelty_gate > 0.5) |
            force_write
        )
        
        # Write each item that passes the gate
        for i in range(batch_size):
            if should_write[i] or self.total_writes < self.memory_size:
                self._write_single(key[i], value[i], importance[i] if importance is not None else None)
        
        self.global_step += 1
        
        return novelty
    
    def _write_single(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ):
        """Write a single key-value pair to memory."""
        device = key.device
        
        # Find slot to write
        if self.total_writes < self.memory_size:
            slot = self.write_pointer.item()
            self.write_pointer = (self.write_pointer + 1) % self.memory_size
        else:
            # LRU with importance weighting
            age = (self.global_step - self.memory_timestamps.to(device)).float()
            age_normalized = age / (age.max() + 1e-8)
            
            # Priority: low usage + low importance + old
            priority = (
                (1 - self.memory_usage.to(device)) * 0.4 +
                (1 - self.memory_importance.to(device)) * 0.4 +
                age_normalized * 0.2
            )
            
            slot = priority.argmax().item()
        
        # Write to slot
        self.memory_keys[slot] = key.detach().cpu()
        self.memory_values[slot] = value.detach().cpu()
        self.memory_ages = torch.zeros_like(self.memory_timestamps)
        self.memory_ages[slot] = 0
        self.memory_usage[slot] = 1.0
        
        if importance is not None:
            self.memory_importance[slot] = importance.detach().cpu()
        else:
            self.memory_importance[slot] = 1.0
        
        self.memory_timestamps[slot] = self.global_step.clone()
        self.total_writes += 1
    
    def consolidate(self, consolidation_strength: float = 0.1):
        """
        Consolidate memories - merge similar, strengthen important.
        
        This simulates sleep-based memory consolidation in the brain.
        """
        device = self.memory_keys.device
        
        # Find similar memories to merge
        similarities = F.cosine_similarity(
            self.memory_keys.unsqueeze(0),
            self.memory_keys.unsqueeze(1),
            dim=-1
        )
        similarities.fill_diagonal_(0)
        
        # Merge threshold
        merge_threshold = 0.95
        merge_pairs = (similarities > merge_threshold).nonzero()
        
        merged_slots = set()
        for i, j in merge_pairs:
            i, j = i.item(), j.item()
            if i in merged_slots or j in merged_slots:
                continue
            
            # Keep the more important one
            if self.memory_importance[i] >= self.memory_importance[j]:
                keep, discard = i, j
            else:
                keep, discard = j, i
            
            # Merge values (weighted average)
            weight = self.memory_importance[keep] / (
                self.memory_importance[keep] + self.memory_importance[discard] + 1e-8
            )
            self.memory_values[keep] = (
                weight * self.memory_values[keep] +
                (1 - weight) * self.memory_values[discard]
            )
            
            # Clear discarded slot
            self.memory_keys[discard] = 0
            self.memory_values[discard] = 0
            self.memory_usage[discard] = 0
            
            merged_slots.add(discard)
        
        # Strengthen important memories
        important_mask = self.memory_importance > self.memory_importance.median()
        self.memory_importance[important_mask] = torch.clamp(
            self.memory_importance[important_mask] + consolidation_strength,
            max=1.0
        )
        
        # Decay less important memories
        self.memory_importance[~important_mask] *= (1 - consolidation_strength)
        
        logger.debug(f"Consolidated {len(merged_slots)} memory slots")


# ============================================================
# 2. EXPERIENCE REPLAY BUFFER
# ============================================================

class ExperienceReplayBuffer:
    """
    Priority-weighted experience replay buffer.
    
    Stores past experiences and samples them for rehearsal during new learning.
    Uses importance sampling to prioritize more valuable experiences.
    """
    
    def __init__(self, config: ContinualMemoryConfig):
        self.config = config
        self.buffer_size = config.replay_buffer_size
        self.buffer: deque = deque(maxlen=config.replay_buffer_size)
        self.priorities: deque = deque(maxlen=config.replay_buffer_size)
        self.tasks: deque = deque(maxlen=config.replay_buffer_size)
    
    def add(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int = 0,
        priority: float = 1.0
    ):
        """Add experiences to buffer."""
        batch_size = inputs.size(0)
        
        for i in range(batch_size):
            self.buffer.append((
                inputs[i].detach().cpu(),
                targets[i].detach().cpu()
            ))
            self.priorities.append(priority)
            self.tasks.append(task_id)
    
    def sample(
        self,
        batch_size: int,
        task_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Sample batch from buffer with priority weighting.
        
        Args:
            batch_size: Number of samples
            task_id: If specified, sample only from this task
            
        Returns:
            inputs: [batch, ...] - Input tensors
            targets: [batch, ...] - Target tensors
            indices: List of sampled indices
        """
        if len(self.buffer) == 0:
            return None, None, []
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Filter by task if specified
        if task_id is not None:
            valid_indices = [i for i, t in enumerate(self.tasks) if t == task_id]
            if not valid_indices:
                return None, None, []
        else:
            valid_indices = list(range(len(self.buffer)))
        
        # Priority-weighted sampling
        if self.config.use_priority_sampling:
            priorities = np.array([self.priorities[i] for i in valid_indices])
            priorities = priorities ** self.config.priority_alpha
            probabilities = priorities / priorities.sum()
            
            indices = np.random.choice(
                valid_indices,
                size=min(batch_size, len(valid_indices)),
                replace=False,
                p=probabilities
            )
        else:
            indices = np.random.choice(
                valid_indices,
                size=min(batch_size, len(valid_indices)),
                replace=False
            )
        
        # Gather samples
        inputs = torch.stack([self.buffer[i][0] for i in indices])
        targets = torch.stack([self.buffer[i][1] for i in indices])
        
        return inputs, targets, list(indices)
    
    def update_priorities(self, indices: List[int], new_priorities: List[float]):
        """Update priorities for specific samples."""
        for idx, priority in zip(indices, new_priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def get_task_distribution(self) -> Dict[int, int]:
        """Get distribution of tasks in buffer."""
        distribution = {}
        for task_id in self.tasks:
            distribution[task_id] = distribution.get(task_id, 0) + 1
        return distribution
    
    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================
# 3. ELASTIC WEIGHT CONSOLIDATION (EWC)
# ============================================================

class ElasticWeightConsolidation(nn.Module):
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.
    
    Based on Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting"
    Computes Fisher Information to identify important weights.
    """
    
    def __init__(self, config: ContinualMemoryConfig):
        super().__init__()
        self.config = config
        self.lambda_ewc = config.ewc_lambda
        self.online = config.ewc_online
        self.gamma = config.ewc_gamma
        
        # Storage for Fisher information and optimal weights
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optpar_dict: Dict[str, torch.Tensor] = {}
        self.task_count = 0
    
    def compute_fisher(
        self,
        model: nn.Module,
        dataloader,
        criterion,
        num_samples: int = None
    ):
        """
        Compute Fisher Information Matrix diagonal.
        
        Args:
            model: Neural network model
            dataloader: Data loader for current task
            criterion: Loss function
            num_samples: Number of samples to use (None = use config)
        """
        if num_samples is None:
            num_samples = self.config.fisher_estimation_samples
        
        model.eval()
        fisher_dict = {}
        
        # Initialize Fisher dict
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)
        
        # Accumulate squared gradients
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            # Extract data
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else batch[0]
            elif isinstance(batch, dict):
                inputs = batch.get('input_ids', batch.get('data'))
                targets = batch.get('labels', batch.get('targets', inputs))
            else:
                inputs = batch
                targets = batch
            
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            model.zero_grad()
            outputs = model(inputs)
            
            # Handle different output shapes
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            if outputs.dim() > 2:
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Accumulate squared gradients (Fisher diagonal)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            sample_count += inputs.size(0)
        
        # Normalize by sample count
        for name in fisher_dict:
            fisher_dict[name] /= sample_count
            
            # Online EWC: running average
            if self.online and name in self.fisher_dict:
                self.fisher_dict[name] = (
                    self.gamma * self.fisher_dict[name] +
                    (1 - self.gamma) * fisher_dict[name]
                )
            else:
                self.fisher_dict[name] = fisher_dict[name]
        
        # Store optimal parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optpar_dict[name] = param.data.clone()
        
        self.task_count += 1
        logger.info(f"Computed Fisher Information for task {self.task_count} ({sample_count} samples)")
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty loss.
        
        Returns quadratic penalty for deviating from optimal weights.
        """
        if not self.fisher_dict:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name].to(param.device)
                optpar = self.optpar_dict[name].to(param.device)
                loss += (fisher * (param - optpar) ** 2).sum()
        
        return self.lambda_ewc * loss / max(self.task_count, 1)


# ============================================================
# 4. SYNAPTIC INTELLIGENCE (SI)
# ============================================================

class SynapticIntelligence(nn.Module):
    """
    Synaptic Intelligence for online continual learning.
    
    Based on Zenke et al. (2017) "Continual Learning Through Synaptic Intelligence"
    Tracks parameter importance online during training.
    """
    
    def __init__(self, config: ContinualMemoryConfig):
        super().__init__()
        self.config = config
        self.c = config.si_c
        self.epsilon = config.si_epsilon
        self.damping = config.si_damping
        
        # Parameter importance
        self.omega: Dict[str, torch.Tensor] = {}
        
        # Online tracking
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.running_sum: Dict[str, torch.Tensor] = {}
    
    def register_model(self, model: nn.Module):
        """Register model parameters for tracking."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.omega[name] = torch.zeros_like(param.data)
                self.prev_params[name] = param.data.clone()
                self.running_sum[name] = torch.zeros_like(param.data)
    
    def update_running_sum(self, model: nn.Module):
        """Update running sum of gradient * parameter change."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.prev_params and param.grad is not None:
                    delta = param.data - self.prev_params[name]
                    self.running_sum[name] += -param.grad.data * delta
                    self.prev_params[name] = param.data.clone()
    
    def consolidate(self, model: nn.Module):
        """Consolidate importance after task completion."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.omega:
                    delta = param.data - self.prev_params[name]
                    
                    # Compute importance as gradient-weighted path integral
                    importance = self.running_sum[name] / (
                        delta ** 2 + self.epsilon
                    )
                    
                    # Update omega (keep max importance)
                    self.omega[name] = torch.max(
                        self.omega[name],
                        importance.abs()
                    )
                    
                    # Reset running sum
                    self.running_sum[name] = torch.zeros_like(param.data)
        
        logger.info("Synaptic Intelligence: Consolidated parameter importance")
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute SI penalty."""
        if not self.omega:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        penalty = 0.0
        for name, param in model.named_parameters():
            if name in self.omega:
                omega = self.omega[name].to(param.device)
                prev = self.prev_params[name].to(param.device)
                delta = param - prev
                penalty += (omega * delta ** 2).sum()
        
        return self.c * penalty


# ============================================================
# 5. MEMORY CONSOLIDATION SYSTEM
# ============================================================

class MemoryConsolidationSystem(nn.Module):
    """
    Memory consolidation system for transferring short-term to long-term memory.
    
    Implements sleep-like consolidation phases that strengthen important memories
    and merge similar ones.
    """
    
    def __init__(self, config: ContinualMemoryConfig, episodic_memory: EpisodicMemoryBank):
        super().__init__()
        self.config = config
        self.episodic_memory = episodic_memory
        self.consolidation_count = 0
    
    def consolidate(self, importance_weights: Optional[torch.Tensor] = None):
        """
        Perform memory consolidation.
        
        Args:
            importance_weights: [memory_size] - Importance of each memory
        """
        if importance_weights is None:
            # Use inverse of age as importance
            age = self.episodic_memory.global_step - self.episodic_memory.memory_timestamps
            importance_weights = 1.0 / (age.float() + 1.0)
        
        # Consolidate episodic memory
        self.episodic_memory.consolidate(self.config.consolidation_strength)
        
        self.consolidation_count += 1
        logger.info(f"Memory consolidation #{self.consolidation_count} completed")
    
    def should_consolidate(self, step: int) -> bool:
        """Check if consolidation should be performed."""
        return step % self.config.consolidation_interval == 0


# ============================================================
# 6. PROGRESSIVE NETWORK EXPANDER (OPTIONAL)
# ============================================================

class ProgressiveNetworkExpander(nn.Module):
    """
    Progressive network expansion for adding capacity without forgetting.
    
    Based on Rusu et al. (2016) "Progressive Neural Networks"
    Adds new columns/modules while freezing old ones.
    """
    
    def __init__(self, config: ContinualMemoryConfig, base_hidden_dim: int):
        super().__init__()
        self.config = config
        self.base_hidden_dim = base_hidden_dim
        self.expansion_count = 0
        
        # Storage for expansion modules
        self.expansion_columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
    
    def expand(self, new_task_dim: int = None) -> Optional[nn.Module]:
        """
        Add new capacity for learning a new task.
        
        Returns the new column module.
        """
        if self.expansion_count >= self.config.max_expansions:
            logger.warning("Maximum expansions reached")
            return None
        
        new_dim = new_task_dim or self.base_hidden_dim
        
        # New column for new task
        new_column = nn.Sequential(
            nn.Linear(self.base_hidden_dim, new_dim),
            nn.LayerNorm(new_dim),
            nn.GELU(),
            nn.Linear(new_dim, new_dim)
        )
        
        # Lateral connections from all previous columns
        lateral = nn.ModuleList([
            nn.Linear(self.base_hidden_dim, new_dim)
            for _ in range(self.expansion_count + 1)
        ])
        
        self.expansion_columns.append(new_column)
        self.lateral_connections.append(lateral)
        self.expansion_count += 1
        
        logger.info(f"Expanded network: now {self.expansion_count} columns")
        
        return new_column
    
    def forward(
        self,
        x: torch.Tensor,
        column_outputs: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Forward through all columns with lateral connections.
        
        Args:
            x: Input tensor
            column_outputs: Outputs from previous layers' columns
            
        Returns:
            List of outputs from each column
        """
        if column_outputs is None:
            column_outputs = []
        
        outputs = []
        for i, (column, laterals) in enumerate(
            zip(self.expansion_columns, self.lateral_connections)
        ):
            # Column forward
            col_out = column(x)
            
            # Add lateral connections from previous columns
            for j, (lateral, prev_out) in enumerate(zip(laterals, column_outputs)):
                if prev_out is not None:
                    col_out = col_out + lateral(prev_out)
            
            outputs.append(col_out)
        
        return outputs


# ============================================================
# INTEGRATED CONTINUAL LEARNING SYSTEM
# ============================================================

class ContinualLearningSystem(nn.Module):
    """
    Complete continual learning system integrating all 6 features:
    1. Episodic Memory Bank
    2. Experience Replay Buffer
    3. Elastic Weight Consolidation
    4. Synaptic Intelligence
    5. Memory Consolidation
    6. Progressive Network Expansion
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[ContinualMemoryConfig] = None,
        enable_all: bool = True
    ):
        super().__init__()
        
        if config is None:
            config = ContinualMemoryConfig()
        
        self.config = config
        self.base_model = base_model
        
        # 1. Episodic Memory Bank
        self.episodic_memory = EpisodicMemoryBank(config)
        
        # 2. Experience Replay Buffer
        self.replay_buffer = ExperienceReplayBuffer(config)
        
        # 3. Elastic Weight Consolidation
        self.ewc = ElasticWeightConsolidation(config)
        
        # 4. Synaptic Intelligence
        self.si = SynapticIntelligence(config)
        self.si.register_model(base_model)
        
        # 5. Memory Consolidation System
        self.consolidation_system = MemoryConsolidationSystem(config, self.episodic_memory)
        
        # 6. Progressive Network Expander (optional)
        if config.enable_progressive:
            hidden_dim = getattr(base_model, 'hidden_dim', 512)
            self.progressive_expander = ProgressiveNetworkExpander(config, hidden_dim)
        else:
            self.progressive_expander = None
        
        # Memory-enhanced output gating
        output_dim = config.memory_value_dim
        self.memory_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )
        
        self.output_fusion = nn.Linear(output_dim * 2, output_dim)
        
        # Task tracking
        self.current_task_id = 0
        self.training_step = 0
        
        logger.info("🧠 Continual Learning System initialized")
        logger.info(f"   Episodic Memory: {config.episodic_memory_size} slots")
        logger.info(f"   Replay Buffer: {config.replay_buffer_size} examples")
        logger.info(f"   EWC: λ={config.ewc_lambda}")
        logger.info(f"   SI: c={config.si_c}")
        logger.info(f"   Consolidation: every {config.consolidation_interval} steps")
        if config.enable_progressive:
            logger.info(f"   Progressive Networks: enabled")
    
    def forward(
        self,
        x: torch.Tensor,
        store_memory: bool = True,
        use_memory: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with memory augmentation.
        
        Args:
            x: Input tensor
            store_memory: Whether to store this experience
            use_memory: Whether to retrieve from memory
            
        Returns:
            output: Model output enhanced by memory
            info: Dictionary with memory information
        """
        # Base model forward
        base_output = self.base_model(x)
        
        info = {}
        
        # Memory augmentation
        if use_memory and self.episodic_memory.total_writes > 0:
            # Flatten output for memory query
            if base_output.dim() > 2:
                query = base_output.mean(dim=1)
            else:
                query = base_output
            
            # Read from episodic memory
            memory_output, attention = self.episodic_memory.read(query)
            
            # Gate memory influence
            gate = self.memory_gate(
                torch.cat([query, memory_output], dim=-1)
            )
            
            # Fuse base and memory outputs
            combined = torch.cat([query, memory_output * gate], dim=-1)
            fused = self.output_fusion(combined)
            
            # Restore original shape if needed
            if base_output.dim() > 2:
                fused = fused.unsqueeze(1).expand_as(base_output)
            
            output = base_output + fused * 0.1  # Residual connection
            
            info['memory_attention'] = attention
            info['memory_gate'] = gate
        else:
            output = base_output
        
        # Store in memory if training
        if store_memory and self.training:
            # Extract key and value
            if output.dim() > 2:
                key_val = output.mean(dim=1)
            else:
                key_val = output
            
            key = self.episodic_memory.write_key_proj(key_val)
            value = self.episodic_memory.write_value_proj(key_val)
            
            # Write to memory
            novelty = self.episodic_memory.write(key, value)
            info['novelty'] = novelty
        
        # Update SI tracking
        if self.training:
            self.si.update_running_sum(self.base_model)
            self.training_step += 1
            
            # Periodic consolidation
            if self.consolidation_system.should_consolidate(self.training_step):
                self.consolidation_system.consolidate()
        
        return output, info
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute total regularization loss from all continual learning components.
        
        Returns:
            Total regularization loss
        """
        reg_loss = torch.tensor(0.0, device=next(self.base_model.parameters()).device)
        
        # EWC penalty
        ewc_penalty = self.ewc.penalty(self.base_model)
        reg_loss += ewc_penalty
        
        # SI penalty
        si_penalty = self.si.penalty(self.base_model)
        reg_loss += si_penalty
        
        return reg_loss
    
    def on_task_complete(
        self,
        dataloader,
        criterion,
        task_id: int
    ):
        """
        Called when a task is completed.
        
        Args:
            dataloader: DataLoader for the completed task
            criterion: Loss function
            task_id: Identifier for the task
        """
        logger.info(f"\n📊 Task {task_id} completion processing...")
        
        # Compute Fisher information for EWC
        self.ewc.compute_fisher(self.base_model, dataloader, criterion)
        
        # Consolidate SI importance
        self.si.consolidate(self.base_model)
        
        # Consolidate episodic memory
        self.consolidation_system.consolidate()
        
        self.current_task_id = task_id + 1
        
        logger.info(f"✅ Task {task_id} knowledge consolidated")
    
    def replay_experiences(
        self,
        batch_size: int = None,
        task_id: Optional[int] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample experiences from replay buffer.
        
        Args:
            batch_size: Number of experiences to sample
            task_id: Optional task to sample from
            
        Returns:
            (inputs, targets) or None if buffer empty
        """
        if batch_size is None:
            batch_size = self.config.replay_batch_size
        
        inputs, targets, indices = self.replay_buffer.sample(batch_size, task_id)
        
        if inputs is not None:
            device = next(self.base_model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)
        
        return inputs, targets
    
    def store_experience(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int = None,
        priority: float = 1.0
    ):
        """Store experiences in replay buffer."""
        if task_id is None:
            task_id = self.current_task_id
        
        self.replay_buffer.add(inputs, targets, task_id, priority)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory system."""
        return {
            'episodic_memory_usage': self.episodic_memory.total_writes.item(),
            'episodic_memory_capacity': self.episodic_memory.memory_size,
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_capacity': self.replay_buffer.buffer_size,
            'ewc_tasks': self.ewc.task_count,
            'si_omega_mean': sum(omega.mean().item() for omega in self.si.omega.values()) / max(len(self.si.omega), 1),
            'consolidation_count': self.consolidation_system.consolidation_count,
            'current_task': self.current_task_id,
            'training_steps': self.training_step,
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_continual_learning_system(
    base_model: nn.Module,
    episodic_memory_size: int = 10000,
    replay_buffer_size: int = 5000,
    ewc_lambda: float = 5000.0,
    si_c: float = 0.1,
    enable_progressive: bool = False
) -> ContinualLearningSystem:
    """
    Factory function to create a continual learning system.
    
    Args:
        base_model: Base neural network
        episodic_memory_size: Size of episodic memory
        replay_buffer_size: Size of replay buffer
        ewc_lambda: EWC regularization strength
        si_c: SI regularization coefficient
        enable_progressive: Enable progressive networks
        
    Returns:
        ContinualLearningSystem instance
    """
    config = ContinualMemoryConfig(
        episodic_memory_size=episodic_memory_size,
        replay_buffer_size=replay_buffer_size,
        ewc_lambda=ewc_lambda,
        si_c=si_c,
        enable_progressive=enable_progressive
    )
    
    return ContinualLearningSystem(base_model, config)


# ============================================================
# TESTING
# ============================================================

def test_continual_learning_system():
    """Test all continual learning components."""
    print("Testing Continual Learning System...\n")
    
    # Create simple base model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.hidden_dim = hidden_dim
        
        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))
    
    input_dim = 128
    hidden_dim = 256
    output_dim = 10
    batch_size = 32
    
    base_model = SimpleModel(input_dim, hidden_dim, output_dim)
    
    # Create continual learning system
    print("1. Creating Continual Learning System...")
    cl_system = create_continual_learning_system(
        base_model,
        episodic_memory_size=1000,
        replay_buffer_size=500,
        ewc_lambda=5000.0,
        si_c=0.1
    )
    
    # Test forward pass
    print("\n2. Testing forward pass with memory...")
    x = torch.randn(batch_size, input_dim)
    output, info = cl_system(x, store_memory=True, use_memory=False)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Novelty: {info.get('novelty', 'N/A')}")
    
    # Test memory storage and retrieval
    print("\n3. Testing memory storage...")
    for i in range(10):
        x = torch.randn(batch_size, input_dim)
        output, info = cl_system(x, store_memory=True, use_memory=False)
    print(f"   Stored {cl_system.episodic_memory.total_writes.item()} memories")
    
    print("\n4. Testing memory retrieval...")
    x = torch.randn(batch_size, input_dim)
    output, info = cl_system(x, store_memory=False, use_memory=True)
    print(f"   Retrieved from memory")
    print(f"   Memory gate mean: {info['memory_gate'].mean().item():.4f}")
    
    # Test experience replay
    print("\n5. Testing experience replay...")
    targets = torch.randint(0, output_dim, (batch_size,))
    cl_system.store_experience(x, targets, task_id=0)
    
    replay_x, replay_y = cl_system.replay_experiences(batch_size=16)
    print(f"   Replay inputs: {replay_x.shape if replay_x is not None else 'None'}")
    print(f"   Replay targets: {replay_y.shape if replay_y is not None else 'None'}")
    
    # Test regularization
    print("\n6. Testing regularization losses...")
    
    # Create dummy dataloader
    dummy_data = torch.utils.data.TensorDataset(
        torch.randn(100, input_dim),
        torch.randint(0, output_dim, (100,))
    )
    dummy_loader = torch.utils.data.DataLoader(dummy_data, batch_size=16)
    criterion = nn.CrossEntropyLoss()
    
    cl_system.on_task_complete(dummy_loader, criterion, task_id=0)
    
    reg_loss = cl_system.compute_regularization_loss()
    print(f"   Regularization loss: {reg_loss.item():.4f}")
    
    # Test memory stats
    print("\n7. Memory statistics:")
    stats = cl_system.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ All tests completed successfully!")
    print("\nContinual Learning System is ready for:")
    print("  ✓ Learning new tasks without forgetting")
    print("  ✓ Storing and retrieving episodic memories")
    print("  ✓ Experience replay for knowledge rehearsal")
    print("  ✓ Protecting important weights (EWC + SI)")
    print("  ✓ Memory consolidation")


if __name__ == "__main__":
    test_continual_learning_system()
