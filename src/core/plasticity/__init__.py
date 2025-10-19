"""
Plasticity mechanisms for continual learning.

This module provides STDP (Spike-Timing-Dependent Plasticity), meta-plasticity,
and continual learning capabilities to prevent catastrophic forgetting.
"""

from .stdp_plasticity import (
    STDPRule,
    ClassicalSTDP,
    TripletSTDP,
    HomeostaticSTDP,
    BCMRule,
    STDPLayer,
    integrate_stdp_into_model
)

from .meta_plasticity import (
    MetaPlasticityController,
    AdaptiveSTDPRule,
    MetaPlasticLayer,
    MetaPlasticNetwork,
    integrate_meta_plasticity
)

from .continual_learning import (
    ContinualLearningSTDP,
    TaskBuffer,
    ContinualLearningTrainer
)

__all__ = [
    # STDP
    'STDPRule',
    'ClassicalSTDP',
    'TripletSTDP',
    'HomeostaticSTDP',
    'BCMRule',
    'STDPLayer',
    'integrate_stdp_into_model',
    
    # Meta-plasticity
    'MetaPlasticityController',
    'AdaptiveSTDPRule',
    'MetaPlasticLayer',
    'MetaPlasticNetwork',
    'integrate_meta_plasticity',
    
    # Continual learning
    'ContinualLearningSTDP',
    'TaskBuffer',
    'ContinualLearningTrainer'
]
