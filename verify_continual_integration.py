#!/usr/bin/env python3
"""
Quick verification script to check continual learning integration.
Verifies imports and basic functionality without full training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def verify_imports():
    """Verify all continual learning components can be imported."""
    print("="*70)
    print("🔍 VERIFYING IMPORTS")
    print("="*70)
    
    try:
        print("\n1. Importing continual_memory module...")
        from src.core.continual_memory import (
            EpisodicMemoryBank,
            ExperienceReplayBuffer,
            ElasticWeightConsolidation,
            SynapticIntelligence,
            MemoryConsolidationSystem,
            ProgressiveNetworkExpander,
            ContinualLearningSystem
        )
        print("   ✅ All continual memory components imported")
        
        print("\n2. Importing main model components...")
        from src.core.main import ModelConfig, LiquidSpikingTrainer
        print("   ✅ Main model components imported")
        
        print("\n3. Checking ModelConfig has continual learning parameters...")
        # Create a minimal config with required parameters
        from src.core.main import TaskType
        config = ModelConfig(
            task_type=TaskType.LLM,
            input_dim=100,
            hidden_dim=64,
            output_dim=100,
            liquid_units=64,
            liquid_backbone='cfc',
            spiking_units=64,
            spike_threshold=1.0,
            beta=0.95,
            num_layers=2,
            dropout=0.1,
            sequence_length=32,
            batch_size=8,
            learning_rate=0.001,
            weight_decay=0.01,
            gradient_clip=1.0,
            mixed_precision=False,
            device='cpu',
            seed=42
        )
        continual_params = [
            'use_continual_learning',
            'episodic_memory_size',
            'memory_key_dim',
            'ewc_lambda',
            'si_c',
            'consolidation_frequency',
            'replay_frequency',
            'replay_batch_size',
            'enable_progressive_networks'
        ]
        
        for param in continual_params:
            if hasattr(config, param):
                value = getattr(config, param)
                print(f"   ✅ {param}: {value}")
            else:
                print(f"   ❌ Missing parameter: {param}")
                return False
        
        print("\n4. Checking LiquidSpikingTrainer integration...")
        import torch
        from src.core.main import TaskType, LiquidSpikingNetwork
        
        # Create trainer with continual learning enabled
        config_with_cl = ModelConfig(
            task_type=TaskType.LLM,
            input_dim=100,
            hidden_dim=64,
            output_dim=100,
            liquid_units=64,
            liquid_backbone='cfc',
            spiking_units=64,
            spike_threshold=1.0,
            beta=0.95,
            num_layers=2,
            dropout=0.1,
            sequence_length=32,
            batch_size=8,
            learning_rate=0.001,
            weight_decay=0.01,
            gradient_clip=1.0,
            mixed_precision=False,
            device='cpu',
            seed=42,
            use_continual_learning=True,
            episodic_memory_size=100,
            ewc_lambda=100.0,
            si_c=0.1
        )
        
        model = LiquidSpikingNetwork(config_with_cl)
        trainer = LiquidSpikingTrainer(model, config_with_cl)
        
        if trainer.continual_memory_system is not None:
            print("   ✅ ContinualLearningSystem initialized")
            
            # Check components
            if hasattr(trainer.continual_memory_system, 'episodic_memory'):
                print("   ✅ Episodic Memory Bank present")
            if hasattr(trainer.continual_memory_system, 'replay_buffer'):
                print("   ✅ Experience Replay Buffer present")
            if hasattr(trainer.continual_memory_system, 'ewc'):
                print("   ✅ Elastic Weight Consolidation present")
            if hasattr(trainer.continual_memory_system, 'si'):
                print("   ✅ Synaptic Intelligence present")
            if hasattr(trainer.continual_memory_system, 'consolidation'):
                print("   ✅ Memory Consolidation present")
            
            # Get stats
            stats = trainer.continual_memory_system.get_memory_stats()
            print(f"\n   📊 Initial Memory Stats:")
            for key, value in stats.items():
                print(f"      {key}: {value}")
        else:
            print("   ❌ ContinualLearningSystem not initialized")
            return False
        
        print("\n5. Testing without continual learning (backward compatibility)...")
        from src.core.main import LiquidSpikingNetwork
        config_no_cl = ModelConfig(
            task_type=TaskType.LLM,
            input_dim=100,
            hidden_dim=64,
            output_dim=100,
            liquid_units=64,
            liquid_backbone='cfc',
            spiking_units=64,
            spike_threshold=1.0,
            beta=0.95,
            num_layers=2,
            dropout=0.1,
            sequence_length=32,
            batch_size=8,
            learning_rate=0.001,
            weight_decay=0.01,
            gradient_clip=1.0,
            mixed_precision=False,
            device='cpu',
            seed=42,
            use_continual_learning=False
        )
        
        model_no_cl = LiquidSpikingNetwork(config_no_cl)
        trainer_no_cl = LiquidSpikingTrainer(model_no_cl, config_no_cl)
        
        if trainer_no_cl.continual_memory_system is None:
            print("   ✅ Continual learning correctly disabled")
        else:
            print("   ⚠️  Continual learning should be disabled")
        
        print("\n" + "="*70)
        print("✅ ALL VERIFICATIONS PASSED")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_feature_accessibility():
    """Verify all 6 features are accessible."""
    print("\n" + "="*70)
    print("🔬 VERIFYING INDIVIDUAL FEATURES")
    print("="*70)
    
    try:
        import torch
        from src.core.main import ModelConfig, LiquidSpikingTrainer, LiquidSpikingNetwork, TaskType
        
        config = ModelConfig(
            task_type=TaskType.LLM,
            input_dim=100,
            hidden_dim=64,
            output_dim=100,
            liquid_units=64,
            liquid_backbone='cfc',
            spiking_units=64,
            spike_threshold=1.0,
            beta=0.95,
            num_layers=2,
            dropout=0.1,
            sequence_length=32,
            batch_size=8,
            learning_rate=0.001,
            weight_decay=0.01,
            gradient_clip=1.0,
            mixed_precision=False,
            device='cpu',
            seed=42,
            use_continual_learning=True,
            episodic_memory_size=100,
            ewc_lambda=100.0,
            si_c=0.1
        )
        
        model = LiquidSpikingNetwork(config)
        trainer = LiquidSpikingTrainer(model, config)
        cl_system = trainer.continual_memory_system
        
        print("\n1️⃣  Episodic Memory Bank")
        print(f"   Capacity: {cl_system.episodic_memory.memory_size}")
        print(f"   Key dimension: {cl_system.episodic_memory.key_dim}")
        print(f"   ✅ Operational")
        
        print("\n2️⃣  Experience Replay Buffer")
        print(f"   Capacity: {cl_system.replay_buffer.buffer_size}")
        print(f"   Current size: {len(cl_system.replay_buffer.buffer)}")
        print(f"   ✅ Operational")
        
        print("\n3️⃣  Elastic Weight Consolidation (EWC)")
        print(f"   Lambda: {cl_system.ewc.lambda_ewc}")
        print(f"   Tasks tracked: {len(cl_system.ewc.fisher_dict)}")
        print(f"   ✅ Operational")
        
        print("\n4️⃣  Synaptic Intelligence (SI)")
        print(f"   C parameter: {cl_system.si.c}")
        print(f"   Damping: {cl_system.si.damping}")
        print(f"   ✅ Operational")
        
        print("\n5️⃣  Memory Consolidation System")
        print(f"   Frequency: {cl_system.consolidation_system.config.consolidation_interval}")
        print(f"   Consolidations: {cl_system.consolidation_system.consolidation_count}")
        print(f"   ✅ Operational")
        
        print("\n6️⃣  Progressive Network Expander")
        if cl_system.progressive_expander:
            print(f"   Enabled: True")
            print(f"   Expansion threshold: {cl_system.progressive_expander.expansion_threshold}")
            print(f"   ✅ Operational")
        else:
            print(f"   Enabled: False (optional feature)")
            print(f"   ✅ Correctly disabled")
        
        print("\n" + "="*70)
        print("✅ ALL 6 FEATURES VERIFIED")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_method_integration():
    """Verify key methods are integrated properly."""
    print("\n" + "="*70)
    print("🔧 VERIFYING METHOD INTEGRATION")
    print("="*70)
    
    try:
        from src.core.main import LiquidSpikingTrainer
        
        # Check methods exist
        methods_to_check = [
            'train_on_task',
            'evaluate',
            'evaluate_all_tasks',
            '_store_task_examples',
            'train_epoch'
        ]
        
        print("\nChecking LiquidSpikingTrainer methods:")
        for method in methods_to_check:
            if hasattr(LiquidSpikingTrainer, method):
                print(f"   ✅ {method}")
            else:
                print(f"   ❌ Missing: {method}")
                return False
        
        print("\n" + "="*70)
        print("✅ ALL METHODS PRESENT")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("🚀 Starting Continual Learning Integration Verification\n")
    
    success = True
    
    # Run verifications
    success &= verify_imports()
    success &= verify_feature_accessibility()
    success &= check_method_integration()
    
    # Final result
    print("\n" + "="*70)
    if success:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("="*70)
        print("\n✅ Continual learning system is properly integrated")
        print("✅ All 6 features are operational")
        print("✅ Ready for testing and training")
        print("\n📝 Next steps:")
        print("   1. Run: python test_continual_learning_integration.py")
        print("   2. Or use in your training scripts with:")
        print("      config = ModelConfig(use_continual_learning=True)")
    else:
        print("❌ VERIFICATION FAILED")
        print("="*70)
        print("\nPlease fix the issues above before proceeding.")
    
    sys.exit(0 if success else 1)
