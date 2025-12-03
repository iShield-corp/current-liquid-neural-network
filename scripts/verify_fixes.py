#!/usr/bin/env python3
"""
Verify that gradient flow fixes are properly applied.
Run this before training to ensure the model will learn.
"""

import sys
import os
import torch

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.main import LiquidSpikingNetwork, create_llm_config, SpikingEncoder
import inspect

def check_fixes():
    """Check if all critical fixes are in place."""
    print("=" * 70)
    print("🔍 CHECKING GRADIENT FLOW FIXES")
    print("=" * 70)
    
    all_good = True
    
    # Check 1: SpikingEncoder surrogate gradient
    print("\n1️⃣  Checking SpikingEncoder surrogate gradient...")
    
    # Inspect the source code
    source = inspect.getsource(SpikingEncoder.__init__)
    
    if "slope=25" in source:
        print("   ✅ Surrogate gradient slope = 25 (GOOD)")
    elif "slope=" in source:
        print("   ⚠️  Surrogate gradient has slope parameter but not 25")
        print("      Found in source, but check value")
        all_good = False
    else:
        print("   ❌ Surrogate gradient has NO slope parameter (BAD)")
        print("      This will cause vanishing gradients!")
        all_good = False
    
    if "Dropout(0.1)" in source:
        print("   ✅ Dropout = 0.1 (GOOD)")
    else:
        print("   ⚠️  Dropout may be 0.2 (higher than recommended)")
    
    # Check 2: Default learning rate
    print("\n2️⃣  Checking default learning rate...")
    config = create_llm_config("gpt2")
    
    if config.learning_rate <= 5e-5:
        print(f"   ✅ Learning rate = {config.learning_rate} (GOOD)")
    elif config.learning_rate <= 1e-4:
        print(f"   ⚠️  Learning rate = {config.learning_rate} (OK, but 5e-5 better)")
    else:
        print(f"   ❌ Learning rate = {config.learning_rate} (TOO HIGH)")
        all_good = False
    
    # Check 3: Gradient clipping
    print("\n3️⃣  Checking gradient clipping...")
    
    if config.gradient_clip <= 0.5:
        print(f"   ✅ Gradient clip = {config.gradient_clip} (GOOD)")
    elif config.gradient_clip <= 1.0:
        print(f"   ⚠️  Gradient clip = {config.gradient_clip} (OK, but 0.5 better)")
    else:
        print(f"   ❌ Gradient clip = {config.gradient_clip} (TOO LOOSE)")
        all_good = False
    
    # Check 4: Test actual gradient flow
    print("\n4️⃣  Testing actual gradient flow...")
    
    try:
        model = LiquidSpikingNetwork(config)
        model.train()
        
        # Create dummy input
        batch_size = 2
        seq_len = 32
        dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        dummy_target = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        output = model(dummy_input)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, output.size(-1)),
            dummy_target.view(-1)
        )
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norms = []
        zero_grads = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                total_params += 1
                
                if grad_norm == 0:
                    zero_grads += 1
            else:
                zero_grads += 1
                total_params += 1
        
        if grad_norms:
            mean_grad = sum(grad_norms) / len(grad_norms)
            max_grad = max(grad_norms)
            
            print(f"   Mean gradient norm: {mean_grad:.6f}")
            print(f"   Max gradient norm: {max_grad:.6f}")
            print(f"   Zero gradients: {zero_grads}/{total_params}")
            
            if mean_grad > 1e-6:
                print("   ✅ Gradients flowing (GOOD)")
            elif mean_grad > 1e-8:
                print("   ⚠️  Gradients very small (may vanish)")
                all_good = False
            else:
                print("   ❌ Gradients vanishing (BAD)")
                all_good = False
        else:
            print("   ❌ No gradients computed")
            all_good = False
            
    except Exception as e:
        print(f"   ❌ Error during gradient test: {e}")
        all_good = False
    
    # Final verdict
    print("\n" + "=" * 70)
    print("🎯 VERDICT")
    print("=" * 70)
    
    if all_good:
        print("\n✅ ALL FIXES IN PLACE!")
        print("   Your model should learn properly now.")
        print("\n🚀 Ready to train:")
        print("   python scripts/cli.py train --task llm --epochs 10")
        print("   python train.py llm --epochs 10")
        return 0
    else:
        print("\n⚠️  SOME ISSUES DETECTED")
        print("   Please review the warnings above.")
        print("\n🔧 To apply fixes:")
        print("   python scripts/emergency_fix_gradients.py")
        return 1


if __name__ == "__main__":
    exit_code = check_fixes()
    sys.exit(exit_code)
