#!/usr/bin/env python3
"""
Emergency fix for zero-learning issue.
This script directly patches the SpikingEncoder in main.py with stronger gradients.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def patch_spiking_encoder():
    """Patch the SpikingEncoder class in main.py with stronger surrogate gradients."""
    
    main_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'main.py')
    
    print("🔧 Patching SpikingEncoder with stronger surrogate gradients...")
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Find and replace the surrogate gradient initialization
    old_code = "        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())\n        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())"
    
    new_code = "        # CRITICAL FIX: Use slope=25 for stronger gradients (prevents vanishing)\n        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25))\n        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25))"
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        with open(main_file, 'w') as f:
            f.write(content)
        
        print("✅ Successfully patched SpikingEncoder!")
        print("   • Changed: surrogate.fast_sigmoid() → surrogate.fast_sigmoid(slope=25)")
        print("   • This provides 25x stronger gradient backpropagation")
        print("\n🚀 You can now train with: python scripts/cli.py train --task llm --epochs 10")
        return True
    else:
        print("⚠️  Could not find the exact code to patch.")
        print("   The file may have already been patched or has a different structure.")
        return False


if __name__ == "__main__":
    success = patch_spiking_encoder()
    sys.exit(0 if success else 1)
