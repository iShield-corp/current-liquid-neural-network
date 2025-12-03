#!/usr/bin/env python3
"""
Simple launcher for the Liquid-Spiking Neural Network GUI.

This script checks dependencies and launches the GUI application.
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    # Check PyQt6
    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")
    
    # Check PyTorch
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    # Check transformers
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    return missing

def install_dependencies(packages):
    """Install missing dependencies."""
    import subprocess
    
    print(f"📦 Installing missing dependencies: {', '.join(packages)}")
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def main():
    """Main launcher function."""
    print("🧠 Liquid-Spiking Neural Network - GUI Launcher")
    print("=" * 60)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        response = input("Would you like to install them now? (y/n): ")
        
        if response.lower() == 'y':
            if not install_dependencies(missing):
                print("❌ Failed to install dependencies. Please install manually:")
                print(f"   pip install {' '.join(missing)}")
                return 1
        else:
            print("❌ Cannot launch GUI without required dependencies.")
            print(f"   Please install: pip install {' '.join(missing)}")
            return 1
    
    print("✅ All dependencies satisfied")
    print("\n🚀 Launching GUI...")
    print("=" * 60)
    
    # Add scripts directory to path
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    sys.path.insert(0, scripts_dir)
    
    # Launch GUI
    try:
        from scripts.gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"\n❌ Failed to launch GUI: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the ssn-cfc directory")
        print("2. Check that scripts/gui.py exists")
        print("3. Verify all dependencies are installed: pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
