#!/usr/bin/env python3
"""
Post-Clone Verification Script for Hybrid Liquid-Spiking Neural Network

Run this script after cloning the repository to verify that all imports work correctly.
"""

import sys
import os

def test_imports():
    """Test all critical imports to ensure the project is properly set up."""
    
    print("🧪 Testing Python imports for Hybrid Liquid-Spiking Neural Network...")
    print("=" * 70)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Core module imports
    print("1. Testing core module imports...")
    tests_total += 1
    try:
        from src.core.main import (
            LiquidSpikingNetwork, ModelConfig, TaskType, 
            LiquidSpikingTrainer, train_llm_model
        )
        print("   ✅ Core modules imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Core module import failed: {e}")
    
    # Test 2: Dataset module imports
    print("2. Testing dataset module imports...")
    tests_total += 1
    try:
        from src.datasets.advanced_programming_datasets import (
            AdvancedProgrammingDataset, ProgrammingDatasetConfig
        )
        from src.datasets.vision_datasets import VisionDatasetConfig
        from src.datasets.robotics_datasets import RoboticsDatasetConfig
        print("   ✅ Dataset modules imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Dataset module import failed: {e}")
    
    # Test 3: CLI script availability
    print("3. Testing CLI script availability...")
    tests_total += 1
    cli_path = os.path.join(os.path.dirname(__file__), 'scripts', 'cli.py')
    if os.path.exists(cli_path):
        print("   ✅ CLI script found")
        tests_passed += 1
    else:
        print("   ❌ CLI script not found")
    
    # Test 4: Package structure
    print("4. Testing package structure...")
    tests_total += 1
    required_files = [
        'src/__init__.py',
        'src/core/__init__.py', 
        'src/datasets/__init__.py',
        'src/core/main.py',
        'src/datasets/advanced_programming_datasets.py',
        'src/datasets/vision_datasets.py',
        'src/datasets/robotics_datasets.py',
        'scripts/cli.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if not missing_files:
        print("   ✅ All required files present")
        tests_passed += 1
    else:
        print(f"   ❌ Missing files: {', '.join(missing_files)}")
    
    # Test 5: PyTorch availability
    print("5. Testing PyTorch availability...")
    tests_total += 1
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} available")
        print(f"   ℹ️  CUDA available: {torch.cuda.is_available()}")
        tests_passed += 1
    except ImportError:
        print("   ❌ PyTorch not installed")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("🎉 SUCCESS: Repository is properly set up!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Test training: python scripts/cli.py train --task llm --epochs 1")
        print("3. Check system info: python scripts/cli.py info --system")
        return True
    else:
        print("❌ FAILED: Some imports are not working properly")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all files were cloned properly")
        print("2. Check that you're in the correct directory")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Create virtual environment if needed")
        return False

if __name__ == "__main__":
    print("🧠 Hybrid Liquid-Spiking Neural Network - Post-Clone Verification")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    print()
    
    success = test_imports()
    sys.exit(0 if success else 1)
