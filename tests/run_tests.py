#!/usr/bin/env python3
"""
Unified test runner for liquid-spiking neural network project with optimization support.

This script provides a centralized way to run all tests including the new
optimization demonstrations and comparisons.
"""

import sys
import os
import argparse
import time
import subprocess

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test_module(test_name, description):
    """Run a specific test module."""
    print(f"\n🧪 Running {description}...")
    print("=" * 60)
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, f"tests/{test_name}.py"], 
                              capture_output=False, text=True, cwd="..")
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully ({duration:.1f}s)")
            return True
        else:
            print(f"❌ {description} failed (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Unified test runner for liquid-spiking neural network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  setup                 - Core component verification
  generation           - Text generation testing  
  demo                 - Focused demonstration
  evaluation           - Model evaluation and metrics
  comprehensive        - Comprehensive generation testing
  optimizations        - Optimization system testing
  optimization-demo    - Quick optimization demonstration
  optimization-comparison - Configuration comparison analysis
  all                  - Run all tests (default)

Examples:
  python run_tests.py                          # Run all tests
  python run_tests.py --test setup             # Run setup tests only
  python run_tests.py --test optimizations     # Test optimization system
  python run_tests.py --test optimization-demo # Quick optimization demo
        """
    )
    
    parser.add_argument(
        '--test', 
        choices=['setup', 'generation', 'demo', 'evaluation', 'comprehensive',
                'optimizations', 'optimization-demo', 'optimization-comparison', 'all'],
        default='all',
        help='Specific test category to run'
    )
    
    args = parser.parse_args()
    
    print("🚀 LIQUID-SPIKING NEURAL NETWORK TEST SUITE")
    print("=" * 65)
    print("🔬 Testing advanced liquid-spiking neural network implementation")
    print("⚡ Including comprehensive training optimizations")
    print("=" * 65)
    
    # Define test modules
    test_modules = {
        'setup': ('test_setup', 'Setup Verification Tests'),
        'generation': ('test_generation', 'Text Generation Tests'),
        'demo': ('demo_generation', 'Focused Generation Demonstration'),
        'evaluation': ('evaluate_model', 'Model Evaluation and Metrics'),
        'comprehensive': ('comprehensive_generation_test', 'Comprehensive Generation Testing'),
        'optimizations': ('test_optimized_training', 'Optimization System Testing'),
        'optimization-demo': ('quick_optimization_demo', 'Quick Optimization Demonstration'),
        'optimization-comparison': ('optimization_comparison', 'Configuration Comparison Analysis')
    }
    
    # Determine which tests to run
    if args.test == 'all':
        tests_to_run = list(test_modules.keys())
    else:
        tests_to_run = [args.test]
    
    # Run selected tests
    results = {}
    total_start_time = time.time()
    
    for test_name in tests_to_run:
        if test_name in test_modules:
            module_name, description = test_modules[test_name]
            results[test_name] = run_test_module(module_name, description)
        else:
            print(f"❌ Unknown test: {test_name}")
            results[test_name] = False
    
    total_duration = time.time() - total_start_time
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for result in results.values() if result)
    failed = len(results) - passed
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        description = test_modules.get(test_name, (test_name, test_name))[1]
        print(f"   {status} - {description}")
    
    print(f"\n📈 Results: {passed} passed, {failed} failed")
    print(f"⏱️  Total time: {total_duration:.1f}s")
    
    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        if args.test == 'all':
            print(f"🚀 Liquid-spiking neural network system fully functional!")
            print(f"⚡ Advanced optimizations verified and working!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

def main():
    parser = argparse.ArgumentParser(description="Test runner for Liquid-Spiking Neural Networks")
    parser.add_argument("--test", choices=["setup", "generation", "evaluation", "demo", "all"], 
                       default="all", help="Which test to run")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧠 LIQUID-SPIKING NEURAL NETWORK TEST RUNNER")
    print("=" * 80)
    
    results = {}
    
    if args.test in ["setup", "all"]:
        results["setup"] = run_setup_test() == 0
    
    if args.test in ["generation", "all"]:
        results["generation"] = run_generation_test()
    
    if args.test in ["evaluation", "all"]:
        results["evaluation"] = run_evaluation_test()
    
    if args.test in ["demo", "all"]:
        results["demo"] = run_demo()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.capitalize():12}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
