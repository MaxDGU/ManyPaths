#!/usr/bin/env python3
"""
Della Test Script for Gradient Alignment Computation

Quick validation that gradient alignment computation works on della
before running full overnight experiments.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import time
import subprocess

def test_gradient_alignment_della():
    """Test gradient alignment with minimal config on della."""
    print("=" * 60)
    print("🧪 DELLA GRADIENT ALIGNMENT TEST")
    print("=" * 60)
    
    # Very minimal test configuration
    test_cmd = [
        sys.executable, 'main.py',
        '--experiment', 'concept',
        '--m', 'mlp',
        '--data-type', 'bits',
        '--num-concept-features', '8',
        '--pcfg-max-depth', '3',
        '--adaptation-steps', '1',
        '--first-order',
        '--epochs', '5',  # Very short for testing
        '--tasks_per_meta_batch', '4',  # Fixed: underscore instead of hyphen
        '--outer_lr', '1e-3',  # Fixed: underscore instead of hyphen
        '--seed', '42',
        '--save',
        '--results_dir', 'results/della_grad_test'  # Use results_dir instead of run-name
    ]
    
    print(f"🚀 Running test command:")
    print(f"   {' '.join(test_cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode == 0:
            duration = time.time() - start_time
            print(f"✅ Test completed successfully in {duration:.1f}s")
            
            # Check gradient alignment data
            success, message = check_gradient_data("results/della_grad_test")
            
            if success:
                print(f"✅ Gradient alignment validation: {message}")
                print("\n🎉 DELLA TEST PASSED!")
                print("Ready to run full overnight experiments.")
                return True
            else:
                print(f"❌ Gradient alignment validation failed: {message}")
                return False
                
        else:
            print(f"❌ Test failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def check_gradient_data(results_dir):
    """Check if gradient alignment data was generated."""
    if not os.path.exists(results_dir):
        return False, "Results directory not found"
    
    # Find trajectory files
    trajectory_files = [f for f in os.listdir(results_dir) if f.endswith('_trajectory.csv')]
    if not trajectory_files:
        return False, "No trajectory files found"
    
    # Check latest trajectory file
    latest_file = max(trajectory_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
    trajectory_path = os.path.join(results_dir, latest_file)
    
    try:
        df = pd.read_csv(trajectory_path)
        
        if 'grad_alignment' not in df.columns:
            return False, "No gradient alignment column"
        
        grad_data = df['grad_alignment'].dropna()
        if len(grad_data) == 0:
            return False, "Gradient alignment column is empty"
        
        return True, f"Found {len(grad_data)} gradient alignment data points (range: {grad_data.min():.3f} to {grad_data.max():.3f})"
        
    except Exception as e:
        return False, f"Error reading trajectory file: {e}"

def main():
    """Run della gradient alignment test."""
    print("Starting della gradient alignment test...")
    
    # Basic environment check
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run test
    success = test_gradient_alignment_della()
    
    if success:
        print("\n📋 NEXT STEPS:")
        print("1. Run full gradient alignment experiments:")
        print("   python della_full_gradient_experiments.py")
        print("2. Or submit SLURM job:")
        print("   sbatch run_gradient_alignment_experiments.slurm")
        return 0
    else:
        print("\n⚠️  Test failed - need to debug before running full experiments")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 