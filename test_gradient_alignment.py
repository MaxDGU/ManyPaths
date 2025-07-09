#!/usr/bin/env python3
"""
Quick test to verify gradient alignment computation works
"""

import torch
import numpy as np
import pandas as pd
import os
from main import main as run_main
import sys

def test_gradient_alignment():
    """Test gradient alignment computation with minimal configuration."""
    print("Testing gradient alignment computation...")
    
    # Minimal test configuration
    test_args = [
        '--experiment', 'concept',
        '--m', 'mlp',
        '--data-type', 'bits',
        '--num-concept-features', '8',
        '--pcfg-max-depth', '3',
        '--adaptation-steps', '1',
        '--first-order',
        '--epochs', '10',  # Very short test
        '--tasks-per-meta-batch', '4',
        '--outer-lr', '1e-3',
        '--seed', '42',
        '--verbose',
        '--save',
        '--run-name', 'grad_align_test'
    ]
    
    # Temporarily redirect sys.argv
    original_argv = sys.argv
    sys.argv = ['test_gradient_alignment.py'] + test_args
    
    try:
        # Run the main function
        run_main()
        
        # Check if gradient alignment data was saved
        results_dir = "results/grad_align_test"
        trajectory_files = [f for f in os.listdir(results_dir) if f.endswith('_trajectory.csv')]
        
        if trajectory_files:
            latest_file = max(trajectory_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
            trajectory_path = os.path.join(results_dir, latest_file)
            
            # Load and check gradient alignment data
            df = pd.read_csv(trajectory_path)
            print(f"Loaded trajectory file: {latest_file}")
            print(f"Columns: {df.columns.tolist()}")
            
            if 'grad_alignment' in df.columns:
                grad_data = df['grad_alignment'].dropna()
                print(f"Gradient alignment data points: {len(grad_data)}")
                print(f"Non-NaN gradient alignments: {len(grad_data)}")
                
                if len(grad_data) > 0:
                    print(f"Gradient alignment range: {grad_data.min():.4f} to {grad_data.max():.4f}")
                    print(f"Mean gradient alignment: {grad_data.mean():.4f}")
                    print("✅ Gradient alignment computation is working!")
                    return True
                else:
                    print("❌ Gradient alignment column exists but contains no data")
                    return False
            else:
                print("❌ No gradient alignment column found")
                return False
        else:
            print("❌ No trajectory files found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

if __name__ == "__main__":
    success = test_gradient_alignment()
    if success:
        print("\n🎉 Test passed! Ready to run targeted experiments.")
    else:
        print("\n⚠️  Test failed. Need to debug gradient alignment computation first.") 