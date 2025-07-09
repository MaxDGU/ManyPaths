#!/usr/bin/env python3
"""
Targeted Gradient Alignment Experiments for Camera-Ready Submission

This script runs minimal targeted experiments to get gradient alignment data
for the K=1 vs K=10 comparison needed for mechanistic explanations.
"""

import subprocess
import os
import time
import sys

def run_experiment(name, adaptation_steps, seed=0, epochs=100):
    """Run a single gradient alignment experiment."""
    print(f"🚀 Starting {name} experiment...")
    
    cmd = [
        sys.executable, 'main.py',
        '--experiment', 'concept',
        '--m', 'mlp',
        '--data-type', 'bits',
        '--num-concept-features', '8',
        '--pcfg-max-depth', '3',
        '--adaptation-steps', str(adaptation_steps),
        '--first-order',
        '--epochs', str(epochs),
        '--seed', str(seed),
        '--verbose',
        '--save',
        '--run-name', f'grad_align_{name}_f8d3_seed{seed}'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        if result.returncode == 0:
            duration = time.time() - start_time
            print(f"✅ {name} experiment completed successfully in {duration:.1f}s")
            return True
        else:
            print(f"❌ {name} experiment failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {name} experiment timed out after 2 hours")
        return False
    except Exception as e:
        print(f"❌ {name} experiment failed with exception: {e}")
        return False

def check_gradient_alignment_data(run_name):
    """Check if gradient alignment data was generated."""
    results_dir = f"results/{run_name}"
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
        import pandas as pd
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
    """Run targeted gradient alignment experiments."""
    print("="*60)
    print("🎯 TARGETED GRADIENT ALIGNMENT EXPERIMENTS")
    print("="*60)
    
    experiments = [
        ("k1", 1, "K=1 First-Order MetaSGD"),
        ("k10", 10, "K=10 First-Order MetaSGD")
    ]
    
    results = {}
    
    for name, adaptation_steps, description in experiments:
        print(f"\n📊 Running {description}...")
        success = run_experiment(name, adaptation_steps, seed=0, epochs=100)
        
        if success:
            # Check gradient alignment data
            run_name = f'grad_align_{name}_f8d3_seed0'
            has_data, message = check_gradient_alignment_data(run_name)
            results[name] = {
                'success': success,
                'has_gradient_data': has_data,
                'message': message
            }
            print(f"📈 Gradient alignment data check: {message}")
        else:
            results[name] = {
                'success': False,
                'has_gradient_data': False,
                'message': "Experiment failed"
            }
    
    # Summary
    print("\n" + "="*60)
    print("📋 EXPERIMENT SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        status = "✅ SUCCESS" if result['success'] and result['has_gradient_data'] else "❌ FAILED"
        print(f"{name.upper()}: {status}")
        print(f"  Message: {result['message']}")
    
    # Next steps
    all_successful = all(r['success'] and r['has_gradient_data'] for r in results.values())
    
    if all_successful:
        print("\n🎉 ALL EXPERIMENTS SUCCESSFUL!")
        print("Next steps:")
        print("1. Run gradient alignment analysis:")
        print("   python gradient_alignment_analysis.py --base_results_dir results/grad_align_k10_f8d3_seed0")
        print("2. Create K=1 vs K=10 gradient alignment comparison")
        print("3. Generate mechanistic explanation figures")
    else:
        print("\n⚠️  SOME EXPERIMENTS FAILED")
        print("Need to debug gradient alignment computation or adjust strategy")

if __name__ == "__main__":
    main() 