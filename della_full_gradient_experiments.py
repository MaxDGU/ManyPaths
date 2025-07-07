#!/usr/bin/env python3
"""
Della Full Gradient Alignment Experiments

Comprehensive gradient alignment experiments for K=1 vs K=10 comparison
designed to run overnight on della cluster.
"""

import subprocess
import os
import sys
import time
import json
from datetime import datetime

def run_single_experiment(config):
    """Run a single gradient alignment experiment."""
    name = config['name']
    print(f"🚀 Starting {name} experiment...")
    
    cmd = [
        sys.executable, 'main.py',
        '--experiment', 'concept',
        '--m', 'mlp',
        '--data-type', 'bits',
        '--num-concept-features', str(config['features']),
        '--pcfg-max-depth', str(config['depth']),
        '--adaptation-steps', str(config['adaptation_steps']),
        '--first-order',
        '--epochs', str(config['epochs']),
        '--seed', str(config['seed']),
        '--save'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        # Run with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.get('timeout', 14400))  # 4 hour default
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {name} completed successfully in {duration:.1f}s")
            return True, duration, "Success"
        else:
            print(f"❌ {name} failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, duration, f"Failed with return code {result.returncode}"
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ {name} timed out after {duration:.1f}s")
        return False, duration, "Timeout"
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ {name} failed with exception: {e}")
        return False, duration, f"Exception: {e}"

def validate_gradient_alignment_data(run_name):
    """Validate gradient alignment data was generated."""
    import pandas as pd
    
    results_dir = "results"  # Use default results directory since --results_dir not available
    if not os.path.exists(results_dir):
        return False, "Results directory not found"
    
    trajectory_files = [f for f in os.listdir(results_dir) if f.endswith('_trajectory.csv')]
    if not trajectory_files:
        return False, "No trajectory files found"
    
    # Look for files that match our experiment pattern
    matching_files = [f for f in trajectory_files if run_name in f or 'concept_mlp' in f]
    if not matching_files:
        return False, f"No trajectory files found matching pattern for {run_name}"
    
    latest_file = max(matching_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
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
    """Run comprehensive gradient alignment experiments."""
    print("=" * 80)
    print("🌙 DELLA OVERNIGHT GRADIENT ALIGNMENT EXPERIMENTS")
    print("=" * 80)
    
    # Experiment configurations
    experiments = [
        # Core K=1 vs K=10 comparison - F8_D3 (highest priority)
        {
            'name': 'k1_f8d3_s0',
            'features': 8,
            'depth': 3,
            'adaptation_steps': 1,
            'epochs': 200,
            'seed': 0,
            'timeout': 7200,  # 2 hours
            'priority': 'HIGH'
        },
        {
            'name': 'k1_f8d3_s1',
            'features': 8,
            'depth': 3,
            'adaptation_steps': 1,
            'epochs': 200,
            'seed': 1,
            'timeout': 7200,
            'priority': 'HIGH'
        },
        {
            'name': 'k10_f8d3_s0',
            'features': 8,
            'depth': 3,
            'adaptation_steps': 10,
            'epochs': 200,
            'seed': 0,
            'timeout': 10800,  # 3 hours (K=10 takes longer)
            'priority': 'HIGH'
        },
        {
            'name': 'k10_f8d3_s1',
            'features': 8,
            'depth': 3,
            'adaptation_steps': 10,
            'epochs': 200,
            'seed': 1,
            'timeout': 10800,
            'priority': 'HIGH'
        },
        
        # Extended comparison - F16_D3 (medium priority)
        {
            'name': 'k1_f16d3_s0',
            'features': 16,
            'depth': 3,
            'adaptation_steps': 1,
            'epochs': 150,
            'seed': 0,
            'timeout': 7200,
            'priority': 'MEDIUM'
        },
        {
            'name': 'k10_f16d3_s0',
            'features': 16,
            'depth': 3,
            'adaptation_steps': 10,
            'epochs': 150,
            'seed': 0,
            'timeout': 10800,
            'priority': 'MEDIUM'
        },
        
        # Complexity scaling - F32_D3 (lower priority)
        {
            'name': 'k1_f32d3_s0',
            'features': 32,
            'depth': 3,
            'adaptation_steps': 1,
            'epochs': 100,
            'seed': 0,
            'timeout': 7200,
            'priority': 'LOW'
        },
        {
            'name': 'k10_f32d3_s0',
            'features': 32,
            'depth': 3,
            'adaptation_steps': 10,
            'epochs': 100,
            'seed': 0,
            'timeout': 10800,
            'priority': 'LOW'
        }
    ]
    
    # Sort by priority (HIGH first)
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    experiments.sort(key=lambda x: priority_order[x['priority']])
    
    print(f"📊 Total experiments to run: {len(experiments)}")
    print(f"🔥 High priority: {len([e for e in experiments if e['priority'] == 'HIGH'])}")
    print(f"🟡 Medium priority: {len([e for e in experiments if e['priority'] == 'MEDIUM'])}")
    print(f"🟢 Low priority: {len([e for e in experiments if e['priority'] == 'LOW'])}")
    
    # Run experiments
    results = []
    total_start_time = time.time()
    
    for i, config in enumerate(experiments, 1):
        print(f"\n{'=' * 60}")
        print(f"🧪 EXPERIMENT {i}/{len(experiments)}: {config['name']} [{config['priority']} PRIORITY]")
        print(f"{'=' * 60}")
        
        success, duration, message = run_single_experiment(config)
        
        # Validate gradient alignment data
        if success:
            has_grad_data, grad_message = validate_gradient_alignment_data(f"grad_align_{config['name']}")
            print(f"📈 Gradient alignment validation: {grad_message}")
        else:
            has_grad_data, grad_message = False, "Experiment failed"
        
        results.append({
            'name': config['name'],
            'priority': config['priority'],
            'success': success,
            'duration': duration,
            'message': message,
            'has_gradient_data': has_grad_data,
            'gradient_message': grad_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save intermediate results
        with open('gradient_experiments_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Progress: {i}/{len(experiments)} complete")
    
    # Final summary
    total_duration = time.time() - total_start_time
    
    print("\n" + "=" * 80)
    print("📋 FINAL EXPERIMENT SUMMARY")
    print("=" * 80)
    
    successful_experiments = [r for r in results if r['success']]
    experiments_with_grad_data = [r for r in results if r['has_gradient_data']]
    
    print(f"🕐 Total time: {total_duration/3600:.1f} hours")
    print(f"✅ Successful experiments: {len(successful_experiments)}/{len(experiments)}")
    print(f"📈 Experiments with gradient data: {len(experiments_with_grad_data)}/{len(experiments)}")
    
    # Priority breakdown
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        priority_results = [r for r in results if r['priority'] == priority]
        priority_success = [r for r in priority_results if r['success'] and r['has_gradient_data']]
        print(f"🎯 {priority} priority: {len(priority_success)}/{len(priority_results)} successful")
    
    print("\n📊 Detailed Results:")
    for result in results:
        status = "✅" if result['success'] and result['has_gradient_data'] else "❌"
        print(f"{status} {result['name']}: {result['message']} | {result['gradient_message']}")
    
    # Next steps
    if len(experiments_with_grad_data) >= 4:  # At least the HIGH priority experiments
        print("\n🎉 SUFFICIENT DATA COLLECTED!")
        print("Next steps:")
        print("1. Run gradient alignment analysis:")
        print("   python gradient_alignment_analysis.py")
        print("2. Generate K=1 vs K=10 comparison plots:")
        print("   python k1_vs_k10_comparison.py")
        print("3. Create mechanistic explanation figures")
    else:
        print("\n⚠️  INSUFFICIENT DATA COLLECTED")
        print("May need to debug gradient alignment computation or re-run failed experiments")
    
    return 0 if len(experiments_with_grad_data) >= 4 else 1

if __name__ == "__main__":
    sys.exit(main()) 