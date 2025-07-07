#!/usr/bin/env python3
"""
Additional Della Experiments for Camera-Ready Paper

High-impact experiments to strengthen the camera-ready submission:
1. More seeds for statistical robustness
2. Broader configuration coverage  
3. Alternative mechanistic analyses
4. Baseline comparisons
"""

import subprocess
import os
import sys
import time
import json
from datetime import datetime

def run_single_experiment(config):
    """Run a single experiment."""
    name = config['name']
    print(f"🚀 Starting {name} experiment...")
    
    cmd = [
        sys.executable, 'main.py',
        '--experiment', config['experiment'],
        '--m', config['model'],
        '--data-type', 'bits',
        '--num-concept-features', str(config['features']),
        '--pcfg-max-depth', str(config['depth']),
        '--adaptation-steps', str(config['adaptation_steps']),
        '--epochs', str(config['epochs']),
        '--seed', str(config['seed']),
        '--save',
        '--no_hyper_search'
    ]
    
    if config.get('first_order', True):
        cmd.append('--first-order')
    
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.get('timeout', 14400))
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

def main():
    """Run additional experiments for camera-ready submission."""
    print("=" * 80)
    print("🌙 ADDITIONAL DELLA EXPERIMENTS FOR CAMERA-READY")
    print("=" * 80)
    
    # BATCH 1: More seeds for statistical robustness (URGENT)
    batch1_experiments = [
        # F8_D3 - Additional seeds for robustness
        {'name': 'k1_f8d3_s2', 'experiment': 'concept', 'model': 'mlp', 'features': 8, 'depth': 3, 'adaptation_steps': 1, 'epochs': 200, 'seed': 2, 'timeout': 7200, 'priority': 'URGENT'},
        {'name': 'k1_f8d3_s3', 'experiment': 'concept', 'model': 'mlp', 'features': 8, 'depth': 3, 'adaptation_steps': 1, 'epochs': 200, 'seed': 3, 'timeout': 7200, 'priority': 'URGENT'},
        {'name': 'k10_f8d3_s2', 'experiment': 'concept', 'model': 'mlp', 'features': 8, 'depth': 3, 'adaptation_steps': 10, 'epochs': 200, 'seed': 2, 'timeout': 10800, 'priority': 'URGENT'},
        {'name': 'k10_f8d3_s3', 'experiment': 'concept', 'model': 'mlp', 'features': 8, 'depth': 3, 'adaptation_steps': 10, 'epochs': 200, 'seed': 3, 'timeout': 10800, 'priority': 'URGENT'},
        
        # F16_D3 - Additional seeds 
        {'name': 'k1_f16d3_s1', 'experiment': 'concept', 'model': 'mlp', 'features': 16, 'depth': 3, 'adaptation_steps': 1, 'epochs': 150, 'seed': 1, 'timeout': 7200, 'priority': 'URGENT'},
        {'name': 'k10_f16d3_s1', 'experiment': 'concept', 'model': 'mlp', 'features': 16, 'depth': 3, 'adaptation_steps': 10, 'epochs': 150, 'seed': 1, 'timeout': 10800, 'priority': 'URGENT'},
    ]
    
    # BATCH 2: Broader configuration coverage (HIGH)
    batch2_experiments = [
        # Different depths for complexity scaling
        {'name': 'k1_f8d5_s0', 'experiment': 'concept', 'model': 'mlp', 'features': 8, 'depth': 5, 'adaptation_steps': 1, 'epochs': 200, 'seed': 0, 'timeout': 8400, 'priority': 'HIGH'},
        {'name': 'k10_f8d5_s0', 'experiment': 'concept', 'model': 'mlp', 'features': 8, 'depth': 5, 'adaptation_steps': 10, 'epochs': 200, 'seed': 0, 'timeout': 12000, 'priority': 'HIGH'},
        
        # Different architectures
        {'name': 'k1_f8d3_cnn_s0', 'experiment': 'concept', 'model': 'cnn', 'features': 8, 'depth': 3, 'adaptation_steps': 1, 'epochs': 150, 'seed': 0, 'timeout': 7200, 'priority': 'HIGH'},
        {'name': 'k10_f8d3_cnn_s0', 'experiment': 'concept', 'model': 'cnn', 'features': 8, 'depth': 3, 'adaptation_steps': 10, 'epochs': 150, 'seed': 0, 'timeout': 10800, 'priority': 'HIGH'},
        
        # Intermediate complexity
        {'name': 'k1_f12d3_s0', 'experiment': 'concept', 'model': 'mlp', 'features': 12, 'depth': 3, 'adaptation_steps': 1, 'epochs': 150, 'seed': 0, 'timeout': 7200, 'priority': 'HIGH'},
        {'name': 'k10_f12d3_s0', 'experiment': 'concept', 'model': 'mlp', 'features': 12, 'depth': 3, 'adaptation_steps': 10, 'epochs': 150, 'seed': 0, 'timeout': 10800, 'priority': 'HIGH'},
    ]
    
    # BATCH 3: Alternative datasets (MEDIUM)
    batch3_experiments = [
        # Mod dataset experiments
        {'name': 'k1_mod_f8d3_s0', 'experiment': 'mod', 'model': 'mlp', 'features': 8, 'depth': 3, 'adaptation_steps': 1, 'epochs': 100, 'seed': 0, 'timeout': 7200, 'priority': 'MEDIUM'},
        {'name': 'k10_mod_f8d3_s0', 'experiment': 'mod', 'model': 'mlp', 'features': 8, 'depth': 3, 'adaptation_steps': 10, 'epochs': 100, 'seed': 0, 'timeout': 10800, 'priority': 'MEDIUM'},
        
        # Omniglot dataset experiments
        {'name': 'k1_omni_s0', 'experiment': 'omniglot', 'model': 'cnn', 'features': 8, 'depth': 3, 'adaptation_steps': 1, 'epochs': 100, 'seed': 0, 'timeout': 7200, 'priority': 'MEDIUM', 'first_order': True},
        {'name': 'k10_omni_s0', 'experiment': 'omniglot', 'model': 'cnn', 'features': 8, 'depth': 3, 'adaptation_steps': 10, 'epochs': 100, 'seed': 0, 'timeout': 10800, 'priority': 'MEDIUM', 'first_order': True},
    ]
    
    # BATCH 4: Long training runs for convergence analysis (LOW)
    batch4_experiments = [
        # Extended training for publication-quality curves
        {'name': 'k1_f8d3_s0_long', 'experiment': 'concept', 'model': 'mlp', 'features': 8, 'depth': 3, 'adaptation_steps': 1, 'epochs': 500, 'seed': 0, 'timeout': 18000, 'priority': 'LOW'},
        {'name': 'k10_f8d3_s0_long', 'experiment': 'concept', 'model': 'mlp', 'features': 8, 'depth': 3, 'adaptation_steps': 10, 'epochs': 500, 'seed': 0, 'timeout': 25200, 'priority': 'LOW'},
        
        # Different K values for mechanistic analysis
        {'name': 'k3_f8d3_s0', 'experiment': 'concept', 'model': 'mlp', 'features': 8, 'depth': 3, 'adaptation_steps': 3, 'epochs': 200, 'seed': 0, 'timeout': 8400, 'priority': 'LOW'},
        {'name': 'k5_f8d3_s0', 'experiment': 'concept', 'model': 'mlp', 'features': 8, 'depth': 3, 'adaptation_steps': 5, 'epochs': 200, 'seed': 0, 'timeout': 9600, 'priority': 'LOW'},
    ]
    
    # Combine all batches
    all_experiments = batch1_experiments + batch2_experiments + batch3_experiments + batch4_experiments
    
    # Sort by priority (URGENT first)
    priority_order = {'URGENT': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    all_experiments.sort(key=lambda x: priority_order[x['priority']])
    
    print(f"📊 Additional experiments to run: {len(all_experiments)}")
    print(f"🔥 URGENT (statistical robustness): {len([e for e in all_experiments if e['priority'] == 'URGENT'])}")
    print(f"🟡 HIGH (broader coverage): {len([e for e in all_experiments if e['priority'] == 'HIGH'])}")
    print(f"🟢 MEDIUM (alternative datasets): {len([e for e in all_experiments if e['priority'] == 'MEDIUM'])}")
    print(f"🔵 LOW (extended analysis): {len([e for e in all_experiments if e['priority'] == 'LOW'])}")
    
    # Create results tracking
    results = {
        'start_time': datetime.now().isoformat(),
        'total_experiments': len(all_experiments),
        'completed': [],
        'failed': [],
        'summary': {}
    }
    
    # Run experiments
    for i, config in enumerate(all_experiments):
        print(f"\n{'='*80}")
        print(f"🎯 EXPERIMENT {i+1}/{len(all_experiments)}: {config['name']} [{config['priority']}]")
        print(f"{'='*80}")
        
        success, duration, status = run_single_experiment(config)
        
        result_entry = {
            'name': config['name'],
            'priority': config['priority'],
            'duration': duration,
            'status': status,
            'success': success
        }
        
        if success:
            results['completed'].append(result_entry)
            print(f"✅ Progress: {len(results['completed'])}/{len(all_experiments)} completed")
        else:
            results['failed'].append(result_entry)
            print(f"❌ Progress: {len(results['failed'])} failed, {len(results['completed'])}/{len(all_experiments)} completed")
        
        # Save intermediate results
        results['end_time'] = datetime.now().isoformat()
        with open('additional_experiments_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "="*80)
    print("📋 FINAL SUMMARY")
    print("="*80)
    print(f"✅ Completed: {len(results['completed'])}/{len(all_experiments)}")
    print(f"❌ Failed: {len(results['failed'])}/{len(all_experiments)}")
    
    # Priority breakdown
    for priority in ['URGENT', 'HIGH', 'MEDIUM', 'LOW']:
        completed = len([r for r in results['completed'] if r['priority'] == priority])
        total = len([e for e in all_experiments if e['priority'] == priority])
        print(f"  {priority}: {completed}/{total}")
    
    # Save final results
    with open('additional_experiments_final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to: additional_experiments_final_results.json")
    
    return len(results['completed']) > 0

if __name__ == "__main__":
    main() 