#!/usr/bin/env python3
"""
Analyze Current Della Experiment Progress

Run this on della to get a comprehensive rundown of completed experiments.
"""

import os
import re
import glob
from collections import defaultdict
import pandas as pd
from datetime import datetime

def parse_filename(filename):
    """Parse experiment details from trajectory filename."""
    # Example: concept_mlp_7_bits_feats8_depth3_adapt1_1stOrd_seed1_epoch_183_trajectory.csv
    pattern = r'concept_mlp_(\d+)_bits_feats(\d+)_depth(\d+)_adapt(\d+)_(\w+)_seed(\d+)(?:_epoch_(\d+))?_trajectory\.csv'
    match = re.match(pattern, os.path.basename(filename))
    
    if match:
        return {
            'hyperparameter_idx': int(match.group(1)),
            'features': int(match.group(2)),
            'depth': int(match.group(3)),
            'adaptation_steps': int(match.group(4)),
            'order': match.group(5),
            'seed': int(match.group(6)),
            'epoch': int(match.group(7)) if match.group(7) else None,
            'is_final': match.group(7) is None
        }
    return None

def analyze_experiment_progress():
    """Analyze progress of all experiments."""
    
    # Find all trajectory files
    trajectory_files = glob.glob('results/*trajectory*.csv')
    if not trajectory_files:
        trajectory_files = glob.glob('*trajectory*.csv')
    if not trajectory_files:
        print("❌ No trajectory files found. Make sure you're in the right directory.")
        return
    
    print("🔍 DELLA EXPERIMENT PROGRESS ANALYSIS")
    print("=" * 80)
    
    # Parse all files
    experiments = defaultdict(list)
    
    for file_path in trajectory_files:
        parsed = parse_filename(file_path)
        if parsed:
            key = (parsed['features'], parsed['depth'], parsed['adaptation_steps'], parsed['seed'])
            parsed['file_path'] = file_path
            parsed['file_size'] = os.path.getsize(file_path)
            parsed['modified_time'] = os.path.getmtime(file_path)
            experiments[key].append(parsed)
    
    # Sort experiments by configuration
    print(f"📊 Found {len(trajectory_files)} trajectory files")
    print(f"📈 Covering {len(experiments)} unique experiments\n")
    
    # Group by configuration
    by_config = defaultdict(list)
    for key, files in experiments.items():
        features, depth, k_steps, seed = key
        config_key = (features, depth, k_steps)
        by_config[config_key].append((seed, files))
    
    # Analysis by configuration
    print("📋 EXPERIMENT STATUS BY CONFIGURATION:")
    print("-" * 80)
    
    total_completed = 0
    total_running = 0
    
    for config_key in sorted(by_config.keys()):
        features, depth, k_steps = config_key
        seeds_data = by_config[config_key]
        
        print(f"\n🎯 F{features}_D{depth}_K{k_steps}:")
        
        for seed, files in sorted(seeds_data):
            # Find the latest file for this seed
            latest_file = max(files, key=lambda x: x['modified_time'])
            
            if latest_file['is_final']:
                status = "✅ COMPLETED"
                total_completed += 1
            else:
                status = f"🔄 RUNNING (epoch {latest_file['epoch']})"
                total_running += 1
            
            # Get file age
            age_minutes = (datetime.now().timestamp() - latest_file['modified_time']) / 60
            
            print(f"  Seed {seed}: {status}")
            print(f"    Latest: {os.path.basename(latest_file['file_path'])}")
            print(f"    Size: {latest_file['file_size']:,} bytes")
            print(f"    Last modified: {age_minutes:.1f} minutes ago")
            
            # Quick data check
            try:
                df = pd.read_csv(latest_file['file_path'])
                print(f"    Progress: {len(df)} episodes, Val Acc: {df['val_acc'].iloc[-1]:.3f}")
            except Exception as e:
                print(f"    ⚠️  Could not read file: {e}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("📊 OVERALL SUMMARY")
    print("=" * 80)
    print(f"✅ Completed experiments: {total_completed}")
    print(f"🔄 Running experiments: {total_running}")
    print(f"📈 Total experiments: {total_completed + total_running}")
    
    # K=1 vs K=10 breakdown
    k1_experiments = [k for k in by_config.keys() if k[2] == 1]
    k10_experiments = [k for k in by_config.keys() if k[2] == 10]
    
    print(f"\n🔥 K=1 experiments: {len(k1_experiments)} configurations")
    for config in sorted(k1_experiments):
        features, depth, k_steps = config
        seeds = len(by_config[config])
        print(f"  F{features}_D{depth}: {seeds} seeds")
    
    print(f"\n🚀 K=10 experiments: {len(k10_experiments)} configurations")
    for config in sorted(k10_experiments):
        features, depth, k_steps = config
        seeds = len(by_config[config])
        print(f"  F{features}_D{depth}: {seeds} seeds")
    
    # Time analysis
    print(f"\n⏰ TIMING ANALYSIS")
    print("-" * 40)
    
    # Find most recent activity
    all_files = []
    for key, files in experiments.items():
        all_files.extend(files)
    
    if all_files:
        most_recent = max(all_files, key=lambda x: x['modified_time'])
        oldest = min(all_files, key=lambda x: x['modified_time'])
        
        recent_age = (datetime.now().timestamp() - most_recent['modified_time']) / 60
        total_duration = (most_recent['modified_time'] - oldest['modified_time']) / 3600
        
        print(f"🕐 Most recent activity: {recent_age:.1f} minutes ago")
        print(f"⏳ Total experiment duration: {total_duration:.1f} hours")
        
        if recent_age < 5:
            print("✅ Experiments appear to be actively running")
        elif recent_age < 30:
            print("⚠️  Recent activity, experiments may be between jobs")
        else:
            print("❌ No recent activity, experiments may have stopped")
    
    # Camera-ready readiness
    print(f"\n📝 CAMERA-READY READINESS")
    print("-" * 40)
    
    f8d3_k1_seeds = len([s for s in by_config.get((8, 3, 1), [])])
    f8d3_k10_seeds = len([s for s in by_config.get((8, 3, 10), [])])
    
    print(f"🎯 F8_D3 K=1: {f8d3_k1_seeds} seeds")
    print(f"🎯 F8_D3 K=10: {f8d3_k10_seeds} seeds")
    
    if f8d3_k1_seeds >= 2 and f8d3_k10_seeds >= 2:
        print("✅ EXCELLENT: Sufficient data for robust statistical analysis!")
    elif f8d3_k1_seeds >= 1 and f8d3_k10_seeds >= 1:
        print("⚠️  MINIMAL: Basic comparison possible, more seeds recommended")
    else:
        print("❌ INSUFFICIENT: Need at least 1 seed each for K=1 and K=10")
    
    return experiments

def main():
    """Run the analysis."""
    experiments = analyze_experiment_progress()
    
    print(f"\n📁 To get detailed analysis, check individual files:")
    print("   - Look for final trajectory files (without epoch number)")
    print("   - Check file sizes - larger files = more training progress") 
    print("   - Check timestamps - recent = currently running")
    
    print(f"\n🚀 Next steps:")
    print("   1. If experiments are still running: monitor progress")
    print("   2. If experiments completed: run analysis scripts")
    print("   3. If experiments stalled: check SLURM logs")

if __name__ == "__main__":
    main() 