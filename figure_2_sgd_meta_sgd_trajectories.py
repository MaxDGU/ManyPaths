#!/usr/bin/env python3
"""
Figure 2: Meta-SGD vs SGD Learning Trajectories with Shaded Error Bars

This script generates Figure 2 from the paper showing Meta-SGD learning curves
with SGD baseline trajectories including shaded standard deviations across seeds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import argparse
import re

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_complete_meta_sgd_trajectories():
    """Load complete Meta-SGD trajectories from concept_multiseed directory"""
    
    trajectories = {
        'F8D3': [],
        'F8D5': [],
        'F32D3': []
    }
    
    # Priority order: concept_multiseed (complete), then run1, then others
    search_dirs = [
        'results/concept_multiseed/',
        'results/run1/',
        'results/',
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        print(f"Searching in {search_dir}...")
        
        # Find trajectory files
        pattern = os.path.join(search_dir, '*trajectory*.csv')
        trajectory_files = glob.glob(pattern)
        
        for file_path in trajectory_files:
            filename = os.path.basename(file_path)
            
            # Skip baseline files
            if 'baseline' in filename.lower():
                continue
            
            # Extract configuration
            if 'feats8_depth3' in filename:
                key = 'F8D3'
            elif 'feats8_depth5' in filename:
                key = 'F8D5'
            elif 'feats32_depth3' in filename:
                key = 'F32D3'
            else:
                continue
                
            # For concept_multiseed, prioritize the highest epoch numbers
            if 'concept_multiseed' in search_dir:
                # Extract epoch number
                epoch_match = re.search(r'epoch_(\d+)', filename)
                if epoch_match:
                    epoch_num = int(epoch_match.group(1))
                    # Only use high epoch numbers (complete trajectories)
                    if epoch_num < 60:
                        continue
            
            try:
                df = pd.read_csv(file_path)
                if len(df) > 50:  # Only use trajectories with substantial data
                    trajectories[key].append({
                        'file': file_path,
                        'data': df,
                        'length': len(df)
                    })
                    print(f"  Loaded {key}: {filename} ({len(df)} steps)")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
    
    # Select best trajectories for each complexity
    final_trajectories = {}
    for key, trajs in trajectories.items():
        if trajs:
            # Sort by length (descending) and take the longest ones
            trajs.sort(key=lambda x: x['length'], reverse=True)
            # Take up to 3 longest trajectories
            final_trajectories[key] = trajs[:3]
            print(f"Selected {len(final_trajectories[key])} trajectories for {key}")
        else:
            print(f"No complete trajectories found for {key}")
            final_trajectories[key] = []
    
    return final_trajectories

def load_sgd_baseline_trajectories():
    """Load SGD baseline trajectories"""
    
    baseline_trajectories = {
        'F8D3': [],
        'F8D5': [],
        'F32D3': []
    }
    
    # Look for baseline trajectory files in nested subdirectories
    baseline_files = glob.glob('results/baseline_sgd/**/*baselinetrajectory*.csv', recursive=True)
    
    for file_path in baseline_files:
        filename = os.path.basename(file_path)
        
        # Extract configuration from the directory name and filename
        # Pattern: results/baseline_sgd/baseline_ms_feat8_dep3_seed1/concept_mlp_14_bits_feats8_depth3_sgdsteps100_lr0.001_runbaseline_ms_feat8_dep3_seed1_seed1_baselinetrajectory.csv
        
        # First try to extract from filename
        if 'feats8_depth3' in filename:
            key = 'F8D3'
        elif 'feats8_depth5' in filename:
            key = 'F8D5'
        elif 'feats32_depth3' in filename:
            key = 'F32D3'
        elif 'feats16_depth3' in filename:
            key = 'F16D3'  # Medium complexity
        else:
            # Try to extract from directory name
            dir_name = os.path.basename(os.path.dirname(file_path))
            if 'feat8_dep3' in dir_name:
                key = 'F8D3'
            elif 'feat8_dep5' in dir_name:
                key = 'F8D5'
            elif 'feat32_dep3' in dir_name:
                key = 'F32D3'
            elif 'feat16_dep3' in dir_name:
                key = 'F16D3'
            else:
                continue
            
        # Skip F16D3 for now since we're focusing on F8D3, F8D5, F32D3
        if key == 'F16D3':
            continue
            
        try:
            df = pd.read_csv(file_path)
            if len(df) > 10:  # Basic sanity check
                baseline_trajectories[key].append({
                    'file': file_path,
                    'data': df,
                    'seed': extract_seed_from_filename(filename)
                })
                print(f"Loaded SGD baseline {key}: {filename} ({len(df)} steps)")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return baseline_trajectories

def extract_seed_from_filename(filename):
    """Extract seed number from filename"""
    seed_match = re.search(r'seed(\d+)', filename)
    return int(seed_match.group(1)) if seed_match else 0

def create_trajectory_comparison():
    """Create Figure 2 style trajectory comparison"""
    
    print("Loading Meta-SGD trajectories...")
    meta_sgd_trajectories = load_complete_meta_sgd_trajectories()
    
    print("Loading SGD baseline trajectories...")
    sgd_baseline_trajectories = load_sgd_baseline_trajectories()
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    complexities = ['F8D3', 'F8D5', 'F32D3']
    complexity_names = ['Simple (F8D3)', 'Medium (F8D5)', 'Complex (F32D3)']
    
    for i, (complexity, name) in enumerate(zip(complexities, complexity_names)):
        ax = axes[i]
        
        # Plot Meta-SGD trajectories
        meta_sgd_accs = []
        meta_sgd_final_accs = []
        
        for traj in meta_sgd_trajectories[complexity]:
            df = traj['data']
            if 'val_accuracy' in df.columns:
                accuracy = df['val_accuracy'].values
                meta_sgd_accs.append(accuracy)
                meta_sgd_final_accs.append(accuracy[-1])
        
        # Plot Meta-SGD mean with std (normalized to 0-1 range)
        if meta_sgd_accs:
            # Normalize all trajectories to same length for cleaner comparison
            max_len = min(500, max(len(acc) for acc in meta_sgd_accs))  # Cap at 500 for readability
            
            padded_accs = []
            for acc in meta_sgd_accs:
                if len(acc) >= max_len:
                    # Subsample if trajectory is longer
                    indices = np.linspace(0, len(acc)-1, max_len).astype(int)
                    padded = acc[indices]
                else:
                    # Pad with last value if shorter
                    padded = np.pad(acc, (0, max_len - len(acc)), mode='edge')
                padded_accs.append(padded)
            
            meta_sgd_mean = np.mean(padded_accs, axis=0)
            meta_sgd_std = np.std(padded_accs, axis=0)
            
            # Normalize x-axis to 0-1 range
            x_range = np.linspace(0, 1, len(meta_sgd_mean))
            ax.plot(x_range, meta_sgd_mean, color='green', linewidth=3, label='Meta-SGD')
            ax.fill_between(x_range, meta_sgd_mean - meta_sgd_std, 
                           meta_sgd_mean + meta_sgd_std, alpha=0.2, color='green')
            
            meta_sgd_final_mean = np.mean(meta_sgd_final_accs)
            meta_sgd_final_std = np.std(meta_sgd_final_accs)
        else:
            meta_sgd_final_mean = 0
            meta_sgd_final_std = 0
            ax.text(0.5, 0.5, f'No complete\nMeta-SGD trajectories\nfor {complexity}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        # Plot SGD baseline trajectories  
        sgd_accs = []
        sgd_final_accs = []
        
        for traj in sgd_baseline_trajectories[complexity]:
            df = traj['data']
            # Handle different column names
            if 'query_accuracy' in df.columns:
                accuracy = df['query_accuracy'].values
            elif 'val_accuracy' in df.columns:
                accuracy = df['val_accuracy'].values
            else:
                continue
                
            sgd_accs.append(accuracy)
            sgd_final_accs.append(accuracy[-1])
        
        # Plot SGD mean with std (normalized to 0-1 range)
        if sgd_accs:
            # Normalize all trajectories to same length for cleaner comparison
            max_len = min(500, max(len(acc) for acc in sgd_accs))  # Cap at 500 for readability
            
            padded_accs = []
            for acc in sgd_accs:
                if len(acc) >= max_len:
                    # Subsample if trajectory is longer
                    indices = np.linspace(0, len(acc)-1, max_len).astype(int)
                    padded = acc[indices]
                else:
                    # Pad with last value if shorter
                    padded = np.pad(acc, (0, max_len - len(acc)), mode='edge')
                padded_accs.append(padded)
            
            sgd_mean = np.mean(padded_accs, axis=0)
            sgd_std = np.std(padded_accs, axis=0)
            
            # Apply heavy smoothing to SGD trajectories to reduce noise
            from scipy.ndimage import gaussian_filter1d
            sgd_mean_smooth = gaussian_filter1d(sgd_mean, sigma=10)
            sgd_std_smooth = gaussian_filter1d(sgd_std, sigma=5)
            
            # Normalize x-axis to 0-1 range
            x_range = np.linspace(0, 1, len(sgd_mean_smooth))
            ax.plot(x_range, sgd_mean_smooth, color='red', linewidth=3, label='SGD Baseline')
            ax.fill_between(x_range, sgd_mean_smooth - sgd_std_smooth, 
                           sgd_mean_smooth + sgd_std_smooth, alpha=0.2, color='red')
            
            sgd_final_mean = np.mean(sgd_final_accs)
            sgd_final_std = np.std(sgd_final_accs)
        else:
            sgd_final_mean = 0
            sgd_final_std = 0
        
        # Formatting
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Progress (Normalized)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add performance summary
        if meta_sgd_final_mean > 0 and sgd_final_mean > 0:
            improvement = meta_sgd_final_mean - sgd_final_mean
            ax.text(0.05, 0.95, f'SGD: {sgd_final_mean:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            ax.text(0.05, 0.85, f'Meta-SGD: {meta_sgd_final_mean:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax.text(0.05, 0.75, f'Δ: +{improvement:.3f}', 
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    plt.suptitle('Learning Trajectories: Meta-SGD vs SGD Baseline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('figure_2_meta_sgd_vs_sgd_trajectories.svg', dpi=300, bbox_inches='tight')
    plt.savefig('figure_2_meta_sgd_vs_sgd_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nFigure 2 trajectories saved!")

if __name__ == "__main__":
    create_trajectory_comparison() 