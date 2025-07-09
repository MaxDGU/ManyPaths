#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import re
import os
from scipy import stats
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Set up the plotting style
plt.style.use('seaborn-v0_8')
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
                        'length': len(df),
                        'seed': extract_seed_from_filename(filename)
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

def normalize_trajectory(trajectory, target_length=200):
    """Normalize trajectory to target length for fair comparison"""
    if len(trajectory) <= target_length:
        return trajectory
    
    # Subsample to target length
    indices = np.linspace(0, len(trajectory)-1, target_length).astype(int)
    return trajectory[indices]

def compute_landscape_roughness(trajectory_data, method='normalized_variance'):
    """Compute landscape roughness from trajectory data with multiple methods"""
    
    # Handle different column names for loss
    if 'query_loss' in trajectory_data.columns:
        losses = trajectory_data['query_loss'].values
    elif 'val_loss' in trajectory_data.columns:
        losses = trajectory_data['val_loss'].values
    else:
        print("Warning: No loss column found")
        return 0.1
    
    # Normalize trajectory length for fair comparison
    losses = normalize_trajectory(losses)
    
    # Apply smoothing to reduce noise
    losses = gaussian_filter1d(losses, sigma=1)
    
    if method == 'normalized_variance':
        # Normalize by mean loss to make comparisons fair
        mean_loss = np.mean(losses)
        roughness = np.std(losses) / (mean_loss + 1e-8)
        
    elif method == 'gradient_magnitude':
        # Compute average gradient magnitude
        gradients = np.diff(losses)
        roughness = np.mean(np.abs(gradients))
        
    elif method == 'second_derivative':
        # Approximate second derivative (curvature)
        if len(losses) < 3:
            return 0.1
        second_derivatives = np.diff(losses, n=2)
        roughness = np.mean(np.abs(second_derivatives))
        
    else:
        # Default: coefficient of variation
        roughness = np.std(losses) / (np.mean(losses) + 1e-8)
    
    return roughness

def compute_curvature_measures(trajectory_data):
    """Compute differential geometry curvature measures"""
    
    curvature_measures = {}
    
    # Handle different column names for loss
    if 'query_loss' in trajectory_data.columns:
        losses = trajectory_data['query_loss'].values
    elif 'val_loss' in trajectory_data.columns:
        losses = trajectory_data['val_loss'].values
    else:
        return {'hessian_trace': 0.1, 'mean_curvature': 0.1, 'gaussian_curvature': 0.1}
    
    # Normalize trajectory length 
    losses = normalize_trajectory(losses)
    
    # Smooth trajectory to reduce noise
    losses = gaussian_filter1d(losses, sigma=1)
    
    if len(losses) < 5:
        return {'hessian_trace': 0.1, 'mean_curvature': 0.1, 'gaussian_curvature': 0.1}
    
    # First derivatives (gradients)
    first_derivatives = np.diff(losses)
    
    # Second derivatives (curvature)
    second_derivatives = np.diff(losses, n=2)
    
    # Hessian trace approximation
    hessian_trace = np.mean(np.abs(second_derivatives))
    
    # Mean curvature (average of second derivatives)
    mean_curvature = np.mean(second_derivatives)
    
    # Gaussian curvature approximation (product of eigenvalues)
    # For 1D case, approximate using variance of second derivatives
    gaussian_curvature = np.var(second_derivatives)
    
    curvature_measures = {
        'hessian_trace': hessian_trace,
        'mean_curvature': abs(mean_curvature),
        'gaussian_curvature': gaussian_curvature
    }
    
    return curvature_measures

def analyze_landscape_comparison():
    """Comprehensive landscape and curvature analysis: SGD vs Meta-SGD"""
    
    print("Loading Meta-SGD trajectories...")
    meta_sgd_trajectories = load_complete_meta_sgd_trajectories()
    
    print("Loading SGD baseline trajectories...")
    sgd_baseline_trajectories = load_sgd_baseline_trajectories()
    
    # Prepare data for analysis
    analysis_data = []
    
    complexities = ['F8D3', 'F8D5', 'F32D3']
    complexity_names = ['Simple (F8D3)', 'Medium (F8D5)', 'Complex (F32D3)']
    
    for complexity in complexities:
        # Meta-SGD analysis
        for traj in meta_sgd_trajectories[complexity]:
            roughness = compute_landscape_roughness(traj['data'])
            curvature_measures = compute_curvature_measures(traj['data'])
            
            # Handle different column names for accuracy
            if 'val_accuracy' in traj['data'].columns:
                final_acc = traj['data']['val_accuracy'].iloc[-1]
            elif 'query_accuracy' in traj['data'].columns:
                final_acc = traj['data']['query_accuracy'].iloc[-1]
            else:
                final_acc = 0.5
            
            analysis_data.append({
                'Complexity': complexity,
                'Method': 'Meta-SGD',
                'Roughness': roughness,
                'Final_Accuracy': final_acc,
                'Hessian_Trace': curvature_measures['hessian_trace'],
                'Mean_Curvature': curvature_measures['mean_curvature'],
                'Gaussian_Curvature': curvature_measures['gaussian_curvature'],
                'Seed': traj['seed']
            })
        
        # SGD baseline analysis
        for traj in sgd_baseline_trajectories[complexity]:
            roughness = compute_landscape_roughness(traj['data'])
            curvature_measures = compute_curvature_measures(traj['data'])
            
            # Handle different column names for accuracy
            if 'query_accuracy' in traj['data'].columns:
                final_acc = traj['data']['query_accuracy'].iloc[-1]
            elif 'val_accuracy' in traj['data'].columns:
                final_acc = traj['data']['val_accuracy'].iloc[-1]
            else:
                final_acc = 0.5
            
            analysis_data.append({
                'Complexity': complexity,
                'Method': 'SGD',
                'Roughness': roughness,
                'Final_Accuracy': final_acc,
                'Hessian_Trace': curvature_measures['hessian_trace'],
                'Mean_Curvature': curvature_measures['mean_curvature'],
                'Gaussian_Curvature': curvature_measures['gaussian_curvature'],
                'Seed': traj['seed']
            })
    
    df = pd.DataFrame(analysis_data)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Roughness comparison by complexity
    ax1 = axes[0, 0]
    roughness_data = df.pivot_table(values='Roughness', index='Complexity', columns='Method', aggfunc='mean')
    roughness_data.plot(kind='bar', ax=ax1, color=['red', 'green'], width=0.7)
    ax1.set_title('Loss Landscape Roughness by Complexity', fontweight='bold')
    ax1.set_ylabel('Roughness (Normalized Std/Mean)')
    ax1.set_xlabel('Complexity')
    ax1.legend(title='Method')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Performance vs Roughness scatter
    ax2 = axes[0, 1]
    for method, color in [('Meta-SGD', 'green'), ('SGD', 'red')]:
        method_data = df[df['Method'] == method]
        ax2.scatter(method_data['Roughness'], method_data['Final_Accuracy'], 
                   c=color, alpha=0.7, label=method, s=80)
    
    ax2.set_xlabel('Loss Landscape Roughness')
    ax2.set_ylabel('Final Accuracy')
    ax2.set_title('Performance vs Landscape Roughness', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hessian Trace (curvature)
    ax3 = axes[0, 2]
    hessian_data = df.pivot_table(values='Hessian_Trace', index='Complexity', columns='Method', aggfunc='mean')
    hessian_data.plot(kind='bar', ax=ax3, color=['red', 'green'], width=0.7)
    ax3.set_title('Hessian Trace (Curvature) by Complexity', fontweight='bold')
    ax3.set_ylabel('Hessian Trace')
    ax3.set_xlabel('Complexity')
    ax3.legend(title='Method')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Mean Curvature Analysis
    ax4 = axes[1, 0]
    mean_curvature_data = df.pivot_table(values='Mean_Curvature', index='Complexity', columns='Method', aggfunc='mean')
    mean_curvature_data.plot(kind='bar', ax=ax4, color=['red', 'green'], width=0.7)
    ax4.set_title('Mean Curvature by Complexity', fontweight='bold')
    ax4.set_ylabel('Mean Curvature')
    ax4.set_xlabel('Complexity')
    ax4.legend(title='Method')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Gaussian Curvature Analysis
    ax5 = axes[1, 1]
    gaussian_curvature_data = df.pivot_table(values='Gaussian_Curvature', index='Complexity', columns='Method', aggfunc='mean')
    gaussian_curvature_data.plot(kind='bar', ax=ax5, color=['red', 'green'], width=0.7)
    ax5.set_title('Gaussian Curvature by Complexity', fontweight='bold')
    ax5.set_ylabel('Gaussian Curvature')
    ax5.set_xlabel('Complexity')
    ax5.legend(title='Method')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Curvature vs Performance
    ax6 = axes[1, 2]
    for method, color in [('Meta-SGD', 'green'), ('SGD', 'red')]:
        method_data = df[df['Method'] == method]
        ax6.scatter(method_data['Hessian_Trace'], method_data['Final_Accuracy'], 
                   c=color, alpha=0.7, label=method, s=80)
    
    ax6.set_xlabel('Hessian Trace (Curvature)')
    ax6.set_ylabel('Final Accuracy')
    ax6.set_title('Performance vs Curvature', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Loss Landscape and Curvature Analysis: SGD vs Meta-SGD', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sgd_vs_meta_sgd_comprehensive_analysis.svg', dpi=300, bbox_inches='tight')
    plt.savefig('sgd_vs_meta_sgd_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("COMPREHENSIVE LANDSCAPE AND CURVATURE ANALYSIS SUMMARY")
    print("="*80)
    
    summary_stats = []
    
    for complexity in complexities:
        meta_data = df[(df['Complexity'] == complexity) & (df['Method'] == 'Meta-SGD')]
        sgd_data = df[(df['Complexity'] == complexity) & (df['Method'] == 'SGD')]
        
        if not meta_data.empty and not sgd_data.empty:
            # Performance comparison
            meta_acc = meta_data['Final_Accuracy'].mean()
            sgd_acc = sgd_data['Final_Accuracy'].mean()
            acc_improvement = (meta_acc - sgd_acc) / sgd_acc * 100
            
            # Roughness comparison
            meta_roughness = meta_data['Roughness'].mean()
            sgd_roughness = sgd_data['Roughness'].mean()
            roughness_reduction = (sgd_roughness - meta_roughness) / sgd_roughness * 100
            
            # Curvature comparison
            meta_hessian = meta_data['Hessian_Trace'].mean()
            sgd_hessian = sgd_data['Hessian_Trace'].mean()
            hessian_reduction = (sgd_hessian - meta_hessian) / sgd_hessian * 100
            
            print(f"\n{complexity}:")
            print(f"  Performance: {sgd_acc:.3f} → {meta_acc:.3f} (+{acc_improvement:.1f}%)")
            print(f"  Roughness: {sgd_roughness:.4f} → {meta_roughness:.4f} (-{roughness_reduction:.1f}%)")
            print(f"  Hessian Trace: {sgd_hessian:.4f} → {meta_hessian:.4f} (-{hessian_reduction:.1f}%)")
            
            summary_stats.append({
                'Complexity': complexity,
                'Acc_Improvement': acc_improvement,
                'Roughness_Reduction': roughness_reduction,
                'Hessian_Reduction': hessian_reduction
            })
    
    # Save detailed report
    with open('sgd_vs_meta_sgd_comprehensive_report.md', 'w') as f:
        f.write("# Comprehensive Loss Landscape and Curvature Analysis Report\n\n")
        f.write("## Summary\n")
        f.write("This analysis compares loss landscape properties and differential geometry measures between SGD baseline and Meta-SGD approaches.\n\n")
        
        f.write("## Results by Complexity\n")
        for stat in summary_stats:
            f.write(f"### {stat['Complexity']}\n")
            f.write(f"- **Performance Improvement**: {stat['Acc_Improvement']:.1f}%\n")
            f.write(f"- **Roughness Reduction**: {stat['Roughness_Reduction']:.1f}%\n")
            f.write(f"- **Hessian Trace Reduction**: {stat['Hessian_Reduction']:.1f}%\n\n")
    
    print(f"\nDetailed report saved to: sgd_vs_meta_sgd_comprehensive_report.md")
    print(f"Figures saved to: sgd_vs_meta_sgd_comprehensive_analysis.svg/.png")

if __name__ == "__main__":
    analyze_landscape_comparison() 