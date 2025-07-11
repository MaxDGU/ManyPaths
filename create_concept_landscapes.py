#!/usr/bin/env python3
"""
Create Loss Landscapes by Concept Type

This script creates loss landscape visualizations for each concept complexity type
using the trajectory data from landscape logging experiments.

Usage:
    python create_concept_landscapes.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import re
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_landscape_files():
    """Load all landscape trajectory CSV files"""
    # Look for landscape trajectory files
    patterns = [
        "della_analysis_results/*landscape_trajectory.csv",
        "*landscape_trajectory.csv"
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    print(f"Found {len(files)} landscape trajectory files")
    
    datasets = {}
    for file in files:
        try:
            df = pd.read_csv(file)
            
            # Extract experiment info from filename
            # Example: concept_mlp_14_bits_feats32_depth7_adapt10_2ndOrd_seed3_landscape_trajectory.csv
            match = re.search(r'feats(\d+)_depth(\d+).*seed(\d+)', file)
            
            if match:
                features, depth, seed = match.groups()
                config = f"F{features}D{depth}"
                key = f"{config}_seed{seed}"
                
                datasets[key] = {
                    'data': df,
                    'config': config,
                    'features': int(features),
                    'depth': int(depth),
                    'seed': int(seed),
                    'file': file
                }
                print(f"✅ Loaded {key}: {len(df)} steps")
            else:
                print(f"⚠️  Could not parse filename: {file}")
                
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
    
    return datasets

def create_concept_complexity_landscapes(datasets):
    """Create loss landscapes for each concept complexity level"""
    
    # Group datasets by concept complexity
    complexity_groups = {}
    for key, data in datasets.items():
        config = data['config']
        if config not in complexity_groups:
            complexity_groups[config] = []
        complexity_groups[config].append(data)
    
    print(f"Found concept complexity types: {list(complexity_groups.keys())}")
    
    # Create figure with subplots for each complexity type
    n_configs = len(complexity_groups)
    cols = min(3, n_configs)
    rows = (n_configs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_configs == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Loss Landscapes by Concept Complexity', fontsize=16)
    
    for i, (config, config_data) in enumerate(complexity_groups.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Combine all seeds for this configuration
        all_steps = []
        all_losses = []
        all_param_norms = []
        
        for data in config_data:
            df = data['data']
            if 'loss' in df.columns and 'theta_norm' in df.columns:
                all_steps.extend(df['step'].values)
                all_losses.extend(df['loss'].values)
                all_param_norms.extend(df['theta_norm'].values)
        
        if all_losses:
            # Create 2D loss landscape using parameter norm and training progress
            # Normalize step to [0, 1] for better visualization
            max_step = max(all_steps)
            normalized_steps = [s / max_step for s in all_steps]
            
            # Create scatter plot colored by loss
            scatter = ax.scatter(all_param_norms, normalized_steps, 
                               c=all_losses, cmap='viridis_r', 
                               alpha=0.6, s=20)
            
            ax.set_xlabel('Parameter Norm')
            ax.set_ylabel('Training Progress (normalized)')
            ax.set_title(f'{config}\n{data["features"]} features, depth {data["depth"]}')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Loss')
    
    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/concept_complexity_landscapes.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/concept_complexity_landscapes.png', dpi=300, bbox_inches='tight')
    print("Generated: figures/concept_complexity_landscapes.pdf")
    print("Generated: figures/concept_complexity_landscapes.png")

def create_3d_loss_landscapes(datasets):
    """Create 3D loss landscapes for each concept type"""
    
    # Group datasets by concept complexity
    complexity_groups = {}
    for key, data in datasets.items():
        config = data['config']
        if config not in complexity_groups:
            complexity_groups[config] = []
        complexity_groups[config].append(data)
    
    # Create 3D plots for each complexity type
    for config, config_data in complexity_groups.items():
        fig = plt.figure(figsize=(12, 8))
        
        # Create two subplots: one 3D surface and one trajectory
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        
        # Combine data from all seeds
        all_steps = []
        all_losses = []
        all_param_norms = []
        all_geodesic = []
        
        for data in config_data:
            df = data['data']
            if all(col in df.columns for col in ['loss', 'theta_norm', 'geodesic_length_from_start']):
                all_steps.extend(df['step'].values)
                all_losses.extend(df['loss'].values)
                all_param_norms.extend(df['theta_norm'].values)
                all_geodesic.extend(df['geodesic_length_from_start'].values)
        
        if all_losses:
            # Normalize step to [0, 1]
            max_step = max(all_steps)
            normalized_steps = [s / max_step for s in all_steps]
            
            # 3D surface plot
            ax1.scatter(all_param_norms, all_geodesic, all_losses, 
                       c=normalized_steps, cmap='plasma', alpha=0.6, s=15)
            ax1.set_xlabel('Parameter Norm')
            ax1.set_ylabel('Geodesic Length')
            ax1.set_zlabel('Loss')
            ax1.set_title(f'3D Loss Landscape: {config}')
            
            # 2D trajectory plot
            ax2.plot(all_param_norms, all_losses, 'b-', alpha=0.7, linewidth=1)
            ax2.scatter(all_param_norms, all_losses, c=normalized_steps, 
                       cmap='plasma', alpha=0.6, s=10)
            ax2.set_xlabel('Parameter Norm')
            ax2.set_ylabel('Loss')
            ax2.set_title(f'Parameter-Loss Trajectory: {config}')
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar for time progression
            scatter = ax2.scatter([], [], c=[], cmap='plasma')
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Training Progress')
        
        fig.suptitle(f'Loss Landscape Analysis: {config}\n{config_data[0]["features"]} features, depth {config_data[0]["depth"]}')
        plt.tight_layout()
        
        # Save individual plots for each config
        filename_base = f'figures/landscape_3d_{config.lower()}'
        plt.savefig(f'{filename_base}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
        print(f"Generated: {filename_base}.pdf")
        print(f"Generated: {filename_base}.png")
        plt.close()

def create_comparative_analysis(datasets):
    """Create comparative analysis across concept types"""
    
    # Group by complexity
    complexity_groups = {}
    for key, data in datasets.items():
        config = data['config']
        if config not in complexity_groups:
            complexity_groups[config] = []
        complexity_groups[config].append(data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparative Analysis Across Concept Types', fontsize=16)
    
    # 1. Final loss comparison
    ax = axes[0, 0]
    config_names = []
    final_losses = []
    
    for config, config_data in complexity_groups.items():
        for data in config_data:
            df = data['data']
            if 'loss' in df.columns and len(df) > 0:
                final_loss = df['loss'].iloc[-1]
                config_names.append(f"{config}_s{data['seed']}")
                final_losses.append(final_loss)
    
    if final_losses:
        ax.bar(range(len(final_losses)), final_losses)
        ax.set_xticks(range(len(config_names)))
        ax.set_xticklabels(config_names, rotation=45)
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Loss by Configuration')
        ax.grid(True, alpha=0.3)
    
    # 2. Parameter growth comparison
    ax = axes[0, 1]
    for config, config_data in complexity_groups.items():
        all_param_norms = []
        for data in config_data:
            df = data['data']
            if 'theta_norm' in df.columns:
                all_param_norms.extend(df['theta_norm'].values)
        
        if all_param_norms:
            # Plot parameter norm distribution
            ax.hist(all_param_norms, bins=30, alpha=0.6, label=config)
    
    ax.set_xlabel('Parameter Norm')
    ax.set_ylabel('Frequency')
    ax.set_title('Parameter Norm Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Geodesic length comparison
    ax = axes[1, 0]
    config_names = []
    final_geodesics = []
    
    for config, config_data in complexity_groups.items():
        for data in config_data:
            df = data['data']
            if 'geodesic_length_from_start' in df.columns and len(df) > 0:
                final_geodesic = df['geodesic_length_from_start'].iloc[-1]
                config_names.append(f"{config}_s{data['seed']}")
                final_geodesics.append(final_geodesic)
    
    if final_geodesics:
        ax.bar(range(len(final_geodesics)), final_geodesics)
        ax.set_xticks(range(len(config_names)))
        ax.set_xticklabels(config_names, rotation=45)
        ax.set_ylabel('Final Geodesic Length')
        ax.set_title('Optimization Path Length by Configuration')
        ax.grid(True, alpha=0.3)
    
    # 4. Loss trajectory comparison
    ax = axes[1, 1]
    for config, config_data in complexity_groups.items():
        for data in config_data:
            df = data['data']
            if 'loss' in df.columns and 'step' in df.columns:
                # Subsample for cleaner plots
                subsample = df[::10] if len(df) > 1000 else df
                ax.plot(subsample['step'], subsample['loss'], 
                       alpha=0.6, linewidth=1, label=f"{config}_s{data['seed']}")
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Trajectories')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/concept_comparative_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/concept_comparative_analysis.png', dpi=300, bbox_inches='tight')
    print("Generated: figures/concept_comparative_analysis.pdf")
    print("Generated: figures/concept_comparative_analysis.png")

def main():
    """Main analysis pipeline"""
    
    # Ensure figures directory exists
    Path('figures').mkdir(exist_ok=True)
    
    print("🔍 Loading landscape trajectory files...")
    datasets = load_landscape_files()
    
    if not datasets:
        print("❌ No landscape files found!")
        print("Run: ./pull_landscape_results_from_della.sh")
        return
    
    print(f"\n📊 Analyzing {len(datasets)} experiments...")
    
    # Create different types of landscape visualizations
    print("\n🎨 Creating concept complexity landscapes...")
    create_concept_complexity_landscapes(datasets)
    
    print("\n🎨 Creating 3D loss landscapes...")
    create_3d_loss_landscapes(datasets)
    
    print("\n🎨 Creating comparative analysis...")
    create_comparative_analysis(datasets)
    
    print("\n✅ Analysis complete!")
    print("\nGenerated files:")
    print("  - figures/concept_complexity_landscapes.pdf")
    print("  - figures/landscape_3d_*.pdf (one per concept type)")
    print("  - figures/concept_comparative_analysis.pdf")

if __name__ == "__main__":
    main() 