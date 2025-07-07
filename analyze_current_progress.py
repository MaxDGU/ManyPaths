#!/usr/bin/env python3
"""
Quick Analysis of Current Experimental Progress

Analyzes whatever trajectory data is currently available to:
1. Show experimental progress
2. Generate preliminary results  
3. Identify which experiments are complete vs running
4. Preview the analysis pipeline outputs
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

def find_trajectory_files(results_dir="results"):
    """Find all trajectory files and parse their metadata."""
    print(f"🔍 Searching for trajectory files in: {results_dir}")
    
    pattern = os.path.join(results_dir, "*_trajectory.csv")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        # Try subdirectories
        pattern = os.path.join(results_dir, "**/*_trajectory.csv")
        files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} trajectory files")
    
    trajectory_info = []
    
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            
            # Parse filename for metadata
            metadata = parse_trajectory_filename(filename)
            if metadata:
                # Get file stats
                stat = os.stat(file_path)
                metadata.update({
                    'file_path': file_path,
                    'file_size': stat.st_size,
                    'modified_time': datetime.fromtimestamp(stat.st_mtime),
                    'is_intermediate': 'epoch_' in filename
                })
                
                # Load basic trajectory info
                try:
                    df = pd.read_csv(file_path)
                    metadata.update({
                        'max_log_step': df['log_step'].max() if 'log_step' in df.columns else 0,
                        'final_accuracy': df['val_accuracy'].iloc[-1] if 'val_accuracy' in df.columns and len(df) > 0 else None,
                        'converged_70': (df['val_accuracy'] >= 0.7).any() if 'val_accuracy' in df.columns else False
                    })
                except:
                    metadata.update({
                        'max_log_step': 0,
                        'final_accuracy': None,
                        'converged_70': False
                    })
                
                trajectory_info.append(metadata)
                
        except Exception as e:
            print(f"Warning: Could not parse {filename}: {e}")
    
    return pd.DataFrame(trajectory_info)

def parse_trajectory_filename(filename):
    """Parse trajectory filename to extract experimental parameters."""
    # Pattern: concept_mlp_14_bits_feats{F}_depth{D}_adapt{K}_{ORDER}_seed{S}_trajectory.csv
    # or: concept_mlp_14_bits_feats{F}_depth{D}_adapt{K}_{ORDER}_seed{S}_epoch_{E}_trajectory.csv
    
    patterns = [
        r"concept_mlp_\d+_bits_feats(\d+)_depth(\d+)_adapt(\d+)_(\w+)Ord_seed(\d+)_epoch_(\d+)_trajectory\.csv",
        r"concept_mlp_\d+_bits_feats(\d+)_depth(\d+)_adapt(\d+)_(\w+)Ord_seed(\d+)_trajectory\.csv"
    ]
    
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            if len(match.groups()) == 6:  # With epoch
                features, depth, adapt_steps, order, seed, epoch = match.groups()
                epoch = int(epoch)
            else:  # Without epoch
                features, depth, adapt_steps, order, seed = match.groups()
                epoch = None
            
            return {
                'features': int(features),
                'depth': int(depth),
                'adaptation_steps': int(adapt_steps),
                'order': order,
                'seed': int(seed),
                'epoch': epoch,
                'config': f"F{features}_D{depth}",
                'method': f"K{adapt_steps}_{order}Ord"
            }
    
    return None

def analyze_experimental_progress(trajectory_df):
    """Analyze which experiments are complete and their progress."""
    print("\n📊 EXPERIMENTAL PROGRESS ANALYSIS")
    print("=" * 50)
    
    if trajectory_df.empty:
        print("❌ No trajectory files found!")
        return
    
    # Group by configuration
    configs = trajectory_df.groupby(['features', 'depth', 'adaptation_steps', 'seed']).agg({
        'max_log_step': 'max',
        'final_accuracy': 'last',
        'converged_70': 'any',
        'modified_time': 'max',
        'is_intermediate': 'any'
    }).reset_index()
    
    print(f"Total unique configurations found: {len(configs)}")
    print(f"Configurations with 70%+ accuracy: {configs['converged_70'].sum()}")
    print(f"Configurations still running: {configs['is_intermediate'].sum()}")
    
    # Analyze by complexity
    print("\n🧠 BY COMPLEXITY LEVEL:")
    for features in sorted(configs['features'].unique()):
        subset = configs[configs['features'] == features]
        print(f"  F{features}: {len(subset)} experiments, {subset['converged_70'].sum()} converged")
    
    # Analyze by adaptation steps
    print("\n🔄 BY ADAPTATION STEPS:")
    for k in sorted(configs['adaptation_steps'].unique()):
        subset = configs[configs['adaptation_steps'] == k]
        print(f"  K={k}: {len(subset)} experiments, {subset['converged_70'].sum()} converged")
    
    # Recent activity
    recent_configs = configs[configs['modified_time'] > datetime.now().replace(hour=0, minute=0, second=0)]
    print(f"\n🕐 RECENT ACTIVITY (today): {len(recent_configs)} experiments")
    
    return configs

def plot_preliminary_results(trajectory_df, output_dir="preliminary_analysis"):
    """Generate preliminary analysis plots."""
    print(f"\n📈 Generating preliminary plots in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    if trajectory_df.empty:
        print("❌ No data for plotting")
        return
    
    # Load actual trajectory data for converged experiments
    converged_trajectories = []
    
    for _, row in trajectory_df.iterrows():
        if row['converged_70'] and not row['is_intermediate']:
            try:
                df = pd.read_csv(row['file_path'])
                df['features'] = row['features']
                df['adaptation_steps'] = row['adaptation_steps']
                df['seed'] = row['seed']
                df['config'] = row['config']
                df['method'] = row['method']
                converged_trajectories.append(df)
            except:
                continue
    
    if not converged_trajectories:
        print("❌ No converged trajectories found for plotting")
        return
    
    combined_df = pd.concat(converged_trajectories, ignore_index=True)
    print(f"✅ Loaded {len(converged_trajectories)} converged trajectories")
    
    # Plot 1: Learning curves by complexity
    plt.figure(figsize=(12, 8))
    
    for config in combined_df['config'].unique():
        subset = combined_df[combined_df['config'] == config]
        
        # Group by method and plot mean trajectory
        for method in subset['method'].unique():
            method_data = subset[subset['method'] == method]
            
            # Compute mean trajectory across seeds
            mean_traj = method_data.groupby('log_step')['val_accuracy'].mean()
            episodes = mean_traj.index * 1000  # Convert to episodes
            
            plt.plot(episodes, mean_traj, label=f"{config}_{method}", linewidth=2)
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Validation Accuracy')
    plt.title('Learning Curves: Current Experimental Progress')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'learning_curves_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Learning curves: {plot_path}")
    plt.show()
    
    # Plot 2: Sample efficiency preview (if K=1 and K=10 data available)
    k1_data = combined_df[combined_df['adaptation_steps'] == 1]
    k10_data = combined_df[combined_df['adaptation_steps'] == 10]
    
    if not k1_data.empty and not k10_data.empty:
        plt.figure(figsize=(10, 6))
        
        # Sample efficiency at 70% threshold
        threshold = 0.7
        efficiency_results = []
        
        for config in combined_df['config'].unique():
            for k_val, k_data in [('K=1', k1_data), ('K=10', k10_data)]:
                subset = k_data[k_data['config'] == config]
                
                if not subset.empty:
                    # Find episodes to reach threshold
                    for seed in subset['seed'].unique():
                        seed_data = subset[subset['seed'] == seed]
                        threshold_reached = seed_data[seed_data['val_accuracy'] >= threshold]
                        
                        if not threshold_reached.empty:
                            episodes_to_threshold = threshold_reached['log_step'].iloc[0] * 1000
                            samples_to_threshold = episodes_to_threshold * 10  # 10 samples per episode
                            
                            efficiency_results.append({
                                'config': config,
                                'method': k_val,
                                'samples': samples_to_threshold,
                                'seed': seed
                            })
        
        if efficiency_results:
            eff_df = pd.DataFrame(efficiency_results)
            
            # Plot sample efficiency comparison
            sns.boxplot(data=eff_df, x='config', y='samples', hue='method')
            plt.yscale('log')
            plt.ylabel('Samples to 70% Accuracy')
            plt.title('Sample Efficiency Preview: K=1 vs K=10')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, 'sample_efficiency_preview.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Sample efficiency preview: {plot_path}")
            plt.show()

def generate_progress_report(trajectory_df, configs_df, output_dir):
    """Generate a progress report."""
    report_path = os.path.join(output_dir, 'progress_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Current Experimental Progress Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n")
        f.write(f"- **Total trajectory files**: {len(trajectory_df)}\n")
        f.write(f"- **Unique configurations**: {len(configs_df)}\n")
        f.write(f"- **Converged (70%+ accuracy)**: {configs_df['converged_70'].sum()}\n")
        f.write(f"- **Currently running**: {configs_df['is_intermediate'].sum()}\n\n")
        
        f.write("## Configuration Status\n\n")
        
        # Status by configuration
        for _, config in configs_df.iterrows():
            status = "✅ Converged" if config['converged_70'] else "🔄 Running" if config['is_intermediate'] else "❌ Failed/Incomplete"
            f.write(f"- **F{config['features']}_D{config['depth']}_K{config['adaptation_steps']}_S{config['seed']}**: {status}\n")
            f.write(f"  - Progress: {config['max_log_step']} log steps\n")
            f.write(f"  - Final accuracy: {config['final_accuracy']:.3f}\n" if config['final_accuracy'] else "  - Final accuracy: N/A\n")
            f.write(f"  - Last updated: {config['modified_time'].strftime('%H:%M:%S')}\n\n")
        
        f.write("## Next Steps\n")
        f.write("1. Monitor running experiments\n")
        f.write("2. Run full analysis pipeline when more data is available\n")
        f.write("3. Start preliminary analysis on converged experiments\n")
    
    print(f"✅ Progress report saved to: {report_path}")

def main():
    """Main analysis function."""
    print("🚀 CURRENT PROGRESS ANALYSIS")
    print("=" * 50)
    
    # Find trajectory files
    trajectory_df = find_trajectory_files()
    
    if trajectory_df.empty:
        print("❌ No trajectory files found. Check your results directory.")
        return
    
    # Analyze progress
    configs_df = analyze_experimental_progress(trajectory_df)
    
    # Generate plots if we have data
    output_dir = "preliminary_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_preliminary_results(trajectory_df, output_dir)
    
    # Generate progress report
    generate_progress_report(trajectory_df, configs_df, output_dir)
    
    print(f"\n✅ Preliminary analysis completed!")
    print(f"📁 Results saved to: {output_dir}/")
    print("📋 Check progress_report.md for detailed status")

if __name__ == "__main__":
    main() 