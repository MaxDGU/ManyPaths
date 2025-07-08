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

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class Figure2Generator:
    def __init__(self, sgd_results_dir="results/baseline_sgd/baseline_run1", 
                 meta_sgd_results_dir="results", output_dir="figures"):
        self.sgd_results_dir = sgd_results_dir
        self.meta_sgd_results_dir = meta_sgd_results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.sgd_trajectories = {}
        self.meta_sgd_trajectories = {}
        
    def load_sgd_trajectories(self):
        """Load SGD baseline trajectory data."""
        print("Loading SGD baseline trajectories...")
        
        # Pattern to match SGD baseline files
        pattern = os.path.join(self.sgd_results_dir, "*baselinetrajectory.csv")
        files = glob.glob(pattern)
        
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                params = self._parse_sgd_filename(filename)
                
                if params:
                    df = pd.read_csv(file_path)
                    
                    # Group by complexity
                    complexity = self._get_complexity_label(params['features'], params['depth'])
                    
                    if complexity not in self.sgd_trajectories:
                        self.sgd_trajectories[complexity] = {}
                    
                    self.sgd_trajectories[complexity][params['seed']] = df
                    
            except Exception as e:
                print(f"Error loading SGD file {file_path}: {e}")
                
        print(f"Loaded SGD trajectories for {len(self.sgd_trajectories)} complexities")
        
    def load_meta_sgd_trajectories(self):
        """Load Meta-SGD trajectory data."""
        print("Loading Meta-SGD trajectories...")
        
        # Pattern to match Meta-SGD trajectory files
        pattern = os.path.join(self.meta_sgd_results_dir, "**/*trajectory.csv")
        files = glob.glob(pattern, recursive=True)
        
        # Dictionary to store the latest epoch for each configuration
        latest_files = {}
        
        for file_path in files:
            # Skip baseline files
            if 'baseline' in file_path:
                continue
                
            try:
                filename = os.path.basename(file_path)
                params = self._parse_meta_sgd_filename(filename)
                
                if params:
                    # Create a unique key for this configuration
                    config_key = (params['features'], params['depth'], 
                                params['order'], params['seed'])
                    
                    # Keep track of the latest epoch for each configuration
                    epoch = params.get('epoch', 0)
                    if config_key not in latest_files or epoch > latest_files[config_key][1]:
                        latest_files[config_key] = (file_path, epoch)
                        
            except Exception as e:
                print(f"Error parsing Meta-SGD filename {filename}: {e}")
        
        # Load the latest trajectory file for each configuration
        for config_key, (file_path, epoch) in latest_files.items():
            try:
                df = pd.read_csv(file_path)
                features, depth, order, seed = config_key
                
                complexity = self._get_complexity_label(features, depth)
                
                if complexity not in self.meta_sgd_trajectories:
                    self.meta_sgd_trajectories[complexity] = {}
                
                self.meta_sgd_trajectories[complexity][seed] = df
                    
            except Exception as e:
                print(f"Error loading Meta-SGD file {file_path}: {e}")
                
        print(f"Loaded Meta-SGD trajectories for {len(self.meta_sgd_trajectories)} complexities")
        
    def _parse_sgd_filename(self, filename):
        """Parse parameters from SGD baseline filename."""
        import re
        
        # Pattern: concept_mlp_14_bits_feats8_depth3_sgdsteps32_lr0.01_runbaseline_run1_seed1_baselinetrajectory.csv
        pattern = r"concept_mlp_\d+_bits_feats(\d+)_depth(\d+)_sgdsteps\d+_lr[\d.]+_runbaseline_run1_seed(\d+)_baselinetrajectory\.csv"
        match = re.match(pattern, filename)
        
        if match:
            return {
                'features': int(match.group(1)),
                'depth': int(match.group(2)),
                'seed': int(match.group(3))
            }
        
        return None
        
    def _parse_meta_sgd_filename(self, filename):
        """Parse parameters from Meta-SGD trajectory filename."""
        import re
        
        # Pattern for files with epoch suffix
        pattern_epoch = r"concept_mlp_\d+_bits_feats(\d+)_depth(\d+)_adapt(\d+)_(\w+)Ord_seed(\d+)_epoch_(\d+)_trajectory\.csv"
        match = re.match(pattern_epoch, filename)
        
        if match:
            return {
                'features': int(match.group(1)),
                'depth': int(match.group(2)),
                'adaptation_steps': int(match.group(3)),
                'order': match.group(4),
                'seed': int(match.group(5)),
                'epoch': int(match.group(6))
            }
        
        # Pattern for files without epoch suffix
        pattern_no_epoch = r"concept_mlp_\d+_bits_feats(\d+)_depth(\d+)_adapt(\d+)_(\w+)Ord_seed(\d+)_trajectory\.csv"
        match = re.match(pattern_no_epoch, filename)
        
        if match:
            return {
                'features': int(match.group(1)),
                'depth': int(match.group(2)),
                'adaptation_steps': int(match.group(3)),
                'order': match.group(4),
                'seed': int(match.group(5)),
                'epoch': 0
            }
        
        return None
        
    def _get_complexity_label(self, features, depth):
        """Get complexity label based on features and depth."""
        if features == 8 and depth == 3:
            return "Simple (F8D3)"
        elif features == 8 and depth == 5:
            return "Medium (F8D5)"
        elif features == 16 and depth == 3:
            return "Medium (F16D3)"
        elif features == 32 and depth == 3:
            return "Complex (F32D3)"
        else:
            return f"F{features}D{depth}"
            
    def prepare_sgd_trajectory_data(self, complexity, max_tasks=10000):
        """Prepare SGD trajectory data with running averages and confidence intervals."""
        if complexity not in self.sgd_trajectories:
            return None, None, None
            
        seeds_data = self.sgd_trajectories[complexity]
        
        # Find the minimum length across all seeds
        min_length = min(len(df) for df in seeds_data.values())
        min_length = min(min_length, max_tasks)
        
        # Create arrays for all seeds
        accuracies = []
        for seed, df in seeds_data.items():
            if len(df) >= min_length:
                acc = df['query_accuracy'].iloc[:min_length].values
                accuracies.append(acc)
        
        if not accuracies:
            return None, None, None
            
        accuracies = np.array(accuracies)
        
        # Compute running averages
        running_means = []
        running_stds = []
        
        window_size = min(100, min_length // 20)  # Adaptive window size
        
        for i in range(min_length):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            
            # Running average across seeds and time
            window_means = []
            for seed_idx in range(len(accuracies)):
                window_mean = np.mean(accuracies[seed_idx, start_idx:end_idx])
                window_means.append(window_mean)
            
            running_means.append(np.mean(window_means))
            running_stds.append(np.std(window_means))
        
        task_indices = np.arange(min_length)
        
        return task_indices, np.array(running_means), np.array(running_stds)
        
    def prepare_meta_sgd_trajectory_data(self, complexity):
        """Prepare Meta-SGD trajectory data."""
        if complexity not in self.meta_sgd_trajectories:
            return None, None, None
            
        seeds_data = self.meta_sgd_trajectories[complexity]
        
        # Find the minimum length across all seeds
        min_length = min(len(df) for df in seeds_data.values())
        
        # Create arrays for all seeds
        accuracies = []
        steps = []
        
        for seed, df in seeds_data.items():
            if len(df) >= min_length:
                acc = df['val_accuracy'].iloc[:min_length].values
                step = df['log_step'].iloc[:min_length].values
                accuracies.append(acc)
                steps.append(step)
        
        if not accuracies:
            return None, None, None
            
        # Use the first seed's steps (assuming they're the same)
        steps = steps[0]
        accuracies = np.array(accuracies)
        
        # Compute mean and std across seeds
        mean_acc = np.mean(accuracies, axis=0)
        std_acc = np.std(accuracies, axis=0)
        
        return steps, mean_acc, std_acc
        
    def plot_figure_2(self):
        """Generate Figure 2 with Meta-SGD trajectories and SGD trajectories with error bars."""
        print("Generating Figure 2...")
        
        # Define complexities to plot
        complexities = ["Simple (F8D3)", "Medium (F8D5)", "Complex (F32D3)"]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Learning Trajectories: Meta-SGD vs SGD Baseline', fontsize=16, fontweight='bold')
        
        colors = {
            'Meta-SGD': '#2E8B57',  # Sea green
            'SGD': '#C7322F'        # Red
        }
        
        for i, complexity in enumerate(complexities):
            ax = axes[i]
            
            # Plot Meta-SGD trajectories
            meta_steps, meta_mean, meta_std = self.prepare_meta_sgd_trajectory_data(complexity)
            if meta_steps is not None:
                ax.plot(meta_steps, meta_mean, color=colors['Meta-SGD'], 
                       linewidth=2, label='Meta-SGD', alpha=0.9)
                ax.fill_between(meta_steps, meta_mean - meta_std, meta_mean + meta_std, 
                               color=colors['Meta-SGD'], alpha=0.3)
            
            # Plot SGD trajectories
            sgd_tasks, sgd_mean, sgd_std = self.prepare_sgd_trajectory_data(complexity)
            if sgd_tasks is not None:
                ax.plot(sgd_tasks, sgd_mean, color=colors['SGD'], 
                       linewidth=2, label='SGD Baseline', alpha=0.9)
                ax.fill_between(sgd_tasks, sgd_mean - sgd_std, sgd_mean + sgd_std, 
                               color=colors['SGD'], alpha=0.3)
            
            # Formatting
            ax.set_title(f'{complexity}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Progress', fontsize=12)
            if i == 0:
                ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_ylim(0.4, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add performance annotations
            if sgd_mean is not None and meta_mean is not None:
                final_sgd = sgd_mean[-1]
                final_meta = meta_mean[-1]
                improvement = final_meta - final_sgd
                
                # Add text box with performance summary
                textstr = f'SGD: {final_sgd:.3f}\nMeta-SGD: {final_meta:.3f}\nΔ: {improvement:+.3f}'
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'figure_2_meta_sgd_vs_sgd_trajectories.svg'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'figure_2_meta_sgd_vs_sgd_trajectories.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Figure 2 saved to {self.output_dir}/figure_2_meta_sgd_vs_sgd_trajectories.svg")
        
    def plot_detailed_trajectory_comparison(self):
        """Generate a more detailed trajectory comparison."""
        print("Generating detailed trajectory comparison...")
        
        # Define complexities to plot
        complexities = ["Simple (F8D3)", "Medium (F8D5)", "Complex (F32D3)"]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detailed Learning Trajectories: Meta-SGD vs SGD Baseline', fontsize=16, fontweight='bold')
        
        colors = {
            'Meta-SGD': '#2E8B57',  # Sea green
            'SGD': '#C7322F'        # Red
        }
        
        for i, complexity in enumerate(complexities):
            # Top row: Accuracy trajectories
            ax_acc = axes[0, i]
            
            # Plot Meta-SGD trajectories
            meta_steps, meta_mean, meta_std = self.prepare_meta_sgd_trajectory_data(complexity)
            if meta_steps is not None:
                ax_acc.plot(meta_steps, meta_mean, color=colors['Meta-SGD'], 
                           linewidth=2, label='Meta-SGD', alpha=0.9)
                ax_acc.fill_between(meta_steps, meta_mean - meta_std, meta_mean + meta_std, 
                                   color=colors['Meta-SGD'], alpha=0.3)
            
            # Plot SGD trajectories
            sgd_tasks, sgd_mean, sgd_std = self.prepare_sgd_trajectory_data(complexity)
            if sgd_tasks is not None:
                ax_acc.plot(sgd_tasks, sgd_mean, color=colors['SGD'], 
                           linewidth=2, label='SGD Baseline', alpha=0.9)
                ax_acc.fill_between(sgd_tasks, sgd_mean - sgd_std, sgd_mean + sgd_std, 
                                   color=colors['SGD'], alpha=0.3)
            
            ax_acc.set_title(f'{complexity} - Accuracy', fontsize=12, fontweight='bold')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.set_ylim(0.4, 1.0)
            ax_acc.grid(True, alpha=0.3)
            ax_acc.legend()
            
            # Bottom row: Show individual seed trajectories
            ax_seeds = axes[1, i]
            
            # Plot individual SGD seeds
            if complexity in self.sgd_trajectories:
                for seed, df in self.sgd_trajectories[complexity].items():
                    tasks = np.arange(len(df))
                    running_avg = pd.Series(df['query_accuracy']).rolling(window=100, min_periods=1).mean()
                    ax_seeds.plot(tasks, running_avg, color=colors['SGD'], 
                                 alpha=0.6, linewidth=1, label=f'SGD Seed {seed}' if seed == 1 else "")
            
            # Plot individual Meta-SGD seeds
            if complexity in self.meta_sgd_trajectories:
                for seed, df in self.meta_sgd_trajectories[complexity].items():
                    ax_seeds.plot(df['log_step'], df['val_accuracy'], color=colors['Meta-SGD'], 
                                 alpha=0.6, linewidth=1, label=f'Meta-SGD Seed {seed}' if seed == 0 else "")
            
            ax_seeds.set_title(f'{complexity} - Individual Seeds', fontsize=12, fontweight='bold')
            ax_seeds.set_xlabel('Training Progress')
            ax_seeds.set_ylabel('Accuracy')
            ax_seeds.set_ylim(0.4, 1.0)
            ax_seeds.grid(True, alpha=0.3)
            if i == 0:
                ax_seeds.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'detailed_trajectory_comparison.svg'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'detailed_trajectory_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Detailed comparison saved to {self.output_dir}/detailed_trajectory_comparison.svg")
        
    def generate_summary_stats(self):
        """Generate summary statistics for the trajectory comparison."""
        print("Generating summary statistics...")
        
        summary_data = []
        
        for complexity in ["Simple (F8D3)", "Medium (F8D5)", "Complex (F32D3)"]:
            # SGD statistics
            if complexity in self.sgd_trajectories:
                sgd_final_accs = []
                for seed, df in self.sgd_trajectories[complexity].items():
                    sgd_final_accs.append(df['query_accuracy'].iloc[-1])
                
                sgd_mean = np.mean(sgd_final_accs)
                sgd_std = np.std(sgd_final_accs)
                
                summary_data.append({
                    'Complexity': complexity,
                    'Method': 'SGD',
                    'Final_Accuracy': sgd_mean,
                    'Std_Dev': sgd_std,
                    'Seeds': len(sgd_final_accs)
                })
            
            # Meta-SGD statistics
            if complexity in self.meta_sgd_trajectories:
                meta_final_accs = []
                for seed, df in self.meta_sgd_trajectories[complexity].items():
                    meta_final_accs.append(df['val_accuracy'].iloc[-1])
                
                meta_mean = np.mean(meta_final_accs)
                meta_std = np.std(meta_final_accs)
                
                summary_data.append({
                    'Complexity': complexity,
                    'Method': 'Meta-SGD',
                    'Final_Accuracy': meta_mean,
                    'Std_Dev': meta_std,
                    'Seeds': len(meta_final_accs)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_df.to_csv(os.path.join(self.output_dir, 'trajectory_comparison_summary.csv'), index=False)
        
        # Print summary
        print("\nTrajectory Comparison Summary:")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        
        # Compute improvements
        print("\nMeta-Learning Improvements:")
        print("=" * 30)
        
        for complexity in ["Simple (F8D3)", "Medium (F8D5)", "Complex (F32D3)"]:
            sgd_row = summary_df[(summary_df['Complexity'] == complexity) & (summary_df['Method'] == 'SGD')]
            meta_row = summary_df[(summary_df['Complexity'] == complexity) & (summary_df['Method'] == 'Meta-SGD')]
            
            if not sgd_row.empty and not meta_row.empty:
                sgd_acc = sgd_row['Final_Accuracy'].iloc[0]
                meta_acc = meta_row['Final_Accuracy'].iloc[0]
                improvement = meta_acc - sgd_acc
                improvement_pct = (improvement / sgd_acc) * 100
                
                print(f"{complexity}: {improvement:+.4f} ({improvement_pct:+.1f}%)")

def main():
    """Main function to generate Figure 2."""
    parser = argparse.ArgumentParser(description='Generate Figure 2: Meta-SGD vs SGD Trajectories')
    parser.add_argument('--sgd-dir', default='results/baseline_sgd/baseline_run1',
                       help='Directory containing SGD baseline results')
    parser.add_argument('--meta-sgd-dir', default='results',
                       help='Directory containing Meta-SGD results')
    parser.add_argument('--output-dir', default='figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Create generator
    generator = Figure2Generator(args.sgd_dir, args.meta_sgd_dir, args.output_dir)
    
    # Load data
    generator.load_sgd_trajectories()
    generator.load_meta_sgd_trajectories()
    
    # Generate figures
    generator.plot_figure_2()
    generator.plot_detailed_trajectory_comparison()
    generator.generate_summary_stats()
    
    print(f"Figure 2 generation complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 