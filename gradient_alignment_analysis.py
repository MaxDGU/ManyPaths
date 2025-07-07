#!/usr/bin/env python3
"""
Gradient Alignment Analysis for ManyPaths Camera-Ready Submission

This script analyzes gradient alignment patterns in existing trajectory data to provide
mechanistic insights into why K=10 adaptation steps work better than K=1, especially
for complex concepts.

Usage:
    python gradient_alignment_analysis.py --base_results_dir results/concept_multiseed \
        --compare_adaptation_steps --stratify_by_complexity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class GradientAlignmentAnalyzer:
    def __init__(self, base_results_dir, output_dir="figures"):
        self.base_results_dir = base_results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.trajectory_data = {}
        self.alignment_stats = {}
        
    def load_trajectory_data(self):
        """Load all trajectory CSV files containing gradient alignment data."""
        print("Loading trajectory data with gradient alignment...")
        
        # Pattern to match trajectory files
        pattern = os.path.join(self.base_results_dir, "*_trajectory.csv")
        files = glob.glob(pattern)
        
        # Dictionary to store the latest epoch for each configuration
        latest_files = {}
        
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                params = self._parse_filename(filename)
                
                if params:
                    # Create a unique key for this configuration
                    config_key = (params['features'], params['depth'], 
                                params['adaptation_steps'], params['order'], params['seed'])
                    
                    # Keep track of the latest epoch for each configuration
                    if config_key not in latest_files or params['epoch'] > latest_files[config_key][1]:
                        latest_files[config_key] = (file_path, params['epoch'])
                        
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
        
        print(f"Found {len(latest_files)} unique configurations")
        
        # Now load the latest trajectory file for each configuration
        loaded_count = 0
        for config_key, (file_path, epoch) in latest_files.items():
            try:
                df = pd.read_csv(file_path)
                
                # Only process files with gradient alignment data
                if 'grad_alignment' not in df.columns:
                    continue
                
                key = (config_key[0], config_key[1], config_key[2], config_key[3], config_key[4])
                self.trajectory_data[key] = df
                loaded_count += 1
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        print(f"Loaded {loaded_count} trajectory files with gradient alignment data")
        
    def _parse_filename(self, filename):
        """Parse parameters from trajectory filename."""
        import re
        
        # Pattern for: concept_mlp_14_bits_feats{F}_depth{D}_adapt{K}_{ORDER}_seed{S}_epoch_{E}_trajectory.csv
        pattern = r"concept_mlp_\d+_bits_feats(\d+)_depth(\d+)_adapt(\d+)_(\w+)Ord_seed(\d+)_epoch_(\d+)_trajectory\.csv"
        match = re.match(pattern, filename)
        
        if match:
            features = int(match.group(1))
            depth = int(match.group(2))
            adapt_steps = int(match.group(3))
            order = match.group(4)  # "1st" or "2nd"
            seed = int(match.group(5))
            epoch = int(match.group(6))
            
            return {
                'features': features,
                'depth': depth,
                'adaptation_steps': adapt_steps,
                'order': order,
                'seed': seed,
                'epoch': epoch
            }
        return None
    
    def compute_alignment_statistics(self):
        """Compute comprehensive alignment statistics."""
        print("Computing gradient alignment statistics...")
        
        stats_results = []
        
        for key, df in self.trajectory_data.items():
            features, depth, adapt_steps, order, seed = key
            
            # Clean gradient alignment data
            alignment = df['grad_alignment'].dropna()
            
            if len(alignment) == 0:
                continue
            
            # Compute various statistics
            stats_dict = {
                'features': features,
                'depth': depth,
                'adaptation_steps': adapt_steps,
                'order': order,
                'seed': seed,
                'method': f"MetaSGD_{order}Ord_K{adapt_steps}",
                'complexity': features * depth,  # Simple complexity metric
                
                # Basic statistics
                'mean_alignment': np.mean(alignment),
                'std_alignment': np.std(alignment),
                'max_alignment': np.max(alignment),
                'min_alignment': np.min(alignment),
                'final_alignment': alignment.iloc[-1] if len(alignment) > 0 else np.nan,
                
                # Convergence metrics
                'alignment_trend': self._compute_trend(alignment),
                'convergence_rate': self._compute_convergence_rate(alignment),
                'stability_index': self._compute_stability(alignment),
                
                # Performance correlation
                'final_accuracy': df['val_accuracy'].iloc[-1] if 'val_accuracy' in df.columns else np.nan,
                'accuracy_alignment_corr': self._compute_alignment_accuracy_correlation(df),
                
                # Training dynamics
                'episodes_to_positive_alignment': self._episodes_to_positive_alignment(alignment),
                'episodes_to_peak_alignment': self._episodes_to_peak_alignment(alignment)
            }
            
            stats_results.append(stats_dict)
        
        self.alignment_stats = pd.DataFrame(stats_results)
        return self.alignment_stats
    
    def _compute_trend(self, alignment_series):
        """Compute overall trend in alignment (slope of linear fit)."""
        if len(alignment_series) < 2:
            return np.nan
        
        x = np.arange(len(alignment_series))
        slope, _, _, _, _ = stats.linregress(x, alignment_series)
        return slope
    
    def _compute_convergence_rate(self, alignment_series):
        """Compute how quickly alignment converges to its final value."""
        if len(alignment_series) < 10:
            return np.nan
        
        # Use exponential decay model: alignment(t) = final + (initial - final) * exp(-rate * t)
        final_val = np.mean(alignment_series[-5:])  # Use last 5 points as "final"
        initial_val = alignment_series.iloc[0]
        
        if abs(final_val - initial_val) < 1e-6:
            return np.inf  # Already converged
        
        # Fit exponential decay
        x = np.arange(len(alignment_series))
        try:
            # Simple exponential fit
            normalized = (alignment_series - final_val) / (initial_val - final_val)
            normalized = np.clip(normalized, 1e-6, None)  # Avoid log(0)
            
            valid_idx = normalized > 0
            if np.sum(valid_idx) < 3:
                return np.nan
            
            slope, _, _, _, _ = stats.linregress(x[valid_idx], np.log(normalized[valid_idx]))
            return -slope  # Convergence rate (positive = faster convergence)
        except:
            return np.nan
    
    def _compute_stability(self, alignment_series):
        """Compute stability index (1 - coefficient of variation of later half)."""
        if len(alignment_series) < 10:
            return np.nan
        
        # Use second half of training
        second_half = alignment_series[len(alignment_series)//2:]
        
        if np.mean(second_half) == 0:
            return np.nan
        
        cv = np.std(second_half) / abs(np.mean(second_half))
        return 1 / (1 + cv)  # Higher values = more stable
    
    def _compute_alignment_accuracy_correlation(self, df):
        """Compute correlation between gradient alignment and validation accuracy."""
        if 'grad_alignment' not in df.columns or 'val_accuracy' not in df.columns:
            return np.nan
        
        alignment = df['grad_alignment'].dropna()
        accuracy = df['val_accuracy'].iloc[:len(alignment)]
        
        if len(alignment) < 5:
            return np.nan
        
        corr, _ = stats.pearsonr(alignment, accuracy)
        return corr
    
    def _episodes_to_positive_alignment(self, alignment_series):
        """Find number of episodes to reach positive gradient alignment."""
        positive_idx = alignment_series > 0
        if np.any(positive_idx):
            return np.where(positive_idx)[0][0] * 1000  # Convert to episodes
        return np.inf
    
    def _episodes_to_peak_alignment(self, alignment_series):
        """Find number of episodes to reach peak alignment."""
        if len(alignment_series) == 0:
            return np.inf
        
        peak_idx = np.argmax(alignment_series)
        return peak_idx * 1000  # Convert to episodes
    
    def plot_alignment_evolution(self):
        """Plot gradient alignment evolution patterns."""
        print("Plotting gradient alignment evolution...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Gradient Alignment Evolution Analysis', fontsize=16)
        
        # 1. Alignment trajectories by adaptation steps
        ax1 = axes[0, 0]
        self._plot_alignment_trajectories_by_k(ax1)
        ax1.set_title('Alignment Trajectories: K=1 vs K=10')
        
        # 2. Final alignment by complexity
        ax2 = axes[0, 1]
        self._plot_final_alignment_by_complexity(ax2)
        ax2.set_title('Final Alignment vs Complexity')
        
        # 3. Convergence rate comparison
        ax3 = axes[0, 2]
        self._plot_convergence_rates(ax3)
        ax3.set_title('Alignment Convergence Rates')
        
        # 4. Alignment-accuracy correlation
        ax4 = axes[1, 0]
        self._plot_alignment_accuracy_correlation(ax4)
        ax4.set_title('Alignment-Accuracy Correlation')
        
        # 5. Stability analysis
        ax5 = axes[1, 1]
        self._plot_stability_analysis(ax5)
        ax5.set_title('Alignment Stability')
        
        # 6. Episodes to positive alignment
        ax6 = axes[1, 2]
        self._plot_episodes_to_positive_alignment(ax6)
        ax6.set_title('Episodes to Positive Alignment')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'gradient_alignment_evolution_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved alignment evolution plot to {output_path}")
        plt.show()
    
    def _plot_alignment_trajectories_by_k(self, ax):
        """Plot alignment trajectories comparing K=1 vs K=10."""
        if self.alignment_stats.empty:
            return
        
        # Group trajectories by adaptation steps
        for adapt_steps in sorted(self.alignment_stats['adaptation_steps'].unique()):
            subset_keys = [(f, d, k, o, s) for f, d, k, o, s in self.trajectory_data.keys() 
                          if k == adapt_steps]
            
            all_alignments = []
            for key in subset_keys[:5]:  # Limit to first 5 seeds for clarity
                df = self.trajectory_data[key]
                alignment = df['grad_alignment'].dropna()
                if len(alignment) > 0:
                    # Normalize episode count
                    episodes = np.arange(len(alignment)) * 1000
                    all_alignments.append((episodes, alignment))
            
            if all_alignments:
                # Plot mean trajectory with confidence bands
                max_len = min(50, max(len(a[1]) for a in all_alignments))  # Limit length
                
                trajectories = []
                for episodes, alignment in all_alignments:
                    if len(alignment) >= max_len:
                        trajectories.append(alignment[:max_len])
                
                if trajectories:
                    trajectories = np.array(trajectories)
                    episodes_norm = np.arange(max_len) * 1000
                    
                    mean_traj = np.mean(trajectories, axis=0)
                    std_traj = np.std(trajectories, axis=0)
                    
                    label = f"K={adapt_steps}"
                    ax.plot(episodes_norm, mean_traj, label=label, linewidth=2)
                    ax.fill_between(episodes_norm, mean_traj - std_traj, mean_traj + std_traj, 
                                   alpha=0.3)
        
        ax.set_xlabel('Training Episodes')
        ax.set_ylabel('Gradient Alignment')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_final_alignment_by_complexity(self, ax):
        """Plot final alignment values by concept complexity."""
        if self.alignment_stats.empty:
            return
        
        # Create complexity-based plot
        plot_data = self.alignment_stats.groupby(['complexity', 'adaptation_steps']).agg({
            'final_alignment': ['mean', 'std']
        }).reset_index()
        
        plot_data.columns = ['complexity', 'adaptation_steps', 'mean_final', 'std_final']
        
        for adapt_steps in sorted(plot_data['adaptation_steps'].unique()):
            subset = plot_data[plot_data['adaptation_steps'] == adapt_steps]
            
            ax.errorbar(subset['complexity'], subset['mean_final'], 
                       yerr=subset['std_final'], 
                       label=f'K={adapt_steps}', marker='o', markersize=8, linewidth=2)
        
        ax.set_xlabel('Concept Complexity (Features × Depth)')
        ax.set_ylabel('Final Gradient Alignment')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_rates(self, ax):
        """Plot convergence rate comparison."""
        if self.alignment_stats.empty:
            return
        
        # Box plot of convergence rates
        plot_data = self.alignment_stats[self.alignment_stats['convergence_rate'].notna()]
        
        if not plot_data.empty:
            sns.boxplot(data=plot_data, x='adaptation_steps', y='convergence_rate', ax=ax)
            ax.set_xlabel('Adaptation Steps (K)')
            ax.set_ylabel('Convergence Rate')
    
    def _plot_alignment_accuracy_correlation(self, ax):
        """Plot alignment-accuracy correlation analysis."""
        if self.alignment_stats.empty:
            return
        
        # Scatter plot of correlation vs complexity
        plot_data = self.alignment_stats[self.alignment_stats['accuracy_alignment_corr'].notna()]
        
        if not plot_data.empty:
            for adapt_steps in sorted(plot_data['adaptation_steps'].unique()):
                subset = plot_data[plot_data['adaptation_steps'] == adapt_steps]
                
                ax.scatter(subset['complexity'], subset['accuracy_alignment_corr'], 
                          label=f'K={adapt_steps}', s=60, alpha=0.7)
        
        ax.set_xlabel('Concept Complexity')
        ax.set_ylabel('Alignment-Accuracy Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    def _plot_stability_analysis(self, ax):
        """Plot alignment stability analysis."""
        if self.alignment_stats.empty:
            return
        
        plot_data = self.alignment_stats[self.alignment_stats['stability_index'].notna()]
        
        if not plot_data.empty:
            sns.boxplot(data=plot_data, x='adaptation_steps', y='stability_index', ax=ax)
            ax.set_xlabel('Adaptation Steps (K)')
            ax.set_ylabel('Stability Index')
    
    def _plot_episodes_to_positive_alignment(self, ax):
        """Plot episodes needed to reach positive alignment."""
        if self.alignment_stats.empty:
            return
        
        plot_data = self.alignment_stats[
            (self.alignment_stats['episodes_to_positive_alignment'] != np.inf) &
            (self.alignment_stats['episodes_to_positive_alignment'].notna())
        ]
        
        if not plot_data.empty:
            for adapt_steps in sorted(plot_data['adaptation_steps'].unique()):
                subset = plot_data[plot_data['adaptation_steps'] == adapt_steps]
                
                ax.scatter(subset['complexity'], subset['episodes_to_positive_alignment'], 
                          label=f'K={adapt_steps}', s=60, alpha=0.7)
        
        ax.set_xlabel('Concept Complexity')
        ax.set_ylabel('Episodes to Positive Alignment')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def analyze_complexity_scaling(self):
        """Analyze how alignment benefits scale with complexity."""
        print("Analyzing alignment benefits vs complexity...")
        
        if self.alignment_stats.empty:
            return {}
        
        results = {}
        
        # Compare K=10 vs K=1 performance by complexity
        comparison_data = []
        
        for complexity in self.alignment_stats['complexity'].unique():
            subset = self.alignment_stats[self.alignment_stats['complexity'] == complexity]
            
            k1_data = subset[subset['adaptation_steps'] == 1]
            k10_data = subset[subset['adaptation_steps'] == 10]
            
            if not k1_data.empty and not k10_data.empty:
                
                # Mean final alignment comparison
                k1_alignment = k1_data['final_alignment'].mean()
                k10_alignment = k10_data['final_alignment'].mean()
                
                # Mean final accuracy comparison
                k1_accuracy = k1_data['final_accuracy'].mean()
                k10_accuracy = k10_data['final_accuracy'].mean()
                
                comparison_data.append({
                    'complexity': complexity,
                    'k1_alignment': k1_alignment,
                    'k10_alignment': k10_alignment,
                    'alignment_improvement': k10_alignment - k1_alignment,
                    'k1_accuracy': k1_accuracy,
                    'k10_accuracy': k10_accuracy,
                    'accuracy_improvement': k10_accuracy - k1_accuracy,
                    'relative_alignment_gain': (k10_alignment - k1_alignment) / abs(k1_alignment) if k1_alignment != 0 else np.inf,
                    'relative_accuracy_gain': (k10_accuracy - k1_accuracy) / k1_accuracy if k1_accuracy != 0 else np.inf
                })
        
        results['complexity_comparison'] = pd.DataFrame(comparison_data)
        
        # Statistical tests
        if len(comparison_data) > 0:
            comp_df = pd.DataFrame(comparison_data)
            
            # Test if improvement scales with complexity
            if len(comp_df) >= 3:
                alignment_corr, alignment_p = stats.pearsonr(comp_df['complexity'], comp_df['alignment_improvement'])
                accuracy_corr, accuracy_p = stats.pearsonr(comp_df['complexity'], comp_df['accuracy_improvement'])
                
                results['scaling_correlations'] = {
                    'alignment_complexity_corr': alignment_corr,
                    'alignment_complexity_p': alignment_p,
                    'accuracy_complexity_corr': accuracy_corr,
                    'accuracy_complexity_p': accuracy_p
                }
        
        return results
    
    def generate_summary_report(self):
        """Generate comprehensive gradient alignment summary report."""
        print("Generating gradient alignment summary report...")
        
        report = []
        report.append("# Gradient Alignment Analysis Report\n")
        
        # Basic statistics
        report.append("## Dataset Overview\n")
        report.append(f"- Total trajectories analyzed: {len(self.trajectory_data)}")
        report.append(f"- Unique configurations: {len(self.alignment_stats[['features', 'depth', 'adaptation_steps']].drop_duplicates())}")
        
        # Adaptation steps comparison
        if not self.alignment_stats.empty:
            report.append("\n## Adaptation Steps Comparison\n")
            
            k_comparison = self.alignment_stats.groupby('adaptation_steps').agg({
                'mean_alignment': ['mean', 'std'],
                'final_alignment': ['mean', 'std'],
                'convergence_rate': ['mean', 'std'],
                'stability_index': ['mean', 'std']
            }).round(4)
            
            report.append(k_comparison.to_string())
        
        # Complexity scaling analysis
        scaling_results = self.analyze_complexity_scaling()
        if 'complexity_comparison' in scaling_results:
            report.append("\n## Complexity Scaling Analysis\n")
            
            comp_df = scaling_results['complexity_comparison']
            
            # Average improvements
            avg_alignment_improvement = comp_df['alignment_improvement'].mean()
            avg_accuracy_improvement = comp_df['accuracy_improvement'].mean()
            
            report.append(f"- Average alignment improvement (K=10 vs K=1): {avg_alignment_improvement:.4f}")
            report.append(f"- Average accuracy improvement (K=10 vs K=1): {avg_accuracy_improvement:.4f}")
            
            if 'scaling_correlations' in scaling_results:
                corr_data = scaling_results['scaling_correlations']
                report.append(f"- Alignment improvement correlation with complexity: {corr_data['alignment_complexity_corr']:.4f} (p={corr_data['alignment_complexity_p']:.4f})")
                report.append(f"- Accuracy improvement correlation with complexity: {corr_data['accuracy_complexity_corr']:.4f} (p={corr_data['accuracy_complexity_p']:.4f})")
        
        # Key insights
        report.append("\n## Key Insights\n")
        
        if not self.alignment_stats.empty:
            k1_mean = self.alignment_stats[self.alignment_stats['adaptation_steps'] == 1]['final_alignment'].mean()
            k10_mean = self.alignment_stats[self.alignment_stats['adaptation_steps'] == 10]['final_alignment'].mean()
            
            if not np.isnan(k1_mean) and not np.isnan(k10_mean):
                improvement = ((k10_mean - k1_mean) / abs(k1_mean)) * 100
                report.append(f"- K=10 shows {improvement:.1f}% better final gradient alignment than K=1")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'gradient_alignment_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Saved gradient alignment report to {report_path}")
        
        return '\n'.join(report)

def main():
    parser = argparse.ArgumentParser(description='Gradient Alignment Analysis')
    parser.add_argument('--base_results_dir', type=str, default='results/concept_multiseed',
                       help='Directory containing trajectory CSV files')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for plots and reports')
    parser.add_argument('--compare_adaptation_steps', action='store_true',
                       help='Compare different adaptation step values')
    parser.add_argument('--stratify_by_complexity', action='store_true',
                       help='Stratify analysis by concept complexity')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GradientAlignmentAnalyzer(args.base_results_dir, args.output_dir)
    
    # Load data
    analyzer.load_trajectory_data()
    
    if not analyzer.trajectory_data:
        print("No trajectory data with gradient alignment found. Check the base_results_dir path.")
        return
    
    # Compute alignment statistics
    alignment_stats = analyzer.compute_alignment_statistics()
    
    # Generate visualizations
    analyzer.plot_alignment_evolution()
    
    # Analyze complexity scaling if requested
    if args.stratify_by_complexity:
        scaling_results = analyzer.analyze_complexity_scaling()
        print("Complexity scaling analysis complete")
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("Gradient alignment analysis complete!")

if __name__ == "__main__":
    main() 