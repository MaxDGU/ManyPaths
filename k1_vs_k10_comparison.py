#!/usr/bin/env python3
"""
K=1 vs K=10 Data Efficiency Comparison for Camera-Ready Submission

This script compares data efficiency between K=1 and K=10 adaptation steps
using trajectory data from two different directories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class K1vsK10Analyzer:
    def __init__(self, k1_results_dir, k10_results_dir, output_dir="k1_vs_k10_results"):
        self.k1_results_dir = k1_results_dir
        self.k10_results_dir = k10_results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.k1_data = {}
        self.k10_data = {}
        self.comparison_results = {}
        
    def load_trajectory_data(self):
        """Load trajectory data from both K=1 and K=10 directories."""
        print("Loading K=1 trajectory data...")
        self.k1_data = self._load_from_directory(self.k1_results_dir, expected_k=1)
        print(f"Loaded {len(self.k1_data)} K=1 trajectory files")
        
        print("Loading K=10 trajectory data...")
        self.k10_data = self._load_from_directory(self.k10_results_dir, expected_k=10)
        print(f"Loaded {len(self.k10_data)} K=10 trajectory files")
        
    def _load_from_directory(self, results_dir, expected_k):
        """Load trajectory files from a specific directory."""
        trajectory_data = {}
        
        # Pattern to match trajectory files
        pattern = os.path.join(results_dir, "*_trajectory.csv")
        files = glob.glob(pattern)
        
        # Dictionary to store the latest epoch for each configuration
        latest_files = {}
        
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                params = self._parse_filename(filename)
                
                if params and params['adaptation_steps'] == expected_k:
                    # Create a unique key for this configuration
                    config_key = (params['features'], params['depth'], 
                                params['order'], params['seed'])
                    
                    # Keep track of the latest epoch for each configuration
                    epoch = params.get('epoch', 0)
                    if config_key not in latest_files or epoch > latest_files[config_key][1]:
                        latest_files[config_key] = (file_path, epoch)
                        
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
        
        # Load the latest trajectory file for each configuration
        for config_key, (file_path, epoch) in latest_files.items():
            try:
                df = pd.read_csv(file_path)
                filename = os.path.basename(file_path)
                params = self._parse_filename(filename)
                
                if params:
                    key = (params['features'], params['depth'], params['order'], params['seed'])
                    trajectory_data[key] = df
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return trajectory_data
        
    def _parse_filename(self, filename):
        """Parse parameters from trajectory filename."""
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
    
    def compute_efficiency_comparison(self, thresholds=[50, 60, 70, 80]):
        """Compare data efficiency between K=1 and K=10."""
        print("Computing efficiency comparison...")
        
        results = []
        
        # Process K=1 data
        for key, df in self.k1_data.items():
            features, depth, order, seed = key
            self._process_trajectory(df, features, depth, order, seed, "K=1", thresholds, results)
        
        # Process K=10 data
        for key, df in self.k10_data.items():
            features, depth, order, seed = key
            self._process_trajectory(df, features, depth, order, seed, "K=10", thresholds, results)
        
        self.comparison_results = pd.DataFrame(results)
        return self.comparison_results
    
    def _process_trajectory(self, df, features, depth, order, seed, k_value, thresholds, results):
        """Process a single trajectory file."""
        if 'val_accuracy' not in df.columns:
            return
            
        # Convert to percentage if needed
        accuracy = df['val_accuracy'].values
        if np.max(accuracy) <= 1.0:
            accuracy = accuracy * 100
            
        # Assuming each log_step represents 1000 episodes
        episodes = df['log_step'].values * 1000
        
        # Assuming 10 samples per episode
        samples = episodes * 10
        
        for threshold in thresholds:
            # Find first point where accuracy >= threshold
            threshold_idx = np.where(accuracy >= threshold)[0]
            
            if len(threshold_idx) > 0:
                samples_to_threshold = samples[threshold_idx[0]]
                converged = True
            else:
                samples_to_threshold = np.inf
                converged = False
                
            results.append({
                'features': features,
                'depth': depth,
                'order': order,
                'seed': seed,
                'k_value': k_value,
                'threshold': threshold,
                'samples_to_threshold': samples_to_threshold,
                'converged': converged,
                'final_accuracy': accuracy[-1],
                'max_accuracy': np.max(accuracy)
            })
    
    def plot_comparison(self, threshold=60):
        """Plot K=1 vs K=10 comparison."""
        print(f"Plotting K=1 vs K=10 comparison for {threshold}% threshold...")
        
        # Filter data for the specified threshold
        plot_data = self.comparison_results[
            (self.comparison_results['threshold'] == threshold) &
            (self.comparison_results['converged'] == True)
        ]
        
        if plot_data.empty:
            print(f"No converged data for {threshold}% threshold")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'K=1 vs K=10 Data Efficiency Comparison: {threshold}% Threshold', fontsize=16)
        
        # 1. Bar plot comparison
        ax1 = axes[0, 0]
        plot_data_agg = plot_data.groupby(['features', 'depth', 'k_value']).agg({
            'samples_to_threshold': ['mean', 'std']
        }).reset_index()
        plot_data_agg.columns = ['features', 'depth', 'k_value', 'mean_samples', 'std_samples']
        
        plot_data_agg['complexity'] = plot_data_agg.apply(
            lambda row: f"F{row['features']}_D{row['depth']}", axis=1
        )
        
        sns.barplot(data=plot_data_agg, x='complexity', y='mean_samples', hue='k_value', ax=ax1)
        ax1.set_yscale('log')
        ax1.set_title('Samples to Threshold by Complexity')
        ax1.set_ylabel('Samples (log scale)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Efficiency gain (K=10 vs K=1)
        ax2 = axes[0, 1]
        
        # Compute efficiency gains
        gains = []
        for features in plot_data['features'].unique():
            for depth in plot_data['depth'].unique():
                k1_data = plot_data[
                    (plot_data['features'] == features) & 
                    (plot_data['depth'] == depth) & 
                    (plot_data['k_value'] == 'K=1')
                ]['samples_to_threshold']
                
                k10_data = plot_data[
                    (plot_data['features'] == features) & 
                    (plot_data['depth'] == depth) & 
                    (plot_data['k_value'] == 'K=10')
                ]['samples_to_threshold']
                
                if len(k1_data) > 0 and len(k10_data) > 0:
                    # Efficiency gain = K1_samples / K10_samples
                    gain = np.mean(k1_data) / np.mean(k10_data)
                    gains.append({
                        'features': features,
                        'depth': depth,
                        'complexity': f"F{features}_D{depth}",
                        'efficiency_gain': gain
                    })
        
        gains_df = pd.DataFrame(gains)
        if not gains_df.empty:
            sns.barplot(data=gains_df, x='complexity', y='efficiency_gain', ax=ax2)
            ax2.set_title('Efficiency Gain (K=1 samples / K=10 samples)')
            ax2.set_ylabel('Efficiency Gain')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        # 3. Scaling with features
        ax3 = axes[1, 0]
        for k_value in ['K=1', 'K=10']:
            k_data = plot_data[plot_data['k_value'] == k_value]
            k_agg = k_data.groupby('features')['samples_to_threshold'].mean()
            
            ax3.plot(k_agg.index, k_agg.values, 'o-', label=k_value, markersize=8)
        
        ax3.set_yscale('log')
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Samples to Threshold (log scale)')
        ax3.set_title('Scaling with Feature Dimension')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Scaling with depth
        ax4 = axes[1, 1]
        for k_value in ['K=1', 'K=10']:
            k_data = plot_data[plot_data['k_value'] == k_value]
            k_agg = k_data.groupby('depth')['samples_to_threshold'].mean()
            
            ax4.plot(k_agg.index, k_agg.values, 's-', label=k_value, markersize=8)
        
        ax4.set_yscale('log')
        ax4.set_xlabel('Concept Depth')
        ax4.set_ylabel('Samples to Threshold (log scale)')
        ax4.set_title('Scaling with Concept Depth')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'k1_vs_k10_comparison_threshold_{threshold}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
        plt.show()
    
    def compute_statistical_tests(self, threshold=60):
        """Compute statistical tests comparing K=1 vs K=10."""
        print("Computing statistical tests...")
        
        test_results = []
        
        # Filter data for the specified threshold
        test_data = self.comparison_results[
            (self.comparison_results['threshold'] == threshold) &
            (self.comparison_results['converged'] == True)
        ]
        
        for features in test_data['features'].unique():
            for depth in test_data['depth'].unique():
                k1_samples = test_data[
                    (test_data['features'] == features) & 
                    (test_data['depth'] == depth) & 
                    (test_data['k_value'] == 'K=1')
                ]['samples_to_threshold']
                
                k10_samples = test_data[
                    (test_data['features'] == features) & 
                    (test_data['depth'] == depth) & 
                    (test_data['k_value'] == 'K=10')
                ]['samples_to_threshold']
                
                if len(k1_samples) > 0 and len(k10_samples) > 0:
                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = stats.mannwhitneyu(k1_samples, k10_samples, alternative='two-sided')
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(k1_samples) - 1) * np.var(k1_samples, ddof=1) + 
                                        (len(k10_samples) - 1) * np.var(k10_samples, ddof=1)) / 
                                       (len(k1_samples) + len(k10_samples) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (np.mean(k1_samples) - np.mean(k10_samples)) / pooled_std
                    else:
                        cohens_d = 0
                    
                    test_results.append({
                        'features': features,
                        'depth': depth,
                        'complexity': f"F{features}_D{depth}",
                        'k1_mean': np.mean(k1_samples),
                        'k10_mean': np.mean(k10_samples),
                        'efficiency_gain': np.mean(k1_samples) / np.mean(k10_samples),
                        'mann_whitney_u': statistic,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05
                    })
        
        return pd.DataFrame(test_results)
    
    def generate_report(self, thresholds=[50, 60, 70, 80]):
        """Generate comprehensive comparison report."""
        print("Generating comparison report...")
        
        report = []
        report.append("# K=1 vs K=10 Data Efficiency Comparison Report\n")
        
        # Basic statistics
        report.append("## Basic Statistics\n")
        report.append(f"- K=1 trajectory files: {len(self.k1_data)}")
        report.append(f"- K=10 trajectory files: {len(self.k10_data)}")
        report.append(f"- Thresholds analyzed: {thresholds}")
        
        # Convergence analysis
        convergence_stats = self.comparison_results.groupby(['threshold', 'k_value'])['converged'].agg(['count', 'sum', 'mean'])
        report.append("\n## Convergence Analysis\n")
        report.append(convergence_stats.to_string())
        
        # Statistical tests for 60% threshold
        stat_tests = self.compute_statistical_tests(threshold=60)
        if not stat_tests.empty:
            report.append("\n## Statistical Tests (60% Threshold)\n")
            report.append(stat_tests.to_string())
            
            # Summary statistics
            significant_count = stat_tests['significant'].sum()
            total_tests = len(stat_tests)
            mean_efficiency_gain = stat_tests['efficiency_gain'].mean()
            
            report.append(f"\n### Summary:")
            report.append(f"- Significant results: {significant_count}/{total_tests}")
            report.append(f"- Mean efficiency gain (K=1/K=10): {mean_efficiency_gain:.2f}")
            report.append(f"- Mean Cohen's d: {stat_tests['cohens_d'].mean():.3f}")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'k1_vs_k10_comparison_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Saved comparison report to {report_path}")
        
        return '\n'.join(report)

def main():
    parser = argparse.ArgumentParser(description='K=1 vs K=10 Data Efficiency Comparison')
    parser.add_argument('--k1_results_dir', type=str, default='results/run1',
                       help='Directory containing K=1 trajectory files')
    parser.add_argument('--k10_results_dir', type=str, default='results/concept_multiseed',
                       help='Directory containing K=10 trajectory files')
    parser.add_argument('--output_dir', type=str, default='k1_vs_k10_results',
                       help='Output directory for comparison results')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[50, 60, 70, 80],
                       help='Accuracy thresholds to analyze')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = K1vsK10Analyzer(args.k1_results_dir, args.k10_results_dir, args.output_dir)
    
    # Load data
    analyzer.load_trajectory_data()
    
    if not analyzer.k1_data and not analyzer.k10_data:
        print("No trajectory data found. Check the directory paths.")
        return
    
    # Compute efficiency comparison
    comparison_results = analyzer.compute_efficiency_comparison(args.thresholds)
    
    # Generate plots for each threshold
    for threshold in args.thresholds:
        analyzer.plot_comparison(threshold)
    
    # Generate report
    analyzer.generate_report(args.thresholds)
    
    print("K=1 vs K=10 comparison analysis complete!")

if __name__ == "__main__":
    main() 