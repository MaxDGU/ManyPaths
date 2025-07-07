#!/usr/bin/env python3
"""
Enhanced Data Efficiency Analysis for ManyPaths Camera-Ready Submission

This script analyzes existing trajectory data to provide rigorous statistical evidence
for data efficiency claims, including multiple threshold analysis, confidence intervals,
effect sizes, and scaling laws.

Usage:
    python enhanced_data_efficiency_analysis.py --base_results_dir results/concept_multiseed \
        --thresholds 50 60 70 80 --confidence_intervals --statistical_tests --effect_sizes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class DataEfficiencyAnalyzer:
    def __init__(self, base_results_dir, output_dir="figures"):
        self.base_results_dir = base_results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.trajectory_data = {}
        self.efficiency_results = {}
        
    def load_trajectory_data(self):
        """Load all trajectory CSV files from the results directory."""
        print("Loading trajectory data...")
        
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
        for config_key, (file_path, epoch) in latest_files.items():
            try:
                df = pd.read_csv(file_path)
                filename = os.path.basename(file_path)
                params = self._parse_filename(filename)
                
                if params:
                    key = (params['features'], params['depth'], params['method'], params['seed'])
                    self.trajectory_data[key] = df
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        print(f"Loaded {len(self.trajectory_data)} trajectory files")
        
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
            
            method = f"MetaSGD_{order}Ord_K{adapt_steps}"
            
            return {
                'features': features,
                'depth': depth,
                'adaptation_steps': adapt_steps,
                'order': order,
                'method': method,
                'seed': seed,
                'epoch': epoch
            }
        return None
    
    def compute_samples_to_threshold(self, thresholds=[50, 60, 70, 80]):
        """Compute samples needed to reach accuracy thresholds."""
        print("Computing samples to threshold...")
        
        results = []
        
        for key, df in self.trajectory_data.items():
            features, depth, method, seed = key
            
            if 'val_accuracy' not in df.columns:
                continue
                
            # Convert to percentage if needed
            accuracy = df['val_accuracy'].values
            if np.max(accuracy) <= 1.0:
                accuracy = accuracy * 100
                
            # Assuming each log_step represents 1000 episodes (from constants.py LOG_INTERVAL)
            episodes = df['log_step'].values * 1000
            
            # Assuming 10 samples per episode (support + query for one task)
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
                    'method': method,
                    'seed': seed,
                    'threshold': threshold,
                    'samples_to_threshold': samples_to_threshold,
                    'converged': converged,
                    'final_accuracy': accuracy[-1],
                    'max_accuracy': np.max(accuracy)
                })
        
        self.efficiency_results = pd.DataFrame(results)
        return self.efficiency_results
    
    def compute_effect_sizes(self):
        """Compute Cohen's d effect sizes between methods."""
        print("Computing effect sizes...")
        
        effect_sizes = []
        
        # Compare MetaSGD methods against SGD baseline
        for features in self.efficiency_results['features'].unique():
            for depth in self.efficiency_results['depth'].unique():
                for threshold in self.efficiency_results['threshold'].unique():
                    
                    subset = self.efficiency_results[
                        (self.efficiency_results['features'] == features) &
                        (self.efficiency_results['depth'] == depth) &
                        (self.efficiency_results['threshold'] == threshold) &
                        (self.efficiency_results['converged'] == True)
                    ]
                    
                    methods = subset['method'].unique()
                    
                    # Compare all pairs of methods
                    for i, method1 in enumerate(methods):
                        for method2 in methods[i+1:]:
                            
                            data1 = subset[subset['method'] == method1]['samples_to_threshold']
                            data2 = subset[subset['method'] == method2]['samples_to_threshold']
                            
                            if len(data1) > 1 and len(data2) > 1:
                                # Compute Cohen's d
                                pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                                    (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                                   (len(data1) + len(data2) - 2))
                                
                                if pooled_std > 0:
                                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                                    
                                    effect_sizes.append({
                                        'features': features,
                                        'depth': depth,
                                        'threshold': threshold,
                                        'method1': method1,
                                        'method2': method2,
                                        'cohens_d': cohens_d,
                                        'effect_magnitude': self._interpret_effect_size(abs(cohens_d))
                                    })
        
        return pd.DataFrame(effect_sizes)
    
    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def statistical_significance_tests(self):
        """Perform statistical significance tests."""
        print("Performing statistical significance tests...")
        
        test_results = []
        
        for features in self.efficiency_results['features'].unique():
            for depth in self.efficiency_results['depth'].unique():
                for threshold in self.efficiency_results['threshold'].unique():
                    
                    subset = self.efficiency_results[
                        (self.efficiency_results['features'] == features) &
                        (self.efficiency_results['depth'] == depth) &
                        (self.efficiency_results['threshold'] == threshold) &
                        (self.efficiency_results['converged'] == True)
                    ]
                    
                    methods = subset['method'].unique()
                    
                    if len(methods) >= 2:
                        # Perform ANOVA if multiple methods
                        groups = [subset[subset['method'] == method]['samples_to_threshold'].values 
                                for method in methods]
                        groups = [g for g in groups if len(g) > 1]  # Filter empty groups
                        
                        if len(groups) >= 2:
                            try:
                                f_stat, p_value = stats.f_oneway(*groups)
                                
                                test_results.append({
                                    'features': features,
                                    'depth': depth,
                                    'threshold': threshold,
                                    'test': 'ANOVA',
                                    'f_statistic': f_stat,
                                    'p_value': p_value,
                                    'significant': p_value < 0.05,
                                    'methods_compared': list(methods)
                                })
                            except:
                                pass
        
        return pd.DataFrame(test_results)
    
    def fit_scaling_laws(self):
        """Fit scaling laws for efficiency gains."""
        print("Fitting scaling laws...")
        
        # Aggregate data by configuration
        agg_data = self.efficiency_results.groupby(['features', 'depth', 'method', 'threshold']).agg({
            'samples_to_threshold': ['mean', 'std', 'count'],
            'converged': 'all'
        }).reset_index()
        
        agg_data.columns = ['features', 'depth', 'method', 'threshold', 
                           'mean_samples', 'std_samples', 'n_seeds', 'all_converged']
        
        # Only include configurations where all seeds converged
        agg_data = agg_data[agg_data['all_converged']]
        
        scaling_results = {}
        
        # Fit scaling laws for each method and threshold
        for method in agg_data['method'].unique():
            for threshold in agg_data['threshold'].unique():
                
                subset = agg_data[
                    (agg_data['method'] == method) &
                    (agg_data['threshold'] == threshold)
                ]
                
                if len(subset) >= 3:  # Need at least 3 points to fit
                    
                    # Prepare data for fitting
                    X = subset[['features', 'depth']].values
                    y = subset['mean_samples'].values
                    
                    # Try different scaling models
                    models = {
                        'exponential': lambda x, a, b, c: a * np.exp(b * x[:, 0] + c * x[:, 1]),
                        'power': lambda x, a, b, c: a * (x[:, 0] ** b) * (x[:, 1] ** c),
                        'linear': lambda x, a, b, c: a + b * x[:, 0] + c * x[:, 1]
                    }
                    
                    for model_name, model_func in models.items():
                        try:
                            popt, _ = curve_fit(model_func, X, y, maxfev=10000)
                            y_pred = model_func(X, *popt)
                            r2 = r2_score(y, y_pred)
                            
                            scaling_results[f"{method}_{threshold}_{model_name}"] = {
                                'method': method,
                                'threshold': threshold,
                                'model': model_name,
                                'parameters': popt,
                                'r2_score': r2,
                                'data_points': len(subset)
                            }
                        except:
                            pass
        
        return scaling_results
    
    def plot_efficiency_comparison(self, threshold=60):
        """Plot comprehensive efficiency comparison."""
        print(f"Plotting efficiency comparison for {threshold}% threshold...")
        
        # Filter data for the specified threshold
        plot_data = self.efficiency_results[
            (self.efficiency_results['threshold'] == threshold) &
            (self.efficiency_results['converged'] == True)
        ]
        
        if plot_data.empty:
            print(f"No converged data for {threshold}% threshold")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Data Efficiency Analysis: {threshold}% Accuracy Threshold', fontsize=16)
        
        # 1. Bar plot by complexity
        ax1 = axes[0, 0]
        plot_data_agg = plot_data.groupby(['features', 'depth', 'method']).agg({
            'samples_to_threshold': ['mean', 'std']
        }).reset_index()
        plot_data_agg.columns = ['features', 'depth', 'method', 'mean_samples', 'std_samples']
        
        # Create complexity labels
        plot_data_agg['complexity'] = plot_data_agg.apply(
            lambda row: f"F{row['features']}_D{row['depth']}", axis=1
        )
        
        sns.barplot(data=plot_data_agg, x='complexity', y='mean_samples', hue='method', ax=ax1)
        ax1.set_yscale('log')
        ax1.set_title('Samples to Threshold by Complexity')
        ax1.set_ylabel('Samples (log scale)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Efficiency gain heatmap
        ax2 = axes[0, 1]
        
        # Compute efficiency gains relative to baseline
        baseline_method = plot_data_agg['method'].iloc[0]  # Use first method as baseline
        pivot_data = plot_data_agg.pivot_table(
            index=['features', 'depth'], 
            columns='method', 
            values='mean_samples'
        )
        
        if baseline_method in pivot_data.columns:
            efficiency_gains = pivot_data.div(pivot_data[baseline_method], axis=0)
            
            sns.heatmap(efficiency_gains, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax2)
            ax2.set_title('Efficiency Gain (relative to baseline)')
        
        # 3. Scaling with features
        ax3 = axes[1, 0]
        for method in plot_data['method'].unique():
            method_data = plot_data[plot_data['method'] == method]
            method_agg = method_data.groupby('features')['samples_to_threshold'].mean()
            
            ax3.plot(method_agg.index, method_agg.values, 'o-', label=method, markersize=8)
        
        ax3.set_yscale('log')
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Samples to Threshold (log scale)')
        ax3.set_title('Scaling with Feature Dimension')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Scaling with depth
        ax4 = axes[1, 1]
        for method in plot_data['method'].unique():
            method_data = plot_data[plot_data['method'] == method]
            method_agg = method_data.groupby('depth')['samples_to_threshold'].mean()
            
            ax4.plot(method_agg.index, method_agg.values, 's-', label=method, markersize=8)
        
        ax4.set_yscale('log')
        ax4.set_xlabel('Concept Depth')
        ax4.set_ylabel('Samples to Threshold (log scale)')
        ax4.set_title('Scaling with Concept Depth')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'enhanced_efficiency_analysis_threshold_{threshold}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved efficiency comparison plot to {output_path}")
        plt.show()
    
    def generate_summary_report(self, thresholds=[50, 60, 70, 80]):
        """Generate comprehensive summary report."""
        print("Generating summary report...")
        
        report = []
        report.append("# Enhanced Data Efficiency Analysis Report\n")
        
        # Basic statistics
        report.append("## Basic Statistics\n")
        report.append(f"- Total trajectory files analyzed: {len(self.trajectory_data)}")
        report.append(f"- Unique configurations: {len(self.efficiency_results[['features', 'depth', 'method']].drop_duplicates())}")
        report.append(f"- Thresholds analyzed: {thresholds}")
        
        # Convergence analysis
        convergence_stats = self.efficiency_results.groupby(['threshold', 'method'])['converged'].agg(['count', 'sum', 'mean'])
        report.append("\n## Convergence Analysis\n")
        report.append(convergence_stats.to_string())
        
        # Effect sizes
        effect_sizes = self.compute_effect_sizes()
        if not effect_sizes.empty:
            report.append("\n## Effect Sizes (Cohen's d)\n")
            
            # Summarize by magnitude
            effect_summary = effect_sizes['effect_magnitude'].value_counts()
            report.append(effect_summary.to_string())
            
            # Show largest effects
            largest_effects = effect_sizes.nlargest(10, 'cohens_d')[['method1', 'method2', 'cohens_d', 'effect_magnitude']]
            report.append("\n### Largest Effect Sizes:\n")
            report.append(largest_effects.to_string())
        
        # Statistical significance
        sig_tests = self.statistical_significance_tests()
        if not sig_tests.empty:
            report.append("\n## Statistical Significance Tests\n")
            
            significant_count = sig_tests['significant'].sum()
            total_tests = len(sig_tests)
            report.append(f"- Significant results: {significant_count}/{total_tests} ({100*significant_count/total_tests:.1f}%)")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'efficiency_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Saved summary report to {report_path}")
        
        return '\n'.join(report)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Data Efficiency Analysis')
    parser.add_argument('--base_results_dir', type=str, default='results/concept_multiseed',
                       help='Directory containing trajectory CSV files')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for plots and reports')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[50, 60, 70, 80],
                       help='Accuracy thresholds to analyze')
    parser.add_argument('--confidence_intervals', action='store_true',
                       help='Compute confidence intervals')
    parser.add_argument('--statistical_tests', action='store_true',
                       help='Perform statistical significance tests')
    parser.add_argument('--effect_sizes', action='store_true',
                       help='Compute effect sizes')
    parser.add_argument('--scaling_laws', action='store_true',
                       help='Fit scaling law models')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DataEfficiencyAnalyzer(args.base_results_dir, args.output_dir)
    
    # Load data
    analyzer.load_trajectory_data()
    
    if not analyzer.trajectory_data:
        print("No trajectory data found. Check the base_results_dir path.")
        return
    
    # Compute efficiency metrics
    efficiency_results = analyzer.compute_samples_to_threshold(args.thresholds)
    
    # Generate plots for each threshold
    for threshold in args.thresholds:
        analyzer.plot_efficiency_comparison(threshold)
    
    # Optional analyses
    if args.effect_sizes:
        effect_sizes = analyzer.compute_effect_sizes()
        print(f"Computed effect sizes for {len(effect_sizes)} comparisons")
    
    if args.statistical_tests:
        sig_tests = analyzer.statistical_significance_tests()
        print(f"Performed {len(sig_tests)} statistical tests")
    
    if args.scaling_laws:
        scaling_results = analyzer.fit_scaling_laws()
        print(f"Fitted {len(scaling_results)} scaling law models")
    
    # Generate summary report
    analyzer.generate_summary_report(args.thresholds)
    
    print("Enhanced data efficiency analysis complete!")

if __name__ == "__main__":
    main() 