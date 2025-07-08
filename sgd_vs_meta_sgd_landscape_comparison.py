#!/usr/bin/env python3
"""
SGD vs Meta-SGD Loss Landscape Roughness and Curvature Comparison

This script compares loss landscape properties between SGD baseline and Meta-SGD
across different concept complexities, adapting the k1_vs_k10 comparison structure.
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

class SGDvsMetaSGDAnalyzer:
    def __init__(self, sgd_results_dir="results/baseline_sgd/baseline_run1", 
                 meta_sgd_results_dir="results", output_dir="sgd_vs_meta_sgd_results"):
        self.sgd_results_dir = sgd_results_dir
        self.meta_sgd_results_dir = meta_sgd_results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.sgd_data = {}
        self.meta_sgd_data = {}
        self.comparison_results = {}
        
    def load_trajectory_data(self):
        """Load trajectory data from both SGD and Meta-SGD."""
        print("Loading SGD baseline trajectory data...")
        self.sgd_data = self._load_sgd_baselines()
        print(f"Loaded {len(self.sgd_data)} SGD baseline files")
        
        print("Loading Meta-SGD trajectory data...")
        self.meta_sgd_data = self._load_meta_sgd_trajectories()
        print(f"Loaded {len(self.meta_sgd_data)} Meta-SGD trajectory files")
        
    def _load_sgd_baselines(self):
        """Load SGD baseline trajectory files."""
        sgd_data = {}
        
        # Pattern to match SGD baseline files
        pattern = os.path.join(self.sgd_results_dir, "*baselinetrajectory.csv")
        files = glob.glob(pattern)
        
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                params = self._parse_sgd_filename(filename)
                
                if params:
                    df = pd.read_csv(file_path)
                    key = (params['features'], params['depth'], params['seed'])
                    sgd_data[key] = df
                    
            except Exception as e:
                print(f"Error loading SGD file {file_path}: {e}")
                
        return sgd_data
        
    def _load_meta_sgd_trajectories(self):
        """Load Meta-SGD trajectory files."""
        meta_sgd_data = {}
        
        # Pattern to match Meta-SGD trajectory files
        pattern = os.path.join(self.meta_sgd_results_dir, "**/*trajectory.csv")
        files = glob.glob(pattern, recursive=True)
        
        # Dictionary to store the latest epoch for each configuration
        latest_files = {}
        
        for file_path in files:
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
                meta_sgd_data[config_key] = df
                    
            except Exception as e:
                print(f"Error loading Meta-SGD file {file_path}: {e}")
                
        return meta_sgd_data
        
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
    
    def compute_landscape_roughness(self):
        """Compute landscape roughness metrics for SGD vs Meta-SGD."""
        print("Computing landscape roughness metrics...")
        
        results = []
        
        # Process SGD data
        for key, df in self.sgd_data.items():
            features, depth, seed = key
            if 'query_accuracy' in df.columns and 'query_loss' in df.columns:
                
                # Compute roughness metrics
                acc_roughness = self._compute_roughness(df['query_accuracy'].values)
                loss_roughness = self._compute_roughness(df['query_loss'].values)
                
                results.append({
                    'features': features,
                    'depth': depth,
                    'seed': seed,
                    'method': 'SGD',
                    'complexity': self._get_complexity_label(features, depth),
                    'accuracy_roughness': acc_roughness,
                    'loss_roughness': loss_roughness,
                    'final_accuracy': df['query_accuracy'].iloc[-1],
                    'final_loss': df['query_loss'].iloc[-1],
                    'mean_accuracy': df['query_accuracy'].mean(),
                    'accuracy_std': df['query_accuracy'].std()
                })
        
        # Process Meta-SGD data
        for key, df in self.meta_sgd_data.items():
            features, depth, order, seed = key
            if 'val_accuracy' in df.columns and 'val_loss' in df.columns:
                
                # Compute roughness metrics
                acc_roughness = self._compute_roughness(df['val_accuracy'].values)
                loss_roughness = self._compute_roughness(df['val_loss'].values)
                
                results.append({
                    'features': features,
                    'depth': depth,
                    'seed': seed,
                    'method': 'Meta-SGD',
                    'complexity': self._get_complexity_label(features, depth),
                    'accuracy_roughness': acc_roughness,
                    'loss_roughness': loss_roughness,
                    'final_accuracy': df['val_accuracy'].iloc[-1],
                    'final_loss': df['val_loss'].iloc[-1],
                    'mean_accuracy': df['val_accuracy'].mean(),
                    'accuracy_std': df['val_accuracy'].std()
                })
        
        self.comparison_results = pd.DataFrame(results)
        return self.comparison_results
    
    def _compute_roughness(self, values):
        """Compute roughness as the standard deviation of first differences."""
        if len(values) < 2:
            return 0.0
        
        first_diffs = np.diff(values)
        return np.std(first_diffs)
    
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
    
    def plot_landscape_roughness_comparison(self):
        """Plot landscape roughness comparison between SGD and Meta-SGD."""
        print("Plotting landscape roughness comparison...")
        
        if self.comparison_results.empty:
            print("No comparison results found. Run compute_landscape_roughness first.")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Loss Landscape Roughness: SGD vs Meta-SGD', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy Roughness by Complexity
        ax1 = axes[0, 0]
        sns.boxplot(data=self.comparison_results, x='complexity', y='accuracy_roughness', 
                   hue='method', ax=ax1)
        ax1.set_title('Accuracy Trajectory Roughness', fontweight='bold')
        ax1.set_xlabel('Concept Complexity')
        ax1.set_ylabel('Roughness (Std of First Differences)')
        ax1.legend(title='Method')
        
        # Plot 2: Loss Roughness by Complexity  
        ax2 = axes[0, 1]
        sns.boxplot(data=self.comparison_results, x='complexity', y='loss_roughness', 
                   hue='method', ax=ax2)
        ax2.set_title('Loss Trajectory Roughness', fontweight='bold')
        ax2.set_xlabel('Concept Complexity')
        ax2.set_ylabel('Roughness (Std of First Differences)')
        ax2.legend(title='Method')
        
        # Plot 3: Final Performance vs Roughness
        ax3 = axes[1, 0]
        for method in ['SGD', 'Meta-SGD']:
            method_data = self.comparison_results[self.comparison_results['method'] == method]
            ax3.scatter(method_data['accuracy_roughness'], method_data['final_accuracy'], 
                       label=method, alpha=0.7, s=60)
        ax3.set_xlabel('Accuracy Roughness')
        ax3.set_ylabel('Final Accuracy')
        ax3.set_title('Final Performance vs Landscape Roughness', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Summary
        ax4 = axes[1, 1]
        perf_summary = self.comparison_results.groupby(['complexity', 'method']).agg({
            'final_accuracy': 'mean',
            'accuracy_roughness': 'mean'
        }).reset_index()
        
        for complexity in perf_summary['complexity'].unique():
            complex_data = perf_summary[perf_summary['complexity'] == complexity]
            sgd_data = complex_data[complex_data['method'] == 'SGD']
            meta_data = complex_data[complex_data['method'] == 'Meta-SGD']
            
            if not sgd_data.empty and not meta_data.empty:
                ax4.arrow(sgd_data['final_accuracy'].iloc[0], sgd_data['accuracy_roughness'].iloc[0],
                         meta_data['final_accuracy'].iloc[0] - sgd_data['final_accuracy'].iloc[0],
                         meta_data['accuracy_roughness'].iloc[0] - sgd_data['accuracy_roughness'].iloc[0],
                         head_width=0.002, head_length=0.005, fc='red', ec='red', alpha=0.7)
                ax4.text(sgd_data['final_accuracy'].iloc[0], sgd_data['accuracy_roughness'].iloc[0] + 0.005,
                        complexity.split()[0], ha='center', fontsize=8)
        
        ax4.set_xlabel('Final Accuracy')
        ax4.set_ylabel('Accuracy Roughness')
        ax4.set_title('SGD → Meta-SGD Performance & Roughness', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sgd_vs_meta_sgd_landscape_roughness.svg'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'sgd_vs_meta_sgd_landscape_roughness.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_curvature_analysis(self):
        """Plot curvature analysis showing the relationship between landscape properties and performance."""
        print("Plotting curvature analysis...")
        
        if self.comparison_results.empty:
            print("No comparison results found. Run compute_landscape_roughness first.")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Loss Landscape Curvature Analysis: SGD vs Meta-SGD', fontsize=16, fontweight='bold')
        
        # Plot 1: Roughness vs Performance by Method
        ax1 = axes[0, 0]
        for method in ['SGD', 'Meta-SGD']:
            method_data = self.comparison_results[self.comparison_results['method'] == method]
            ax1.scatter(method_data['accuracy_roughness'], method_data['mean_accuracy'], 
                       label=method, alpha=0.7, s=60)
        ax1.set_xlabel('Accuracy Roughness')
        ax1.set_ylabel('Mean Accuracy')
        ax1.set_title('Mean Performance vs Landscape Roughness', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Stability (1/std) vs Performance
        ax2 = axes[0, 1]
        self.comparison_results['stability'] = 1 / (self.comparison_results['accuracy_std'] + 1e-8)
        for method in ['SGD', 'Meta-SGD']:
            method_data = self.comparison_results[self.comparison_results['method'] == method]
            ax2.scatter(method_data['stability'], method_data['mean_accuracy'], 
                       label=method, alpha=0.7, s=60)
        ax2.set_xlabel('Stability (1/std)')
        ax2.set_ylabel('Mean Accuracy')
        ax2.set_title('Performance vs Stability', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Complexity-dependent Roughness
        ax3 = axes[1, 0]
        roughness_by_complexity = self.comparison_results.groupby(['complexity', 'method']).agg({
            'accuracy_roughness': ['mean', 'std']
        }).reset_index()
        roughness_by_complexity.columns = ['complexity', 'method', 'roughness_mean', 'roughness_std']
        
        sns.barplot(data=roughness_by_complexity, x='complexity', y='roughness_mean', 
                   hue='method', ax=ax3)
        ax3.set_title('Landscape Roughness by Complexity', fontweight='bold')
        ax3.set_xlabel('Concept Complexity')
        ax3.set_ylabel('Mean Accuracy Roughness')
        ax3.legend(title='Method')
        
        # Plot 4: Meta-Learning Benefit vs Roughness
        ax4 = axes[1, 1]
        
        # Calculate meta-learning benefit
        sgd_perf = self.comparison_results[self.comparison_results['method'] == 'SGD'].groupby('complexity')['mean_accuracy'].mean()
        meta_perf = self.comparison_results[self.comparison_results['method'] == 'Meta-SGD'].groupby('complexity')['mean_accuracy'].mean()
        
        sgd_rough = self.comparison_results[self.comparison_results['method'] == 'SGD'].groupby('complexity')['accuracy_roughness'].mean()
        
        benefit_data = []
        for complexity in sgd_perf.index:
            if complexity in meta_perf.index:
                benefit = meta_perf[complexity] - sgd_perf[complexity]
                roughness = sgd_rough[complexity]
                benefit_data.append({'complexity': complexity, 'benefit': benefit, 'sgd_roughness': roughness})
        
        benefit_df = pd.DataFrame(benefit_data)
        
        if not benefit_df.empty:
            ax4.scatter(benefit_df['sgd_roughness'], benefit_df['benefit'], s=100, alpha=0.7)
            for i, row in benefit_df.iterrows():
                ax4.annotate(row['complexity'].split()[0], 
                           (row['sgd_roughness'], row['benefit']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax4.set_xlabel('SGD Landscape Roughness')
            ax4.set_ylabel('Meta-Learning Benefit (Δ Accuracy)')
            ax4.set_title('Meta-Learning Benefit vs SGD Landscape Roughness', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sgd_vs_meta_sgd_curvature_analysis.svg'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'sgd_vs_meta_sgd_curvature_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("Generating summary report...")
        
        if self.comparison_results.empty:
            print("No comparison results found. Run compute_landscape_roughness first.")
            return
        
        # Compute summary statistics
        summary_stats = self.comparison_results.groupby(['complexity', 'method']).agg({
            'final_accuracy': ['mean', 'std'],
            'accuracy_roughness': ['mean', 'std'],
            'loss_roughness': ['mean', 'std']
        }).round(4)
        
        # Create report
        report = f"""# SGD vs Meta-SGD Loss Landscape Analysis Report

## Summary Statistics

### Performance Comparison
{summary_stats.to_string()}

## Key Findings

### Landscape Roughness Analysis
"""
        
        # Add complexity-specific analysis
        for complexity in self.comparison_results['complexity'].unique():
            complex_data = self.comparison_results[self.comparison_results['complexity'] == complexity]
            sgd_data = complex_data[complex_data['method'] == 'SGD']
            meta_data = complex_data[complex_data['method'] == 'Meta-SGD']
            
            if not sgd_data.empty and not meta_data.empty:
                sgd_acc = sgd_data['final_accuracy'].mean()
                meta_acc = meta_data['final_accuracy'].mean()
                sgd_rough = sgd_data['accuracy_roughness'].mean()
                meta_rough = meta_data['accuracy_roughness'].mean()
                
                improvement = meta_acc - sgd_acc
                roughness_change = meta_rough - sgd_rough
                
                report += f"""
### {complexity}
- **SGD Performance**: {sgd_acc:.4f} (roughness: {sgd_rough:.4f})
- **Meta-SGD Performance**: {meta_acc:.4f} (roughness: {meta_rough:.4f})
- **Improvement**: {improvement:+.4f} ({improvement/sgd_acc*100:+.1f}%)
- **Roughness Change**: {roughness_change:+.4f} ({roughness_change/sgd_rough*100:+.1f}%)
"""
        
        # Save report
        with open(os.path.join(self.output_dir, 'sgd_vs_meta_sgd_landscape_report.md'), 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to {self.output_dir}/sgd_vs_meta_sgd_landscape_report.md")
        print(report)

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='SGD vs Meta-SGD Landscape Analysis')
    parser.add_argument('--sgd-dir', default='results/baseline_sgd/baseline_run1',
                       help='Directory containing SGD baseline results')
    parser.add_argument('--meta-sgd-dir', default='results',
                       help='Directory containing Meta-SGD results')
    parser.add_argument('--output-dir', default='sgd_vs_meta_sgd_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SGDvsMetaSGDAnalyzer(args.sgd_dir, args.meta_sgd_dir, args.output_dir)
    
    # Run analysis
    analyzer.load_trajectory_data()
    analyzer.compute_landscape_roughness()
    analyzer.plot_landscape_roughness_comparison()
    analyzer.plot_curvature_analysis()
    analyzer.generate_summary_report()
    
    print(f"Analysis complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 