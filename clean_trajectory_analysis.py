#!/usr/bin/env python3
"""
Clean Trajectory Analysis Script
==============================

Fixes the messy trajectory analysis plots by:
1. Reducing visual clutter
2. Consistent color schemes
3. Clear legends and labels
4. Statistical summaries
5. Publication-ready figures

This script replaces the messy della_trajectory_analysis.py output.

Author: Camera-Ready Pipeline
Date: December 2024
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import re
from datetime import datetime

# Clean color scheme
CLEAN_COLORS = {
    'K1': '#e74c3c',    # Clear red
    'K10': '#3498db',   # Clear blue
    'F8D3': '#2ecc71',  # Green for simple
    'F16D3': '#f39c12', # Orange for medium
    'F32D3': '#9b59b6'  # Purple for complex
}

# Figure configuration
FIGURE_CONFIG = {
    'style': 'whitegrid',
    'context': 'paper',
    'font_scale': 1.2,
    'rc': {
        'figure.figsize': (12, 8),
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': True,
        'legend.shadow': True,
        'grid.alpha': 0.3
    }
}

class CleanTrajectoryAnalyzer:
    """Clean trajectory analyzer that fixes messy plots"""
    
    def __init__(self, results_dir: str = "results", output_dir: str = "clean_analysis"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up clean plotting
        self.setup_clean_plotting()
        
        # Data containers
        self.trajectory_data = {}
        self.summary_stats = {}
        
        print(f"🎨 Clean Trajectory Analyzer initialized")
        print(f"   Input: {self.results_dir}")
        print(f"   Output: {self.output_dir}")
    
    def setup_clean_plotting(self):
        """Setup clean plotting style"""
        sns.set_style(FIGURE_CONFIG['style'])
        sns.set_context(FIGURE_CONFIG['context'], font_scale=FIGURE_CONFIG['font_scale'])
        plt.rcParams.update(FIGURE_CONFIG['rc'])
    
    def load_trajectory_data(self) -> Dict[str, pd.DataFrame]:
        """Load trajectory data with clean organization"""
        trajectory_files = list(self.results_dir.rglob("*trajectory*.csv"))
        
        organized_data = {}
        
        for file in trajectory_files:
            exp_info = self.parse_experiment_info(file.name)
            if exp_info:
                key = f"{exp_info['config']}_K{exp_info['adaptation_steps']}_S{exp_info['seed']}"
                
                try:
                    df = pd.read_csv(file)
                    # Clean the data
                    df = self.clean_trajectory_data(df)
                    organized_data[key] = df
                    print(f"   ✅ Loaded {key}: {len(df)} episodes")
                except Exception as e:
                    print(f"   ❌ Failed to load {file.name}: {e}")
        
        return organized_data
    
    def parse_experiment_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Parse experiment info from filename"""
        # Pattern: *feats{F}_depth{D}_adapt{K}*seed{S}*
        pattern = r"feats(\d+)_depth(\d+)_adapt(\d+).*seed(\d+)"
        match = re.search(pattern, filename)
        
        if match:
            features = int(match.group(1))
            depth = int(match.group(2))
            adaptation_steps = int(match.group(3))
            seed = int(match.group(4))
            
            return {
                'features': features,
                'depth': depth,
                'adaptation_steps': adaptation_steps,
                'seed': seed,
                'config': f"F{features}D{depth}"
            }
        
        return None
    
    def clean_trajectory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean trajectory data for analysis"""
        # Remove any NaN values
        df = df.dropna()
        
        # Ensure we have required columns
        if 'val_accuracy' not in df.columns:
            print("   ⚠️  Warning: No val_accuracy column found")
            return df
        
        # Smooth noisy trajectories (optional)
        if len(df) > 10:
            # Rolling average with window size 5
            df['val_accuracy_smooth'] = df['val_accuracy'].rolling(window=5, center=True).mean()
            df['val_accuracy_smooth'].fillna(df['val_accuracy'], inplace=True)
        else:
            df['val_accuracy_smooth'] = df['val_accuracy']
        
        return df
    
    def compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute clean summary statistics"""
        summary_stats = {}
        
        # Group by configuration and method
        config_groups = {}
        
        for exp_key, df in self.trajectory_data.items():
            # Parse key: F8D3_K1_S1
            parts = exp_key.split('_')
            config = parts[0]  # F8D3
            method = parts[1]  # K1 or K10
            
            group_key = f"{config}_{method}"
            if group_key not in config_groups:
                config_groups[group_key] = []
            
            # Final accuracy (last 10 episodes)
            final_acc = df['val_accuracy'].tail(10).mean()
            config_groups[group_key].append(final_acc)
        
        # Compute statistics for each group
        for group_key, accuracies in config_groups.items():
            summary_stats[group_key] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'n': len(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'median': np.median(accuracies)
            }
        
        # Compare K1 vs K10 for each configuration
        comparison_stats = {}
        configs = set(key.split('_')[0] for key in config_groups.keys())
        
        for config in configs:
            k1_key = f"{config}_K1"
            k10_key = f"{config}_K10"
            
            if k1_key in config_groups and k10_key in config_groups:
                k1_acc = config_groups[k1_key]
                k10_acc = config_groups[k10_key]
                
                # Statistical test
                if len(k1_acc) > 1 and len(k10_acc) > 1:
                    t_stat, p_value = stats.ttest_ind(k10_acc, k1_acc)
                    effect_size = self.compute_cohens_d(k10_acc, k1_acc)
                else:
                    t_stat, p_value = None, None
                    effect_size = None
                
                comparison_stats[config] = {
                    'K1_mean': np.mean(k1_acc),
                    'K1_std': np.std(k1_acc),
                    'K1_n': len(k1_acc),
                    'K10_mean': np.mean(k10_acc),
                    'K10_std': np.std(k10_acc),
                    'K10_n': len(k10_acc),
                    'improvement': np.mean(k10_acc) - np.mean(k1_acc),
                    't_stat': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size
                }
        
        return {'group_stats': summary_stats, 'comparisons': comparison_stats}
    
    def compute_cohens_d(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Cohen's d effect size"""
        if len(x1) <= 1 or len(x2) <= 1:
            return None
        
        n1, n2 = len(x1), len(x2)
        s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (np.mean(x1) - np.mean(x2)) / pooled_std
        return d
    
    def create_clean_learning_curves(self):
        """Create clean learning curves (fixes messy plot issue)"""
        print("🎨 Creating clean learning curves...")
        
        # Group trajectories by configuration
        config_groups = {}
        for exp_key, df in self.trajectory_data.items():
            config = exp_key.split('_')[0]
            if config not in config_groups:
                config_groups[config] = {}
            
            method = exp_key.split('_')[1]
            if method not in config_groups[config]:
                config_groups[config][method] = []
            
            config_groups[config][method].append(df)
        
        # Create subplots for each configuration
        n_configs = len(config_groups)
        fig, axes = plt.subplots(1, n_configs, figsize=(6*n_configs, 6))
        
        if n_configs == 1:
            axes = [axes]
        
        for i, (config, methods) in enumerate(sorted(config_groups.items())):
            ax = axes[i]
            
            # Plot each method
            for method, trajectories in methods.items():
                color = CLEAN_COLORS[method]
                
                # Compute mean and std across seeds
                if len(trajectories) > 1:
                    # Average across seeds
                    max_len = max(len(df) for df in trajectories)
                    
                    # Pad trajectories to same length
                    padded_trajectories = []
                    for df in trajectories:
                        vals = df['val_accuracy_smooth'].values
                        if len(vals) < max_len:
                            # Pad with last value
                            padded_vals = np.pad(vals, (0, max_len - len(vals)), 'edge')
                        else:
                            padded_vals = vals[:max_len]
                        padded_trajectories.append(padded_vals)
                    
                    # Compute mean and std
                    mean_trajectory = np.mean(padded_trajectories, axis=0)
                    std_trajectory = np.std(padded_trajectories, axis=0)
                    
                    episodes = range(len(mean_trajectory))
                    
                    # Plot mean with confidence interval
                    ax.plot(episodes, mean_trajectory, color=color, linewidth=3, 
                           label=f'{method} (n={len(trajectories)})')
                    ax.fill_between(episodes, 
                                   mean_trajectory - std_trajectory,
                                   mean_trajectory + std_trajectory,
                                   color=color, alpha=0.2)
                
                else:
                    # Single trajectory
                    df = trajectories[0]
                    episodes = range(len(df))
                    ax.plot(episodes, df['val_accuracy_smooth'], color=color, 
                           linewidth=3, label=f'{method} (n=1)')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Validation Accuracy')
            ax.set_title(f'{config} Learning Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / "clean_learning_curves.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "clean_learning_curves.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved clean learning curves")
    
    def create_clean_comparison_plot(self):
        """Create clean K1 vs K10 comparison plot"""
        print("📊 Creating clean comparison plot...")
        
        if not self.summary_stats['comparisons']:
            print("   ⚠️  No comparison data available")
            return
        
        # Prepare data for plotting
        configs = sorted(self.summary_stats['comparisons'].keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Bar chart with error bars
        k1_means = [self.summary_stats['comparisons'][c]['K1_mean'] for c in configs]
        k10_means = [self.summary_stats['comparisons'][c]['K10_mean'] for c in configs]
        k1_stds = [self.summary_stats['comparisons'][c]['K1_std'] for c in configs]
        k10_stds = [self.summary_stats['comparisons'][c]['K10_std'] for c in configs]
        
        x = np.arange(len(configs))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, k1_means, width, yerr=k1_stds, 
                       label='K=1', color=CLEAN_COLORS['K1'], alpha=0.8, capsize=5)
        bars2 = ax1.bar(x + width/2, k10_means, width, yerr=k10_stds, 
                       label='K=10', color=CLEAN_COLORS['K10'], alpha=0.8, capsize=5)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Final Validation Accuracy')
        ax1.set_title('K=1 vs K=10 Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add significance markers
        for i, config in enumerate(configs):
            stats = self.summary_stats['comparisons'][config]
            if stats['p_value'] and stats['p_value'] < 0.05:
                # Add significance marker
                height = max(stats['K1_mean'] + stats['K1_std'], 
                           stats['K10_mean'] + stats['K10_std'])
                ax1.text(i, height + 0.02, '*', ha='center', va='bottom', 
                        fontsize=16, fontweight='bold')
        
        # Plot 2: Improvement and effect sizes
        improvements = [self.summary_stats['comparisons'][c]['improvement'] for c in configs]
        effect_sizes = [self.summary_stats['comparisons'][c]['effect_size'] or 0 for c in configs]
        
        ax2.bar(x, improvements, color='#34495e', alpha=0.8)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Accuracy Improvement (K=10 - K=1)')
        ax2.set_title('Improvement and Effect Sizes')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs)
        ax2.grid(True, alpha=0.3)
        
        # Add effect size labels
        for i, (imp, eff) in enumerate(zip(improvements, effect_sizes)):
            ax2.text(i, imp + 0.005, f'ES: {eff:.2f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / "clean_k1_vs_k10_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "clean_k1_vs_k10_comparison.pdf", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved clean comparison plot")
    
    def create_statistical_summary_table(self):
        """Create clean statistical summary table"""
        print("📋 Creating statistical summary table...")
        
        # Create summary table
        summary_data = []
        
        for config, stats in self.summary_stats['comparisons'].items():
            significance = "✅ Yes" if stats['p_value'] and stats['p_value'] < 0.05 else "❌ No"
            
            summary_data.append({
                'Configuration': config,
                'K=1 Accuracy': f"{stats['K1_mean']:.3f} ± {stats['K1_std']:.3f}",
                'K=10 Accuracy': f"{stats['K10_mean']:.3f} ± {stats['K10_std']:.3f}",
                'Improvement': f"{stats['improvement']:.3f}",
                'p-value': f"{stats['p_value']:.3f}" if stats['p_value'] else "N/A",
                'Effect Size': f"{stats['effect_size']:.2f}" if stats['effect_size'] else "N/A",
                'Significant': significance,
                'N (K=1)': stats['K1_n'],
                'N (K=10)': stats['K10_n']
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Save as CSV
        df_summary.to_csv(self.output_dir / "clean_statistical_summary.csv", index=False)
        
        # Create formatted text table
        with open(self.output_dir / "clean_statistical_summary.txt", 'w') as f:
            f.write("Camera-Ready Statistical Summary (Clean Analysis)\n")
            f.write("=" * 60 + "\n\n")
            f.write(df_summary.to_string(index=False))
            f.write("\n\n")
            
            f.write("Key Findings:\n")
            f.write("-" * 20 + "\n")
            
            for config, stats in self.summary_stats['comparisons'].items():
                improvement = stats['improvement']
                significance = "statistically significant" if stats['p_value'] and stats['p_value'] < 0.05 else "not significant"
                
                f.write(f"• {config}: {improvement:.3f} improvement ({significance})\n")
        
        print(f"   ✅ Saved statistical summary")
    
    def generate_clean_report(self):
        """Generate clean analysis report"""
        print("📄 Generating clean analysis report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Clean Trajectory Analysis Report
Generated: {timestamp}

## Overview
This report presents a clean analysis of the trajectory data, fixing the messy visualization issues identified in the preliminary results.

## Data Summary
- **Total experiments**: {len(self.trajectory_data)}
- **Configurations**: {len(self.summary_stats['comparisons'])}
- **Statistical comparisons**: {len(self.summary_stats['comparisons'])}

## Key Improvements Made
1. **Reduced visual clutter**: Averaged trajectories across seeds
2. **Consistent color scheme**: Clear distinction between K=1 and K=10
3. **Statistical rigor**: Proper error bars and significance testing
4. **Clean legends**: Removed overlapping labels

## Statistical Findings
"""
        
        for config, stats in self.summary_stats['comparisons'].items():
            improvement = stats['improvement']
            significance = "statistically significant" if stats['p_value'] and stats['p_value'] < 0.05 else "not significant"
            
            report += f"""
### {config}
- **K=1 Performance**: {stats['K1_mean']:.3f} ± {stats['K1_std']:.3f} (n={stats['K1_n']})
- **K=10 Performance**: {stats['K10_mean']:.3f} ± {stats['K10_std']:.3f} (n={stats['K10_n']})
- **Improvement**: {improvement:.3f} accuracy points
- **Statistical significance**: {significance}
- **Effect size**: {stats['effect_size']:.2f} (Cohen's d)
"""
        
        report += """
## Figures Generated
1. **clean_learning_curves.png**: Averaged learning curves with confidence intervals
2. **clean_k1_vs_k10_comparison.png**: Bar chart comparison with significance markers
3. **clean_statistical_summary.csv**: Detailed statistical results

## Camera-Ready Insights
- All figures are publication-ready with consistent styling
- Statistical significance is clearly marked
- Effect sizes provide practical significance assessment
- Clean visualization eliminates the messy plot issues

## Next Steps
1. Integrate with loss landscape analysis
2. Add gradient alignment dynamics
3. Prepare final publication figures
"""
        
        # Save report
        with open(self.output_dir / "clean_analysis_report.md", 'w') as f:
            f.write(report)
        
        print(f"   ✅ Saved clean analysis report")
    
    def run_clean_analysis(self):
        """Run complete clean analysis pipeline"""
        print("🚀 Running clean trajectory analysis...")
        
        # Load data
        print("📊 Loading trajectory data...")
        self.trajectory_data = self.load_trajectory_data()
        
        if not self.trajectory_data:
            print("❌ No trajectory data found!")
            return
        
        # Compute statistics
        print("📈 Computing statistics...")
        self.summary_stats = self.compute_summary_statistics()
        
        # Create clean visualizations
        self.create_clean_learning_curves()
        self.create_clean_comparison_plot()
        self.create_statistical_summary_table()
        
        # Generate report
        self.generate_clean_report()
        
        print("✅ Clean analysis complete!")
        print(f"   Output directory: {self.output_dir}")
        
        return self.output_dir

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Clean Trajectory Analysis')
    parser.add_argument('--results_dir', default='results', help='Results directory')
    parser.add_argument('--output_dir', default='clean_analysis', help='Output directory')
    parser.add_argument('--focus-on-key-metrics', action='store_true', 
                       help='Focus on key metrics only')
    
    args = parser.parse_args()
    
    # Create clean analyzer
    analyzer = CleanTrajectoryAnalyzer(args.results_dir, args.output_dir)
    
    # Run clean analysis
    output_dir = analyzer.run_clean_analysis()
    
    print(f"\n🎉 Clean trajectory analysis complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Key files generated:")
    print(f"   - clean_learning_curves.png: Fixed messy trajectory plots")
    print(f"   - clean_k1_vs_k10_comparison.png: Clear performance comparison")
    print(f"   - clean_statistical_summary.csv: Statistical results")
    print(f"   - clean_analysis_report.md: Complete analysis report")

if __name__ == "__main__":
    main() 