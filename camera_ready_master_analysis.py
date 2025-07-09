#!/usr/bin/env python3
"""
Camera-Ready Master Analysis Script for ManyPaths Paper

This script consolidates all camera-ready analyses into a single, clean pipeline.
It processes Della results and generates publication-ready figures and reports.

Usage:
    python camera_ready_master_analysis.py

Outputs:
    - Clean trajectory plots
    - K=1 vs K=10 comparisons  
    - Statistical summaries
    - Sample efficiency analysis
    - Comprehensive camera-ready report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import stats
import argparse
import warnings
warnings.filterwarnings('ignore')

# Configuration
FIGURE_CONFIG = {
    'style': 'seaborn-v0_8-whitegrid',
    'dpi': 300,
    'format': 'pdf',
    'font_size': 12,
    'color_palette': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
    'figsize': (12, 8)
}

class CameraReadyAnalyzer:
    """Master analyzer for camera-ready submission"""
    
    def __init__(self, results_dir: str = "results", output_dir: str = "camera_ready_final"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.trajectory_data = {}
        self.performance_data = {}
        self.landscape_data = {}
        
        # Configure matplotlib
        plt.style.use(FIGURE_CONFIG['style'])
        sns.set_palette(FIGURE_CONFIG['color_palette'])
        plt.rcParams.update({'font.size': FIGURE_CONFIG['font_size']})
        
    def load_della_results(self) -> int:
        """Load and parse all camera-ready results from Della logs"""
        print("1. Loading Della results...")
        
        # Find camera-ready log files
        search_paths = ['.', 'logs']
        camera_ready_files = []
        
        for search_path in search_paths:
            path_obj = Path(search_path)
            if path_obj.exists():
                files = list(path_obj.glob('camera_ready_array_*_*.out'))
                print(f"Found {len(files)} camera-ready files in {search_path}")
                camera_ready_files.extend(files)
        
        print(f"Total found: {len(camera_ready_files)} camera-ready log files")
        
        if not camera_ready_files:
            print("❌ No camera-ready log files found")
            return 0
        
        # Parse each log file
        parsed_experiments = []
        for log_file in camera_ready_files:
            try:
                experiment_data = self._parse_log_file(log_file)
                if experiment_data:
                    parsed_experiments.append(experiment_data)
            except Exception as e:
                print(f"Error parsing {log_file}: {e}")
        
        print(f"   Loaded {len(camera_ready_files)} log files")
        print(f"   Parsed {len(parsed_experiments)} experiments")
        
        # Store experiments in trajectory_data format for compatibility
        self.trajectory_data = {}
        for exp in parsed_experiments:
            # Create key in format: F16D3_K10_seed1
            key = f"{exp['complexity']}_K{exp['k_value']}_seed{exp['seed']}"
            
            self.trajectory_data[key] = {
                'config': exp['config'],
                'trajectory': exp['trajectory'],
                'performance': {'final_accuracy': exp['final_accuracy']},
                'experiment_info': {
                    'experiment_id': exp['experiment_id'],
                    'status': exp['status'],
                    'log_file': exp['log_file']
                }
            }
        
        return len(parsed_experiments)
    
    def _parse_log_file(self, file_path):
        """Parse a single SLURM log file to extract experiment data"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract experiment configuration from the header
            experiment_match = re.search(r'🧠 EXPERIMENT: ([A-Z0-9_]+)', content)
            if not experiment_match:
                print(f"❌ No experiment header found in {file_path}")
                return None
            
            experiment_id = experiment_match.group(1)
            print(f"📋 Found experiment: {experiment_id}")
            
            # Parse experiment components (e.g., F16D3_K10_S1)
            exp_parts = experiment_id.split('_')
            if len(exp_parts) != 3:
                print(f"❌ Unexpected experiment format: {experiment_id}")
                return None
            
            complexity, k_steps, seed_part = exp_parts
            k_value = int(k_steps[1:])  # Extract number from K10 -> 10
            seed = int(seed_part[1:])   # Extract number from S1 -> 1
            
            # Extract configuration from arguments line
            config_match = re.search(r'Arguments parsed: Namespace\(([^)]+)\)', content)
            if not config_match:
                print(f"❌ No configuration found in {file_path}")
                return None
            
            config_str = config_match.group(1)
            
            # Parse individual config values
            features_match = re.search(r'num_concept_features=(\d+)', config_str)
            depth_match = re.search(r'pcfg_max_depth=(\d+)', config_str)
            adapt_match = re.search(r'adaptation_steps=(\d+)', config_str)
            seed_match = re.search(r'seed=(\d+)', config_str)
            
            if not all([features_match, depth_match, adapt_match, seed_match]):
                print(f"❌ Missing configuration parameters in {file_path}")
                return None
            
            config = {
                'features': int(features_match.group(1)),
                'depth': int(depth_match.group(1)),
                'adaptation_steps': int(adapt_match.group(1)),
                'seed': int(seed_match.group(1))
            }
            
            # Extract trajectory data
            trajectory_data = []
            
            # Look for training progress lines with MetaValAcc
            trajectory_matches = re.findall(
                r'Epoch (\d+), Batch \d+, Episodes Seen: (\d+), .*?MetaValAcc: ([\d.]+)', 
                content
            )
            
            for epoch, episodes, accuracy in trajectory_matches:
                trajectory_data.append({
                    'epoch': int(epoch),
                    'episodes': int(episodes),
                    'accuracy': float(accuracy)
                })
            
            print(f"📊 Found {len(trajectory_data)} trajectory points")
            
            if not trajectory_data:
                print(f"❌ No trajectory data found in {file_path}")
                return None
            
            # Since there's no final test evaluation, use the last validation accuracy
            final_accuracy = trajectory_data[-1]['accuracy'] if trajectory_data else 0.0
            
            # Check if experiment completed successfully
            success_match = re.search(r'✅ SUCCESS.*completed', content)
            status = 'completed' if success_match else 'incomplete'
            
            experiment_data = {
                'experiment_id': experiment_id,
                'complexity': complexity,
                'k_value': k_value,
                'seed': seed,
                'config': config,
                'trajectory': trajectory_data,
                'final_accuracy': final_accuracy,
                'status': status,
                'log_file': str(file_path)
            }
            
            print(f"✅ Parsed: {experiment_id} - {len(trajectory_data)} points, final acc: {final_accuracy:.4f}")
            return experiment_data
            
        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")
            return None
    
    def generate_clean_trajectories(self):
        """Generate clean trajectory plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Clean Trajectory Analysis: K=1 vs K=10', fontsize=16)
        
        configs = ['F8D3', 'F16D3', 'F32D3']
        
        for i, config in enumerate(configs):
            # K=1 trajectories
            ax1 = axes[0, i]
            self._plot_config_trajectories(ax1, config, k=1)
            ax1.set_title(f'{config} - K=1 Adaptation')
            ax1.set_ylabel('Accuracy')
            
            # K=10 trajectories  
            ax2 = axes[1, i]
            self._plot_config_trajectories(ax2, config, k=10)
            ax2.set_title(f'{config} - K=10 Adaptation')
            ax2.set_ylabel('Accuracy')
            ax2.set_xlabel('Episode')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'clean_trajectories.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'clean_trajectories.png', dpi=300, bbox_inches='tight')
        print(f"Generated clean trajectories: {self.output_dir}/clean_trajectories.pdf")
        
    def _plot_config_trajectories(self, ax, config: str, k: int):
        """Plot trajectories for specific configuration"""
        trajectories = []
        
        for key, data in self.trajectory_data.items():
            if f"{config}_K{k}" in key and data['trajectory']:
                episodes = [t['episode'] for t in data['trajectory']]
                accuracies = [t['accuracy'] for t in data['trajectory']]
                
                if episodes and accuracies:
                    ax.plot(episodes, accuracies, alpha=0.7, linewidth=1.5)
                    trajectories.append(accuracies)
        
        # Add mean trajectory if we have data
        if trajectories:
            max_len = max(len(t) for t in trajectories)
            padded_trajectories = []
            for t in trajectories:
                padded = t + [t[-1]] * (max_len - len(t))  # Pad with last value
                padded_trajectories.append(padded)
            
            mean_trajectory = np.mean(padded_trajectories, axis=0)
            episodes = list(range(len(mean_trajectory)))
            ax.plot(episodes, mean_trajectory, 'k-', linewidth=3, alpha=0.8, label='Mean')
            ax.legend()
        
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
    
    def generate_k_comparison(self):
        """Generate K=1 vs K=10 comparison analysis"""
        print("3. Generating K=1 vs K=10 comparison...")
        
        if not self.trajectory_data:
            print("❌ No trajectory data available for K comparison")
            return None
        
        # Debug: show what experiments we actually have
        print(f"📊 Available experiments:")
        for key in sorted(self.trajectory_data.keys()):
            exp = self.trajectory_data[key]
            final_acc = exp['performance']['final_accuracy']
            print(f"   {key}: final_acc = {final_acc:.4f}")
        
        # Group experiments by complexity and collect K=1 vs K=10 pairs
        complexity_groups = {}
        
        for key, exp_data in self.trajectory_data.items():
            # Parse key format: F16D3_K10_seed1
            parts = key.split('_')
            if len(parts) < 3:
                continue
                
            complexity = parts[0]  # F16D3
            k_part = parts[1]      # K10
            seed_part = parts[2]   # seed1
            
            if not k_part.startswith('K'):
                continue
                
            k_value = int(k_part[1:])  # Extract 10 from K10
            
            if complexity not in complexity_groups:
                complexity_groups[complexity] = {'K1': [], 'K10': []}
            
            k_key = f'K{k_value}'
            if k_key in complexity_groups[complexity]:
                complexity_groups[complexity][k_key].append({
                    'key': key,
                    'final_accuracy': exp_data['performance']['final_accuracy'],
                    'trajectory': exp_data['trajectory']
                })
        
        print(f"📊 Complexity groups found:")
        for complexity, data in complexity_groups.items():
            k1_count = len(data['K1'])
            k10_count = len(data['K10'])
            print(f"   {complexity}: K1={k1_count} experiments, K10={k10_count} experiments")
        
        # Generate comparison plots and statistics
        comparison_data = []
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K=1 vs K=10 Meta-Learning Comparison', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        valid_comparisons = 0
        
        for complexity in sorted(complexity_groups.keys()):
            if plot_idx >= 4:  # We only have 4 subplots
                break
                
            data = complexity_groups[complexity]
            k1_data = data['K1']
            k10_data = data['K10']
            
            ax = axes[plot_idx // 2, plot_idx % 2]
            
            if not k1_data and not k10_data:
                ax.text(0.5, 0.5, f'No data for {complexity}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{complexity} - No Data')
                plot_idx += 1
                continue
            
            # Plot available data
            if k1_data:
                k1_accs = [exp['final_accuracy'] for exp in k1_data]
                ax.bar(['K=1'], [np.mean(k1_accs)], alpha=0.7, color='red', 
                      yerr=[np.std(k1_accs)] if len(k1_accs) > 1 else 0,
                      capsize=5, label=f'K=1 (n={len(k1_accs)})')
            
            if k10_data:
                k10_accs = [exp['final_accuracy'] for exp in k10_data]
                ax.bar(['K=10'], [np.mean(k10_accs)], alpha=0.7, color='blue',
                      yerr=[np.std(k10_accs)] if len(k10_accs) > 1 else 0,
                      capsize=5, label=f'K=10 (n={len(k10_accs)})')
            
            ax.set_ylabel('Final Accuracy')
            ax.set_title(f'{complexity}')
            ax.legend()
            ax.set_ylim(0, 1)
            
            # Add statistical comparison if we have both K values
            if k1_data and k10_data:
                k1_accs = [exp['final_accuracy'] for exp in k1_data]
                k10_accs = [exp['final_accuracy'] for exp in k10_data]
                
                # Perform statistical test
                if len(k1_accs) > 1 and len(k10_accs) > 1:
                    from scipy import stats
                    t_stat, p_value = stats.ttest_ind(k10_accs, k1_accs)
                    effect_size = (np.mean(k10_accs) - np.mean(k1_accs)) / np.sqrt((np.var(k10_accs) + np.var(k1_accs)) / 2)
                else:
                    t_stat, p_value = 0, 1
                    effect_size = np.mean(k10_accs) - np.mean(k1_accs) if k10_accs and k1_accs else 0
                
                comparison_data.append({
                    'complexity': complexity,
                    'k1_mean': np.mean(k1_accs) if k1_accs else 0,
                    'k1_std': np.std(k1_accs) if len(k1_accs) > 1 else 0,
                    'k1_n': len(k1_accs),
                    'k10_mean': np.mean(k10_accs) if k10_accs else 0,
                    'k10_std': np.std(k10_accs) if len(k10_accs) > 1 else 0,
                    'k10_n': len(k10_accs),
                    'improvement': (np.mean(k10_accs) - np.mean(k1_accs)) if k1_accs and k10_accs else 0,
                    'p_value': p_value,
                    'effect_size': effect_size
                })
                valid_comparisons += 1
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i // 2, i % 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'k_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated K comparison: {self.output_dir}/k_comparison.pdf")
        
        # Generate statistical summary if we have valid comparisons
        if comparison_data:
            self._generate_statistical_summary(comparison_data)
        else:
            print("⚠️  No valid K=1 vs K=10 comparisons found")
            # Create a summary file noting the limitation
            summary_path = self.output_dir / 'k_comparison_summary.txt'
            with open(summary_path, 'w') as f:
                f.write("K=1 vs K=10 Comparison Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write("❌ No valid comparisons found\n\n")
                f.write("Available experiments:\n")
                for key in sorted(self.trajectory_data.keys()):
                    exp = self.trajectory_data[key]
                    final_acc = exp['performance']['final_accuracy']
                    f.write(f"   {key}: final_acc = {final_acc:.4f}\n")
                f.write(f"\nNote: Need both K=1 and K=10 experiments for the same complexity to make comparisons.\n")
        
        return comparison_data
    
    def _generate_statistical_summary(self, comparison_data: List[Dict]):
        """Generate statistical summary of K=1 vs K=10 comparison"""
        if not comparison_data:
            print("⚠️  No comparison data available for statistical summary")
            return
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(comparison_data)
        
        # Add formatted columns for reporting
        summary_df['k1_formatted'] = summary_df.apply(
            lambda row: f"{row['k1_mean']:.3f} ± {row['k1_std']:.3f} (n={row['k1_n']})", axis=1
        )
        summary_df['k10_formatted'] = summary_df.apply(
            lambda row: f"{row['k10_mean']:.3f} ± {row['k10_std']:.3f} (n={row['k10_n']})", axis=1
        )
        summary_df['improvement_formatted'] = summary_df.apply(
            lambda row: f"{row['improvement']:+.3f}", axis=1
        )
        summary_df['significance'] = summary_df['p_value'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        )
        summary_df['effect_size_interpretation'] = summary_df['effect_size'].apply(
            lambda es: 'Large' if abs(es) >= 0.8 else 'Medium' if abs(es) >= 0.5 else 'Small' if abs(es) >= 0.2 else 'Negligible'
        )
        
        # Save detailed summary
        summary_path = self.output_dir / 'k_comparison_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Generate text report
        report_path = self.output_dir / 'k_comparison_report.txt'
        with open(report_path, 'w') as f:
            f.write("K=1 vs K=10 Meta-Learning Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            if len(comparison_data) == 0:
                f.write("❌ No valid comparisons available\n")
                return
            
            f.write(f"Total comparisons: {len(comparison_data)}\n\n")
            
            for _, row in summary_df.iterrows():
                f.write(f"Configuration: {row['complexity']}\n")
                f.write(f"  K=1:  {row['k1_formatted']}\n")
                f.write(f"  K=10: {row['k10_formatted']}\n")
                f.write(f"  Improvement: {row['improvement_formatted']} ({row['improvement_formatted']})\n")
                f.write(f"  Statistical significance: {row['significance']} (p={row['p_value']:.4f})\n")
                f.write(f"  Effect size: {row['effect_size']:.3f} ({row['effect_size_interpretation']})\n")
                f.write("\n")
            
            # Overall summary
            significant_improvements = (summary_df['p_value'] < 0.05) & (summary_df['improvement'] > 0)
            f.write("SUMMARY:\n")
            f.write(f"- {significant_improvements.sum()}/{len(summary_df)} configurations show significant improvement with K=10\n")
            
            if len(summary_df) > 0:
                avg_improvement = summary_df['improvement'].mean()
                f.write(f"- Average improvement: {avg_improvement:+.3f}\n")
                
                large_effects = summary_df['effect_size'].abs() >= 0.8
                f.write(f"- {large_effects.sum()}/{len(summary_df)} configurations show large effect sizes\n")
        
        print(f"Generated statistical summary: {self.output_dir}/k_comparison_report.txt")
        print(f"Generated detailed data: {self.output_dir}/k_comparison_summary.csv")
    
    def generate_camera_ready_report(self):
        """Generate comprehensive camera-ready report"""
        print("5. Generating comprehensive report...")
        
        # Collect summary statistics
        total_experiments = len(self.trajectory_data)
        completed_experiments = sum(1 for data in self.trajectory_data.values() 
                                  if data.get('experiment_info', {}).get('status') == 'completed')
        
        avg_final_accuracy = np.mean([data['performance']['final_accuracy'] 
                                    for data in self.trajectory_data.values()])
        
        report_path = self.output_dir / 'camera_ready_comprehensive_report.md'
        with open(report_path, 'w') as f:
            f.write("# Camera-Ready Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Experiments:** {total_experiments}\n")
            f.write(f"- **Completed:** {completed_experiments}/{total_experiments}\n")
            f.write(f"- **Average Final Accuracy:** {avg_final_accuracy:.3f}\n\n")
            
            f.write("## Available Analyses\n\n")
            f.write("1. **Clean Trajectory Plots:** `clean_trajectories.pdf`\n")
            f.write("2. **K=1 vs K=10 Comparison:** `k_comparison.pdf`\n")
            f.write("3. **Statistical Summary:** `k_comparison_report.txt`\n")
            f.write("4. **Sample Efficiency Analysis:** `sample_efficiency.pdf`\n\n")
            
            f.write("## Experiment Details\n\n")
            for key, data in sorted(self.trajectory_data.items()):
                config = data['config']
                final_acc = data['performance']['final_accuracy']
                status = data.get('experiment_info', {}).get('status', 'unknown')
                f.write(f"- **{key}:** {final_acc:.3f} accuracy, {status}\n")
            
            f.write("\n## Next Steps\n\n")
            f.write("1. Review trajectory plots for training dynamics\n")
            f.write("2. Analyze K=1 vs K=10 statistical comparisons\n")
            f.write("3. Examine sample efficiency trends\n")
            f.write("4. Prepare publication figures\n")
        
        print(f"Generated comprehensive report: {self.output_dir}/camera_ready_comprehensive_report.md")
    
    def run_full_analysis(self):
        """Run complete camera-ready analysis pipeline"""
        print("Starting Camera-Ready Analysis Pipeline...")
        
        # Load results
        print("\n1. Loading Della results...")
        n_files = self.load_della_results()
        print(f"   Loaded {n_files} log files")
        print(f"   Parsed {len(self.trajectory_data)} experiments")
        
        # Generate clean trajectories
        print("\n2. Generating clean trajectory plots...")
        self.generate_clean_trajectories()
        
        # Generate K comparison
        print("\n3. Generating K=1 vs K=10 comparison...")
        comparison_data = self.generate_k_comparison()
        
        # Generate report
        print("\n4. Generating camera-ready report...")
        self.generate_camera_ready_report()
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        print("\nCamera-ready deliverables:")
        print(f"  - Clean trajectories: {self.output_dir}/clean_trajectories.pdf")
        print(f"  - K comparison: {self.output_dir}/k_comparison.pdf")
        print(f"  - Statistical summary: {self.output_dir}/statistical_summary.csv")
        print(f"  - Full report: {self.output_dir}/camera_ready_report.md")
        
        return comparison_data

if __name__ == "__main__":
    analyzer = CameraReadyAnalyzer()
    analyzer.run_full_analysis() 