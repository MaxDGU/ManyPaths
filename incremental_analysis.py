#!/usr/bin/env python3
"""
Incremental Analysis Script for Camera-Ready Submission
====================================================

Processes results incrementally as they come in from focused_camera_ready_array.slurm
Allows analysis of F8D3 → F16D3 → F32D3 progressively without waiting for all results.

Usage:
    python incremental_analysis.py --phase 1  # F8D3 results
    python incremental_analysis.py --phase 2  # F8D3 + F16D3 results  
    python incremental_analysis.py --phase 3  # All results (F32D3 may be partial)

Author: Camera-Ready Pipeline
Date: December 2024
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import json
import re
from datetime import datetime
import subprocess

# Expected experiment configurations from focused_camera_ready_array.slurm
EXPECTED_CONFIGS = {
    'F8D3': {'features': 8, 'depth': 3, 'seeds': [1, 2, 3], 'adaptation_steps': [1, 10]},
    'F16D3': {'features': 16, 'depth': 3, 'seeds': [1, 2, 3], 'adaptation_steps': [1, 10]},
    'F32D3': {'features': 32, 'depth': 3, 'seeds': [1, 2, 3], 'adaptation_steps': [1, 10]}
}

# Time estimates for completion (in hours)
TIME_ESTIMATES = {
    'F8D3': {'K1': 20, 'K10': 30},
    'F16D3': {'K1': 40, 'K10': 60},
    'F32D3': {'K1': 60, 'K10': 70}
}

@dataclass
class IncrementalResult:
    """Container for incremental analysis results"""
    phase: int
    configs_analyzed: List[str]
    total_experiments: int
    completed_experiments: int
    preliminary_insights: Dict[str, Any]
    figures_generated: List[str]
    recommendations: List[str]

class IncrementalAnalyzer:
    """Incremental analysis for camera-ready submission"""
    
    def __init__(self, della_results_dir: str = "/scratch/gpfs/mg7411/ManyPaths/results", 
                 local_results_dir: str = "results",
                 output_dir: str = "incremental_analysis"):
        self.della_results_dir = Path(della_results_dir)
        self.local_results_dir = Path(local_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis state
        self.discovered_experiments = {}
        self.trajectory_data = {}
        self.final_accuracies = {}
        self.statistical_results = {}
        
        print(f"📊 Incremental Analyzer initialized")
        print(f"   Della results: {self.della_results_dir}")
        print(f"   Local results: {self.local_results_dir}")
        print(f"   Output: {self.output_dir}")
    
    def check_della_job_status(self) -> Dict[str, Any]:
        """Check status of della jobs"""
        try:
            # Check SLURM queue
            result = subprocess.run(['squeue', '-u', 'mg7411', '--format=%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                camera_ready_jobs = [line for line in lines if 'camera_ready' in line]
                
                job_status = {
                    'total_jobs': len(camera_ready_jobs),
                    'running_jobs': len([line for line in camera_ready_jobs if ' R ' in line]),
                    'pending_jobs': len([line for line in camera_ready_jobs if ' PD ' in line]),
                    'completed_jobs': 18 - len(camera_ready_jobs)  # Total expected - currently queued
                }
                
                return job_status
            else:
                return {'error': 'Could not check job status'}
                
        except Exception as e:
            return {'error': f'Job status check failed: {e}'}
    
    def discover_available_results(self, phase: int) -> Dict[str, List[str]]:
        """Discover what results are available for this phase"""
        configs_to_check = []
        
        if phase >= 1:
            configs_to_check.append('F8D3')
        if phase >= 2:
            configs_to_check.append('F16D3')
        if phase >= 3:
            configs_to_check.append('F32D3')
        
        discovered = {}
        
        for config in configs_to_check:
            config_info = EXPECTED_CONFIGS[config]
            found_experiments = []
            
            # Search for result files
            for seed in config_info['seeds']:
                for k in config_info['adaptation_steps']:
                    # Pattern matching for trajectory files
                    pattern = f"*feats{config_info['features']}_depth{config_info['depth']}_adapt{k}*seed{seed}*"
                    
                    # Check local results first
                    local_files = list(self.local_results_dir.rglob(pattern))
                    if local_files:
                        found_experiments.extend(local_files)
                    
                    # If running on della, check della results
                    if self.della_results_dir.exists():
                        della_files = list(self.della_results_dir.rglob(pattern))
                        found_experiments.extend(della_files)
            
            discovered[config] = found_experiments
        
        return discovered
    
    def load_trajectory_data(self, discovered_files: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """Load trajectory data from discovered files"""
        trajectory_data = {}
        
        for config, files in discovered_files.items():
            for file_path in files:
                if 'trajectory' in file_path.name and file_path.suffix == '.csv':
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Extract experiment info from filename
                        exp_info = self.parse_experiment_info(file_path.name)
                        if exp_info:
                            exp_key = f"{config}_K{exp_info['adaptation_steps']}_S{exp_info['seed']}"
                            trajectory_data[exp_key] = df
                            print(f"   ✅ Loaded {exp_key}: {len(df)} episodes")
                    
                    except Exception as e:
                        print(f"   ❌ Failed to load {file_path}: {e}")
        
        return trajectory_data
    
    def parse_experiment_info(self, filename: str) -> Optional[Dict[str, int]]:
        """Parse experiment information from filename"""
        # Pattern: *feats{F}_depth{D}_adapt{K}*seed{S}*
        pattern = r"feats(\d+)_depth(\d+)_adapt(\d+).*seed(\d+)"
        match = re.search(pattern, filename)
        
        if match:
            return {
                'features': int(match.group(1)),
                'depth': int(match.group(2)),
                'adaptation_steps': int(match.group(3)),
                'seed': int(match.group(4))
            }
        
        return None
    
    def compute_completion_status(self, discovered_files: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute completion status for each configuration"""
        completion_status = {}
        
        for config, files in discovered_files.items():
            expected_experiments = len(EXPECTED_CONFIGS[config]['seeds']) * len(EXPECTED_CONFIGS[config]['adaptation_steps'])
            completed_experiments = len([f for f in files if 'trajectory' in f.name])
            
            completion_status[config] = {
                'expected': expected_experiments,
                'completed': completed_experiments,
                'completion_rate': completed_experiments / expected_experiments if expected_experiments > 0 else 0,
                'missing_experiments': expected_experiments - completed_experiments
            }
        
        return completion_status
    
    def analyze_k1_vs_k10_incremental(self, trajectory_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze K=1 vs K=10 with available data"""
        # Group by configuration
        config_results = {}
        
        for exp_key, df in trajectory_data.items():
            # Parse experiment key: F8D3_K1_S1
            parts = exp_key.split('_')
            if len(parts) >= 3:
                config = parts[0]  # F8D3
                method = parts[1]  # K1 or K10
                
                if config not in config_results:
                    config_results[config] = {'K1': [], 'K10': []}
                
                # Compute final accuracy
                if 'val_accuracy' in df.columns and len(df) > 0:
                    final_acc = df['val_accuracy'].tail(10).mean()  # Last 10 episodes
                    config_results[config][method].append(final_acc)
        
        # Compute statistics
        statistical_results = {}
        
        for config, results in config_results.items():
            k1_results = results['K1']
            k10_results = results['K10']
            
            if len(k1_results) > 0 and len(k10_results) > 0:
                # Basic statistics
                k1_mean = np.mean(k1_results)
                k10_mean = np.mean(k10_results)
                improvement = k10_mean - k1_mean
                
                # Statistical test if sufficient data
                if len(k1_results) > 1 and len(k10_results) > 1:
                    t_stat, p_value = stats.ttest_ind(k10_results, k1_results)
                    effect_size = self.compute_cohens_d(k10_results, k1_results)
                else:
                    t_stat, p_value = None, None
                    effect_size = None
                
                statistical_results[config] = {
                    'K1_mean': k1_mean,
                    'K1_std': np.std(k1_results),
                    'K1_n': len(k1_results),
                    'K10_mean': k10_mean,
                    'K10_std': np.std(k10_results),
                    'K10_n': len(k10_results),
                    'improvement': improvement,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size
                }
        
        return statistical_results
    
    def compute_cohens_d(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Cohen's d effect size"""
        if len(x1) <= 1 or len(x2) <= 1:
            return None
        
        n1, n2 = len(x1), len(x2)
        s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (np.mean(x1) - np.mean(x2)) / pooled_std
        return d
    
    def generate_preliminary_insights(self, phase: int, statistical_results: Dict[str, Any], 
                                    completion_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate preliminary insights for this phase"""
        insights = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'completion_summary': completion_status,
            'statistical_findings': {},
            'trends': [],
            'recommendations': []
        }
        
        # Analyze statistical findings
        for config, stats in statistical_results.items():
            insights['statistical_findings'][config] = {
                'improvement': stats['improvement'],
                'significant': stats['p_value'] < 0.05 if stats['p_value'] else False,
                'effect_size': stats['effect_size'],
                'confidence': 'high' if stats['K1_n'] >= 2 and stats['K10_n'] >= 2 else 'low'
            }
        
        # Identify trends
        if len(statistical_results) > 1:
            configs_ordered = sorted(statistical_results.keys())
            improvements = [statistical_results[c]['improvement'] for c in configs_ordered]
            
            if len(improvements) >= 2:
                if improvements[-1] > improvements[0]:
                    insights['trends'].append("Improvements increase with complexity")
                else:
                    insights['trends'].append("Improvements decrease with complexity")
        
        # Generate recommendations
        if phase == 1:
            insights['recommendations'].append("F8D3 analysis complete - proceed to F16D3")
        elif phase == 2:
            insights['recommendations'].append("F8D3+F16D3 analysis complete - wait for F32D3")
        elif phase == 3:
            insights['recommendations'].append("Begin final analysis integration")
        
        return insights
    
    def create_phase_visualizations(self, phase: int, trajectory_data: Dict[str, pd.DataFrame], 
                                  statistical_results: Dict[str, Any]):
        """Create visualizations for this phase"""
        figures_generated = []
        
        # 1. Learning curves for available configurations
        if trajectory_data:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = {'K1': '#FF6B6B', 'K10': '#4ECDC4'}
            
            for exp_key, df in trajectory_data.items():
                # Parse key
                parts = exp_key.split('_')
                config = parts[0]
                method = parts[1]
                seed = parts[2]
                
                color = colors.get(method, 'gray')
                alpha = 0.7
                
                episodes = range(len(df))
                ax.plot(episodes, df['val_accuracy'], 
                       color=color, alpha=alpha, linewidth=2,
                       label=f"{config} {method} {seed}")
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Validation Accuracy')
            ax.set_title(f'Phase {phase} Learning Curves')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f"phase_{phase}_learning_curves.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            figures_generated.append(filename)
        
        # 2. K1 vs K10 comparison for available data
        if statistical_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            configs = list(statistical_results.keys())
            k1_means = [statistical_results[c]['K1_mean'] for c in configs]
            k10_means = [statistical_results[c]['K10_mean'] for c in configs]
            k1_stds = [statistical_results[c]['K1_std'] for c in configs]
            k10_stds = [statistical_results[c]['K10_std'] for c in configs]
            
            x = np.arange(len(configs))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, k1_means, width, yerr=k1_stds, 
                          label='K=1', color='#FF6B6B', alpha=0.8)
            bars2 = ax.bar(x + width/2, k10_means, width, yerr=k10_stds, 
                          label='K=10', color='#4ECDC4', alpha=0.8)
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Final Validation Accuracy')
            ax.set_title(f'Phase {phase} K=1 vs K=10 Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(configs)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f"phase_{phase}_k1_vs_k10.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            figures_generated.append(filename)
        
        return figures_generated
    
    def generate_phase_report(self, result: IncrementalResult):
        """Generate phase report"""
        report = f"""
# Phase {result.phase} Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Phase**: {result.phase}/3
- **Configurations analyzed**: {', '.join(result.configs_analyzed)}
- **Total experiments**: {result.total_experiments}
- **Completed experiments**: {result.completed_experiments}
- **Completion rate**: {result.completed_experiments/result.total_experiments*100:.1f}%

## Statistical Findings
"""
        
        for config, findings in result.preliminary_insights['statistical_findings'].items():
            significance = "✅ Significant" if findings['significant'] else "❌ Not significant"
            report += f"""
### {config}
- **Improvement**: {findings['improvement']:.3f}
- **Significance**: {significance}
- **Effect size**: {findings['effect_size']:.2f if findings['effect_size'] else 'N/A'}
- **Confidence**: {findings['confidence']}
"""
        
        report += f"""
## Trends Identified
"""
        for trend in result.preliminary_insights['trends']:
            report += f"- {trend}\n"
        
        report += f"""
## Recommendations
"""
        for rec in result.recommendations:
            report += f"- {rec}\n"
        
        report += f"""
## Figures Generated
"""
        for fig in result.figures_generated:
            report += f"- {fig}\n"
        
        # Save report
        with open(self.output_dir / f"phase_{result.phase}_report.md", 'w') as f:
            f.write(report)
    
    def run_phase_analysis(self, phase: int) -> IncrementalResult:
        """Run analysis for specified phase"""
        print(f"🚀 Running Phase {phase} Analysis...")
        
        # Check job status
        job_status = self.check_della_job_status()
        if 'error' not in job_status:
            print(f"   📊 Job status: {job_status}")
        
        # Discover available results
        discovered_files = self.discover_available_results(phase)
        print(f"   📁 Discovered files: {sum(len(files) for files in discovered_files.values())}")
        
        # Load trajectory data
        trajectory_data = self.load_trajectory_data(discovered_files)
        
        # Compute completion status
        completion_status = self.compute_completion_status(discovered_files)
        
        # Analyze K=1 vs K=10
        statistical_results = self.analyze_k1_vs_k10_incremental(trajectory_data)
        
        # Generate insights
        insights = self.generate_preliminary_insights(phase, statistical_results, completion_status)
        
        # Create visualizations
        figures = self.create_phase_visualizations(phase, trajectory_data, statistical_results)
        
        # Generate recommendations
        recommendations = []
        if phase == 1 and completion_status.get('F8D3', {}).get('completion_rate', 0) >= 0.8:
            recommendations.append("F8D3 analysis complete - proceed to Phase 2")
        elif phase == 2 and completion_status.get('F16D3', {}).get('completion_rate', 0) >= 0.8:
            recommendations.append("F16D3 analysis complete - proceed to Phase 3")
        elif phase == 3:
            recommendations.append("Begin final integration with master analysis")
        
        # Create result object
        result = IncrementalResult(
            phase=phase,
            configs_analyzed=list(discovered_files.keys()),
            total_experiments=sum(len(files) for files in discovered_files.values()),
            completed_experiments=len(trajectory_data),
            preliminary_insights=insights,
            figures_generated=figures,
            recommendations=recommendations
        )
        
        # Generate report
        self.generate_phase_report(result)
        
        print(f"✅ Phase {phase} analysis complete!")
        return result

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Incremental Analysis for Camera-Ready Submission')
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3],
                       help='Analysis phase (1=F8D3, 2=F8D3+F16D3, 3=All)')
    parser.add_argument('--della_results', default='/scratch/gpfs/mg7411/ManyPaths/results',
                       help='Della results directory')
    parser.add_argument('--local_results', default='results',
                       help='Local results directory')
    parser.add_argument('--output_dir', default='incremental_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = IncrementalAnalyzer(args.della_results, args.local_results, args.output_dir)
    
    # Run phase analysis
    result = analyzer.run_phase_analysis(args.phase)
    
    print(f"\n🎉 Phase {args.phase} analysis complete!")
    print(f"📁 Results saved to: {analyzer.output_dir}")
    print(f"📊 Experiments analyzed: {result.completed_experiments}/{result.total_experiments}")
    print(f"📈 Figures generated: {len(result.figures_generated)}")
    
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   - {rec}")

if __name__ == "__main__":
    main() 