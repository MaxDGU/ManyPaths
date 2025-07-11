#!/usr/bin/env python3
"""
Della Landscape Analysis Script

Run landscape analysis directly on della without needing to pull files locally.
This script looks for landscape trajectory files in the correct della paths.

Usage:
    python della_landscape_analysis.py
"""

import os
import sys
import glob
from pathlib import Path

# Add the current directory to path for imports
sys.path.insert(0, '/scratch/gpfs/mg7411/ManyPaths')

# Import the main landscape creation functions
from create_concept_landscapes import (
    load_landscape_files, 
    create_concept_complexity_landscapes,
    create_3d_loss_landscapes, 
    create_comparative_analysis
)

def check_della_environment():
    """Check if we're running on della and files exist"""
    print("🔍 Checking della environment...")
    
    # Check if we're on della
    hostname = os.uname().nodename
    print(f"Hostname: {hostname}")
    
    if 'della' not in hostname:
        print("⚠️  Warning: Not running on della cluster")
    
    # Check working directory
    cwd = os.getcwd()
    print(f"Working directory: {cwd}")
    
    # Check for landscape trajectory files in results directory
    results_pattern = "/scratch/gpfs/mg7411/ManyPaths/results/*landscape_trajectory.csv"
    local_results_pattern = "results/*landscape_trajectory.csv"
    
    results_files = glob.glob(results_pattern)
    local_results_files = glob.glob(local_results_pattern)
    
    print(f"\n📁 File discovery:")
    print(f"  Files in /scratch/gpfs/mg7411/ManyPaths/results/: {len(results_files)}")
    print(f"  Files in ./results/: {len(local_results_files)}")
    
    # Show found files
    all_files = results_files + local_results_files
    for file in all_files[:10]:  # Show first 10
        print(f"    {file}")
    
    if len(all_files) > 10:
        print(f"    ... and {len(all_files) - 10} more files")
    
    # Check for log files
    logs_pattern = "/scratch/gpfs/mg7411/ManyPaths/logs/mp_landscape_overnight_*.out"
    local_logs_pattern = "logs/mp_landscape_overnight_*.out"
    
    log_files = glob.glob(logs_pattern) + glob.glob(local_logs_pattern)
    print(f"  Log files found: {len(log_files)}")
    
    for file in log_files[:5]:  # Show first 5 log files
        print(f"    {file}")
    
    return len(all_files) > 0

def main():
    """Main analysis pipeline for della"""
    
    print("🌄 DELLA LANDSCAPE ANALYSIS")
    print("=" * 50)
    
    # Check environment
    if not check_della_environment():
        print("❌ No landscape trajectory files found!")
        print("\nExpected locations:")
        print("  - /scratch/gpfs/mg7411/ManyPaths/results/*landscape_trajectory.csv")
        print("  - ./results/*landscape_trajectory.csv")
        return 1
    
    # Ensure figures directory exists
    Path('figures').mkdir(exist_ok=True)
    
    print("\n🔍 Loading landscape trajectory files...")
    datasets = load_landscape_files()
    
    if not datasets:
        print("❌ No valid landscape files could be loaded!")
        return 1
    
    print(f"\n📊 Analyzing {len(datasets)} experiments...")
    
    # Show what we found
    configs = {}
    for key, data in datasets.items():
        config = data['config']
        if config not in configs:
            configs[config] = []
        configs[config].append(data['seed'])
    
    print("\n📋 Experiment summary:")
    for config, seeds in configs.items():
        print(f"  {config}: {len(seeds)} seeds ({sorted(seeds)})")
    
    # Create different types of landscape visualizations
    try:
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
        
        # List all generated files
        figure_files = glob.glob("figures/*.pdf")
        print(f"\n📁 Total files generated: {len(figure_files)}")
        for file in sorted(figure_files):
            print(f"    {file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 