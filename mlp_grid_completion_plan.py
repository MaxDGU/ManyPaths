#!/usr/bin/env python3
"""
MLP Grid Completion Plan - Camera-Ready Focus

Strategic experiment planning for MLP-only concept learning grid:
- Complete F8-F32 × D3-D7 grid with robust statistics
- Focus purely on MLP architecture for consistency  
- Prioritize by scientific necessity for camera-ready
"""

import json
from datetime import datetime

def create_mlp_grid_experiments():
    """Create focused MLP grid completion experiments."""
    
    # Current status (from della analysis)
    current_status = {
        "F8_D3": {"k1_seeds": 6, "k10_seeds": 4, "status": "complete"},
        "F8_D5": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial"},
        "F8_D7": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial"},
        "F16_D3": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial"},
        "F16_D5": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial"},
        "F16_D7": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial"},
        "F32_D3": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial"},
        "F32_D5": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial"},
        "F32_D7": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial"}
    }
    
    # Target seeds for robust statistics
    target_seeds = 4  # Minimum for good confidence intervals
    
    extension_experiments = []
    
    # Define grid configurations by scientific priority
    grid_configs = [
        # PRIORITY 1: Core complexity scaling (features)
        {"features": 16, "depth": 3, "priority": "P1_CORE", "justification": "Feature complexity scaling"},
        {"features": 32, "depth": 3, "priority": "P1_CORE", "justification": "High feature complexity"},
        
        # PRIORITY 2: Depth scaling (logical complexity)  
        {"features": 8, "depth": 5, "priority": "P2_DEPTH", "justification": "Simple features, medium depth"},
        {"features": 16, "depth": 5, "priority": "P2_DEPTH", "justification": "Medium features, medium depth"},
        {"features": 32, "depth": 5, "priority": "P2_DEPTH", "justification": "High features, medium depth"},
        
        # PRIORITY 3: Maximum complexity
        {"features": 8, "depth": 7, "priority": "P3_MAX", "justification": "Simple features, max depth"},
        {"features": 16, "depth": 7, "priority": "P3_MAX", "justification": "Medium features, max depth"},  
        {"features": 32, "depth": 7, "priority": "P3_MAX", "justification": "Maximum complexity"}
    ]
    
    for config in grid_configs:
        config_name = f"F{config['features']}_D{config['depth']}"
        current = current_status[config_name]
        
        # Calculate needed experiments
        k1_needed = max(0, target_seeds - current["k1_seeds"])
        k10_needed = max(0, target_seeds - current["k10_seeds"])
        
        # Time estimates based on complexity
        base_time_k1 = 3600 + (config['features'] * 100) + (config['depth'] * 600)  # Scale with complexity
        base_time_k10 = base_time_k1 * 1.5  # K=10 takes ~50% longer
        
        # Add K=1 experiments
        for seed in range(current["k1_seeds"], target_seeds):
            extension_experiments.append({
                "name": f"k1_f{config['features']}d{config['depth']}_s{seed}",
                "experiment": "concept",
                "model": "mlp",
                "features": config['features'],
                "depth": config['depth'],
                "adaptation_steps": 1,
                "epochs": 200 if config['priority'] == 'P1_CORE' else 150,  # More epochs for core
                "seed": seed,
                "timeout": int(base_time_k1),
                "priority": config['priority'],
                "config": config_name,
                "justification": config['justification']
            })
        
        # Add K=10 experiments
        for seed in range(current["k10_seeds"], target_seeds):
            extension_experiments.append({
                "name": f"k10_f{config['features']}d{config['depth']}_s{seed}",
                "experiment": "concept", 
                "model": "mlp",
                "features": config['features'],
                "depth": config['depth'],
                "adaptation_steps": 10,
                "epochs": 200 if config['priority'] == 'P1_CORE' else 150,
                "seed": seed,
                "timeout": int(base_time_k10),
                "priority": config['priority'],
                "config": config_name,
                "justification": config['justification']
            })
    
    return extension_experiments

def analyze_mlp_grid_requirements():
    """Analyze computational requirements for MLP grid completion."""
    
    experiments = create_mlp_grid_experiments()
    
    # Group by priority
    by_priority = {}
    by_config = {}
    
    for exp in experiments:
        priority = exp["priority"]
        config = exp["config"]
        
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(exp)
        
        if config not in by_config:
            by_config[config] = {"k1": 0, "k10": 0, "total_time": 0}
        
        if exp["adaptation_steps"] == 1:
            by_config[config]["k1"] += 1
        else:
            by_config[config]["k10"] += 1
        by_config[config]["total_time"] += exp["timeout"] / 3600
    
    analysis = {
        "total_experiments": len(experiments),
        "total_time_hours": sum(exp["timeout"] for exp in experiments) / 3600,
        "by_priority": {},
        "by_config": by_config,
        "statistical_impact": {}
    }
    
    # Priority analysis
    for priority, exps in by_priority.items():
        total_time = sum(exp["timeout"] for exp in exps) / 3600
        configs = list(set(exp["config"] for exp in exps))
        
        analysis["by_priority"][priority] = {
            "count": len(exps),
            "total_time_hours": total_time,
            "configurations": configs,
            "description": {
                "P1_CORE": "Essential complexity scaling evidence",
                "P2_DEPTH": "Logical complexity (depth) scaling",
                "P3_MAX": "Maximum complexity validation"
            }.get(priority, "Unknown")
        }
    
    # Statistical impact analysis
    analysis["statistical_impact"] = {
        "current_robust": ["F8_D3"],  # Only this has 4+ seeds
        "after_p1": ["F8_D3", "F16_D3", "F32_D3"],  # Feature scaling covered
        "after_p2": ["F8_D3", "F16_D3", "F32_D3", "F8_D5", "F16_D5", "F32_D5"],  # + depth scaling
        "after_p3": ["F8_D3", "F16_D3", "F32_D3", "F8_D5", "F16_D5", "F32_D5", "F8_D7", "F16_D7", "F32_D7"]  # Complete
    }
    
    return analysis, experiments

def create_mlp_slurm_scripts():
    """Create priority-based SLURM scripts for MLP experiments."""
    
    analysis, experiments = analyze_mlp_grid_requirements()
    
    # Group by priority
    by_priority = {}
    for exp in experiments:
        priority = exp["priority"]
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(exp)
    
    slurm_scripts = {}
    priority_names = {
        "P1_CORE": "core_complexity",
        "P2_DEPTH": "depth_scaling", 
        "P3_MAX": "max_complexity"
    }
    
    for priority, exps in by_priority.items():
        script_name = f"run_mlp_{priority_names[priority]}.slurm"
        
        # Time estimate with 25% buffer
        total_time_seconds = sum(exp["timeout"] for exp in exps) * 1.25
        time_hours = max(1, int(total_time_seconds / 3600) + 1)
        
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=mlp_{priority_names[priority]}
#SBATCH --output=mlp_{priority_names[priority]}_%j.out
#SBATCH --error=mlp_{priority_names[priority]}_%j.err
#SBATCH --time={time_hours:02d}:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "MLP GRID COMPLETION - {priority} ({priority_names[priority].upper()})"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Priority: {analysis['by_priority'][priority]['description']}"
echo "Experiments: {len(exps)}, Est. time: {time_hours}h"
echo "Configurations: {', '.join(analysis['by_priority'][priority]['configurations'])}"

# Load modules
module load anaconda3/2023.3
module load cudatoolkit/11.8

# Activate conda environment
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

"""
        
        for i, exp in enumerate(exps):
            timeout_hours = exp['timeout'] / 3600
            
            cmd_args = [
                f"--experiment {exp['experiment']}",
                f"--m {exp['model']}",
                "--data-type bits",
                f"--num-concept-features {exp['features']}",
                f"--pcfg-max-depth {exp['depth']}",
                f"--adaptation-steps {exp['adaptation_steps']}",
                f"--epochs {exp['epochs']}",
                f"--seed {exp['seed']}",
                "--save",
                "--no_hyper_search",
                "--first-order"
            ]
            
            slurm_content += f"""
echo "🧠 EXPERIMENT {i+1}/{len(exps)}: {exp['name']}"
echo "   Config: {exp['config']}, K={exp['adaptation_steps']}, Seed={exp['seed']}"
echo "   Timeout: {timeout_hours:.1f}h, Justification: {exp['justification']}"
echo "   Command: python main.py {' '.join(cmd_args)}"
timeout {exp['timeout']} python main.py {' '.join(cmd_args)}
echo "   ✅ Completed experiment {i+1}/{len(exps)} at $(date)"
echo "   =========================================="
"""
        
        slurm_content += f"""
echo "============================================"
echo "🎯 {priority_names[priority].upper()} PRIORITY COMPLETED"
echo "Job finished at: $(date)"
echo "Experiments completed: {len(exps)}"
echo "Statistical coverage: {', '.join(analysis['by_priority'][priority]['configurations'])}"
echo "============================================"
"""
        
        slurm_scripts[script_name] = slurm_content
    
    return slurm_scripts

def main():
    """Generate focused MLP grid completion plan."""
    
    print("🧠 MLP GRID COMPLETION PLAN - CAMERA-READY FOCUS")
    print("=" * 80)
    
    # Generate analysis
    analysis, experiments = analyze_mlp_grid_requirements()
    
    print(f"📊 MLP GRID REQUIREMENTS")
    print(f"Total experiments needed: {analysis['total_experiments']}")
    print(f"Total compute time: {analysis['total_time_hours']:.1f} hours")
    print(f"Focus: MLP-only for consistent explainability")
    print()
    
    # Priority breakdown
    for priority, data in analysis["by_priority"].items():
        print(f"🎯 {priority} - {data['description']}:")
        print(f"   Experiments: {data['count']}")
        print(f"   Compute time: {data['total_time_hours']:.1f} hours")
        print(f"   Configurations: {', '.join(data['configurations'])}")
        print()
    
    # Statistical impact
    print("📈 STATISTICAL IMPACT:")
    impact = analysis["statistical_impact"]
    print(f"   Currently robust: {len(impact['current_robust'])} configs")
    print(f"   After P1 (core): {len(impact['after_p1'])} configs")
    print(f"   After P2 (depth): {len(impact['after_p2'])} configs") 
    print(f"   After P3 (complete): {len(impact['after_p3'])} configs")
    print()
    
    # Configuration breakdown
    print("📋 CONFIGURATION REQUIREMENTS:")
    for config, data in analysis["by_config"].items():
        print(f"   {config}: +{data['k1']} K=1, +{data['k10']} K=10 ({data['total_time']:.1f}h)")
    print()
    
    # Generate SLURM scripts
    print("🚀 GENERATING MLP-FOCUSED SLURM SCRIPTS...")
    slurm_scripts = create_mlp_slurm_scripts()
    
    # Save everything
    import os
    os.makedirs('mlp_grid_completion', exist_ok=True)
    
    # Save experiment plan
    with open('mlp_grid_completion/mlp_experiments.json', 'w') as f:
        json.dump(experiments, f, indent=2)
    
    # Save analysis
    with open('mlp_grid_completion/mlp_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save SLURM scripts
    for script_name, content in slurm_scripts.items():
        with open(f'mlp_grid_completion/{script_name}', 'w') as f:
            f.write(content)
    
    print(f"✅ MLP grid completion plan saved to mlp_grid_completion/")
    print(f"📁 Generated files:")
    print(f"   - mlp_experiments.json: Complete MLP experiment specifications")
    print(f"   - mlp_analysis.json: Requirements and impact analysis")
    for script_name in slurm_scripts.keys():
        priority_desc = script_name.split('_')[2].replace('.slurm', '')
        print(f"   - {script_name}: {priority_desc.upper()} priority experiments")
    
    print(f"\n🎯 RECOMMENDED EXECUTION ORDER:")
    print(f"1. P1_CORE: Feature complexity scaling (F16_D3, F32_D3)")
    print(f"2. P2_DEPTH: Logical complexity scaling (D5 across features)")
    print(f"3. P3_MAX: Maximum complexity validation (D7 across features)")
    print(f"\n🧠 MLP-ONLY BENEFITS:")
    print(f"   ✅ Consistent architecture for clean comparisons")
    print(f"   ✅ Better explainability without architectural confounds")
    print(f"   ✅ More compute budget for statistical robustness")
    
    return analysis, experiments

if __name__ == "__main__":
    main() 