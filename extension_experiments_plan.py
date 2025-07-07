#!/usr/bin/env python3
"""
Extension Experiments Plan for Features×Depth Grid Completion

Strategic experiment planning to complete the concept learning grid:
- Fill missing seeds for statistical robustness  
- Complete feature×depth combinations
- Add architectural comparisons
- Prioritize for camera-ready submission
"""

import json
from datetime import datetime

def create_extension_experiments():
    """Create comprehensive extension experiment plan."""
    
    # Current status analysis (from della analysis)
    current_status = {
        "F8_D3": {"k1_seeds": 6, "k10_seeds": 4, "status": "complete", "priority": "done"},
        "F8_D5": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial", "priority": "medium"},
        "F8_D7": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial", "priority": "low"},
        "F16_D3": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial", "priority": "urgent"},
        "F16_D5": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial", "priority": "medium"},
        "F16_D7": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial", "priority": "low"},
        "F32_D3": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial", "priority": "urgent"},
        "F32_D5": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial", "priority": "medium"},
        "F32_D7": {"k1_seeds": 1, "k10_seeds": 1, "status": "partial", "priority": "low"}
    }
    
    # Target: 3-4 seeds minimum for robust statistics
    target_seeds = 4
    
    extension_experiments = []
    
    # BATCH 1: URGENT - Core complexity sweep (F16_D3, F32_D3)
    urgent_configs = [
        {"features": 16, "depth": 3, "config": "F16_D3", "justification": "Core complexity scaling"},
        {"features": 32, "depth": 3, "config": "F32_D3", "justification": "High complexity baseline"}
    ]
    
    for config in urgent_configs:
        current = current_status[config["config"]]
        k1_needed = max(0, target_seeds - current["k1_seeds"])
        k10_needed = max(0, target_seeds - current["k10_seeds"])
        
        # Add K=1 experiments
        for seed in range(2, 2 + k1_needed):  # Start from seed 2 (already have 0, 1)
            extension_experiments.append({
                "name": f"k1_{config['config'].lower()}_s{seed}",
                "experiment": "concept",
                "model": "mlp",
                "features": config["features"],
                "depth": config["depth"],
                "adaptation_steps": 1,
                "epochs": 200,
                "seed": seed,
                "timeout": 7200,  # 2 hours
                "priority": "URGENT",
                "batch": "core_complexity",
                "justification": config["justification"]
            })
        
        # Add K=10 experiments  
        for seed in range(2, 2 + k10_needed):
            extension_experiments.append({
                "name": f"k10_{config['config'].lower()}_s{seed}",
                "experiment": "concept",
                "model": "mlp", 
                "features": config["features"],
                "depth": config["depth"],
                "adaptation_steps": 10,
                "epochs": 200,
                "seed": seed,
                "timeout": 10800,  # 3 hours
                "priority": "URGENT",
                "batch": "core_complexity",
                "justification": config["justification"]
            })
    
    # BATCH 2: HIGH - Extended complexity grid  
    high_configs = [
        {"features": 8, "depth": 5, "config": "F8_D5", "justification": "Depth scaling for simple features"},
        {"features": 16, "depth": 5, "config": "F16_D5", "justification": "Medium complexity scaling"},
        {"features": 32, "depth": 5, "config": "F32_D5", "justification": "High complexity scaling"}
    ]
    
    for config in high_configs:
        current = current_status[config["config"]]
        # Target 3 seeds for medium priority configs
        target_medium = 3
        k1_needed = max(0, target_medium - current["k1_seeds"])
        k10_needed = max(0, target_medium - current["k10_seeds"])
        
        # Add K=1 experiments
        for seed in range(1, 1 + k1_needed):  # Start from seed 1
            extension_experiments.append({
                "name": f"k1_{config['config'].lower()}_s{seed}",
                "experiment": "concept",
                "model": "mlp",
                "features": config["features"],
                "depth": config["depth"],
                "adaptation_steps": 1,
                "epochs": 150,  # Slightly fewer epochs for efficiency
                "seed": seed,
                "timeout": 7200,
                "priority": "HIGH",
                "batch": "extended_grid",
                "justification": config["justification"]
            })
        
        # Add K=10 experiments
        for seed in range(1, 1 + k10_needed):
            extension_experiments.append({
                "name": f"k10_{config['config'].lower()}_s{seed}",
                "experiment": "concept",
                "model": "mlp",
                "features": config["features"],
                "depth": config["depth"],
                "adaptation_steps": 10,
                "epochs": 150,
                "seed": seed,
                "timeout": 10800,
                "priority": "HIGH",
                "batch": "extended_grid",
                "justification": config["justification"]
            })
    
    # BATCH 3: MEDIUM - Architectural comparisons
    arch_experiments = [
        # CNN architecture on key configurations
        {"features": 8, "depth": 3, "model": "cnn", "justification": "CNN vs MLP comparison"},
        {"features": 16, "depth": 3, "model": "cnn", "justification": "CNN scaling"},
        # LSTM/Transformer if supported
        {"features": 8, "depth": 3, "model": "lstm", "justification": "Sequential processing"},
        {"features": 8, "depth": 3, "model": "transformer", "justification": "Attention mechanism"}
    ]
    
    for config in arch_experiments:
        for k_steps in [1, 10]:
            for seed in range(2):  # 2 seeds for architectural comparisons
                extension_experiments.append({
                    "name": f"k{k_steps}_f{config['features']}d{config['depth']}_{config['model']}_s{seed}",
                    "experiment": "concept",
                    "model": config["model"],
                    "features": config["features"],
                    "depth": config["depth"],
                    "adaptation_steps": k_steps,
                    "epochs": 100,  # Shorter for architectural comparisons
                    "seed": seed,
                    "timeout": 7200 if k_steps == 1 else 10800,
                    "priority": "MEDIUM",
                    "batch": "architecture_comparison",
                    "justification": config["justification"]
                })
    
    # BATCH 4: LOW - Complete grid and extended analysis
    low_configs = [
        {"features": 8, "depth": 7, "config": "F8_D7", "justification": "Max depth simple features"},
        {"features": 16, "depth": 7, "config": "F16_D7", "justification": "Max depth medium features"},
        {"features": 32, "depth": 7, "config": "F32_D7", "justification": "Max complexity"},
        # Alternative datasets
        {"features": 8, "depth": 3, "experiment": "mod", "justification": "Domain transfer"},
        {"features": 8, "depth": 3, "experiment": "omniglot", "model": "cnn", "justification": "Visual domain"}
    ]
    
    for config in low_configs:
        if "experiment" in config and config["experiment"] != "concept":
            # Alternative dataset experiments
            for k_steps in [1, 10]:
                for seed in range(2):
                    extension_experiments.append({
                        "name": f"k{k_steps}_{config['experiment']}_f{config['features']}d{config['depth']}_s{seed}",
                        "experiment": config["experiment"],
                        "model": config.get("model", "mlp"),
                        "features": config["features"],
                        "depth": config["depth"],
                        "adaptation_steps": k_steps,
                        "epochs": 100,
                        "seed": seed,
                        "timeout": 7200 if k_steps == 1 else 10800,
                        "priority": "LOW",
                        "batch": "alternative_domains",
                        "justification": config["justification"]
                    })
        else:
            # High depth concept experiments
            current = current_status[config["config"]]
            # Target 2 seeds for low priority
            target_low = 2
            k1_needed = max(0, target_low - current["k1_seeds"])
            k10_needed = max(0, target_low - current["k10_seeds"])
            
            for seed in range(1, 1 + k1_needed):
                extension_experiments.append({
                    "name": f"k1_{config['config'].lower()}_s{seed}",
                    "experiment": "concept",
                    "model": "mlp",
                    "features": config["features"],
                    "depth": config["depth"],
                    "adaptation_steps": 1,
                    "epochs": 100,
                    "seed": seed,
                    "timeout": 9600,  # More time for complex concepts
                    "priority": "LOW",
                    "batch": "complete_grid",
                    "justification": config["justification"]
                })
            
            for seed in range(1, 1 + k10_needed):
                extension_experiments.append({
                    "name": f"k10_{config['config'].lower()}_s{seed}",
                    "experiment": "concept",
                    "model": "mlp",
                    "features": config["features"],
                    "depth": config["depth"],
                    "adaptation_steps": 10,
                    "epochs": 100,
                    "seed": seed,
                    "timeout": 14400,  # 4 hours for complex K=10
                    "priority": "LOW",
                    "batch": "complete_grid",
                    "justification": config["justification"]
                })
    
    return extension_experiments

def analyze_extension_requirements():
    """Analyze computational requirements for extension experiments."""
    
    experiments = create_extension_experiments()
    
    # Group by priority
    by_priority = {}
    for exp in experiments:
        priority = exp["priority"]
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(exp)
    
    analysis = {
        "total_experiments": len(experiments),
        "by_priority": {},
        "time_estimates": {},
        "strategic_recommendations": {}
    }
    
    for priority, exps in by_priority.items():
        total_time = sum(exp["timeout"] for exp in exps) / 3600  # Convert to hours
        analysis["by_priority"][priority] = {
            "count": len(exps),
            "total_time_hours": total_time,
            "configurations": list(set(f"F{exp['features']}_D{exp['depth']}" for exp in exps if 'features' in exp))
        }
    
    # Strategic recommendations
    analysis["strategic_recommendations"] = {
        "camera_ready_minimum": {
            "priority": "URGENT",
            "experiments": len(by_priority.get("URGENT", [])),
            "time_hours": analysis["by_priority"].get("URGENT", {}).get("total_time_hours", 0),
            "justification": "Essential for robust F16_D3 and F32_D3 comparisons"
        },
        "comprehensive_coverage": {
            "priority": "URGENT + HIGH",
            "experiments": len(by_priority.get("URGENT", [])) + len(by_priority.get("HIGH", [])),
            "time_hours": (analysis["by_priority"].get("URGENT", {}).get("total_time_hours", 0) + 
                          analysis["by_priority"].get("HIGH", {}).get("total_time_hours", 0)),
            "justification": "Complete core grid with architectural depth scaling"
        },
        "full_analysis": {
            "priority": "ALL",
            "experiments": len(experiments),
            "time_hours": sum(grp.get("total_time_hours", 0) for grp in analysis["by_priority"].values()),
            "justification": "Complete experimental coverage including architectures and domains"
        }
    }
    
    return analysis, experiments

def create_extension_slurm_scripts():
    """Create SLURM scripts for each priority level."""
    
    analysis, experiments = analyze_extension_requirements()
    
    # Group experiments by priority
    by_priority = {}
    for exp in experiments:
        priority = exp["priority"]
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(exp)
    
    slurm_scripts = {}
    
    for priority, exps in by_priority.items():
        script_name = f"run_extension_{priority.lower()}.slurm"
        
        # Estimate time needed (add 20% buffer)
        total_time_seconds = sum(exp["timeout"] for exp in exps) * 1.2
        time_hours = int(total_time_seconds / 3600) + 1
        
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=extension_{priority.lower()}
#SBATCH --output=extension_{priority.lower()}_%j.out
#SBATCH --error=extension_{priority.lower()}_%j.err
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

echo "EXTENSION EXPERIMENTS - {priority} PRIORITY"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Total experiments: {len(exps)}"
echo "Estimated time: {time_hours} hours"

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

# Run experiments
"""
        
        for i, exp in enumerate(exps):
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
echo "🚀 EXPERIMENT {i+1}/{len(exps)}: {exp['name']}"
echo "   Priority: {priority}, Batch: {exp['batch']}"
echo "   Justification: {exp['justification']}"
timeout {exp['timeout']} python main.py {' '.join(cmd_args)}
echo "   Completed experiment {i+1}/{len(exps)} at $(date)"
echo "   ----------------------------------------"
"""
        
        slurm_content += f"""
echo "============================================"
echo "{priority} PRIORITY EXPERIMENTS COMPLETED"
echo "Job finished at: $(date)"
echo "Total experiments run: {len(exps)}"
echo "============================================"
"""
        
        slurm_scripts[script_name] = slurm_content
    
    return slurm_scripts

def main():
    """Generate extension experiment plan and SLURM scripts."""
    
    print("🧪 EXTENSION EXPERIMENTS PLAN FOR FEATURES×DEPTH GRID")
    print("=" * 80)
    
    # Generate experiments and analysis
    analysis, experiments = analyze_extension_requirements()
    
    print(f"📊 EXTENSION REQUIREMENTS ANALYSIS")
    print(f"Total additional experiments needed: {analysis['total_experiments']}")
    print()
    
    for priority, data in analysis["by_priority"].items():
        print(f"🎯 {priority} PRIORITY:")
        print(f"   Experiments: {data['count']}")
        print(f"   Compute time: {data['total_time_hours']:.1f} hours")
        print(f"   Configurations: {', '.join(data['configurations'])}")
        print()
    
    print("📋 STRATEGIC RECOMMENDATIONS:")
    for strategy, rec in analysis["strategic_recommendations"].items():
        print(f"\n🎯 {strategy.upper()}:")
        print(f"   Focus: {rec['priority']}")
        print(f"   Experiments: {rec['experiments']}")
        print(f"   Time: {rec['time_hours']:.1f} hours")
        print(f"   Justification: {rec['justification']}")
    
    # Create SLURM scripts
    print("\n🚀 GENERATING SLURM SCRIPTS...")
    slurm_scripts = create_extension_slurm_scripts()
    
    # Save everything
    import os
    os.makedirs('extension_experiments', exist_ok=True)
    
    # Save experiment plan
    with open('extension_experiments/experiments_plan.json', 'w') as f:
        json.dump(experiments, f, indent=2)
    
    # Save analysis
    with open('extension_experiments/requirements_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save SLURM scripts
    for script_name, content in slurm_scripts.items():
        with open(f'extension_experiments/{script_name}', 'w') as f:
            f.write(content)
    
    print(f"\n✅ Extension plan saved to extension_experiments/")
    print(f"📁 Generated files:")
    print(f"   - experiments_plan.json: Complete experiment specifications")
    print(f"   - requirements_analysis.json: Computational requirements")
    for script_name in slurm_scripts.keys():
        print(f"   - {script_name}: SLURM script for {script_name.split('_')[2].upper()} priority")
    
    print(f"\n🎯 RECOMMENDED EXECUTION ORDER:")
    print(f"1. URGENT: Essential for camera-ready (F16_D3, F32_D3 robustness)")
    print(f"2. HIGH: Comprehensive grid coverage (depth scaling)")  
    print(f"3. MEDIUM: Architectural comparisons (CNN, LSTM, Transformer)")
    print(f"4. LOW: Complete analysis (max depth, alternative domains)")
    
    return analysis, experiments

if __name__ == "__main__":
    main() 