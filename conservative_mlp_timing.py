#!/usr/bin/env python3
"""
Conservative MLP Timing Model

Based on actual observations and reasonable scaling:
- Complexity scaling is sublinear (larger models more efficient per unit)
- Conservative estimates based on actual F8_D3 performance
- Account for della scheduling realities
"""

import json
import os

def conservative_timing_model():
    """Conservative timing based on realistic scaling."""
    
    # Actual F8_D3 observations (from your della analysis)
    f8d3_k1_base = 12  # Conservative estimate for K=1
    f8d3_k10_base = 18  # Conservative estimate for K=10
    
    def estimate_time(features, depth, k_steps, epochs=200):
        """Conservative time estimate with sublinear scaling."""
        
        # Base complexity
        base_complexity = 8 * 3  # F8_D3 = 24
        current_complexity = features * depth
        
        # Sublinear complexity scaling (square root reduces growth)
        complexity_factor = (current_complexity / base_complexity) ** 0.7
        
        # Base time
        if k_steps == 1:
            base_time = f8d3_k1_base
        else:
            base_time = f8d3_k10_base
        
        # Epoch scaling
        epoch_factor = epochs / 200
        
        # Conservative estimate
        estimated_time = base_time * complexity_factor * epoch_factor
        
        # Add modest safety buffer
        safety_buffer = 1.4  # 40% buffer for all experiments
        
        return estimated_time * safety_buffer
    
    return estimate_time

def create_conservative_slurm_scripts():
    """Create SLURM scripts with conservative timing."""
    
    # Load experiments
    with open('mlp_grid_completion/mlp_experiments.json', 'r') as f:
        experiments = json.load(f)
    
    timing_func = conservative_timing_model()
    
    # Update with conservative timing
    for exp in experiments:
        conservative_time = timing_func(
            exp['features'], 
            exp['depth'], 
            exp['adaptation_steps'], 
            exp['epochs']
        )
        exp['conservative_timeout'] = int(conservative_time * 3600)
        exp['conservative_hours'] = conservative_time
    
    # Group by priority
    by_priority = {}
    for exp in experiments:
        priority = exp["priority"]
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(exp)
    
    priority_info = {
        "P1_CORE": {"name": "core", "urgent": True},
        "P2_DEPTH": {"name": "depth", "urgent": False},
        "P3_MAX": {"name": "max", "urgent": False}
    }
    
    slurm_scripts = {}
    
    for priority, exps in by_priority.items():
        info = priority_info[priority]
        script_name = f"run_mlp_{info['name']}_conservative.slurm"
        
        # Conservative time allocation
        total_time = sum(exp["conservative_hours"] for exp in exps)
        slurm_hours = min(int(total_time * 1.2) + 2, 48)  # 20% scheduling buffer, max 48h
        
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=mlp_{info['name']}_grid
#SBATCH --output=mlp_{info['name']}_grid_%j.out
#SBATCH --error=mlp_{info['name']}_grid_%j.err
#SBATCH --time={slurm_hours:02d}:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "MLP GRID COMPLETION - {priority}"
echo "Conservative timing model (sublinear scaling)"
echo "Host: $(hostname), Date: $(date)"
echo "Experiments: {len(exps)}, Allocated: {slurm_hours}h, Est: {total_time:.1f}h"
echo "Configs: {', '.join(set(exp.get('config', f'F{exp["features"]}_D{exp["depth"]}') for exp in exps))}"

module load anaconda3/2023.3
module load cudatoolkit/11.8
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname)"
echo "============================================"

"""
        
        for i, exp in enumerate(exps):
            config_name = exp.get('config', f"F{exp['features']}_D{exp['depth']}")
            
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
echo "Experiment {i+1}/{len(exps)}: {config_name}_K{exp['adaptation_steps']}_S{exp['seed']}"
echo "  Est: {exp['conservative_hours']:.1f}h, Timeout: {exp['conservative_timeout']}s"
echo "  Started: $(date)"

timeout {exp['conservative_timeout']} python main.py {' '.join(cmd_args)}
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after {exp['conservative_hours']:.1f}h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"
"""
        
        slurm_content += f"""
echo "============================================"
echo "{priority} COMPLETED at $(date)"
echo "Time allocation: {slurm_hours}h, Estimated: {total_time:.1f}h"
echo "============================================"
"""
        
        slurm_scripts[script_name] = slurm_content
    
    return slurm_scripts, experiments

def main():
    """Generate conservative timing plan."""
    
    print("🕒 CONSERVATIVE MLP TIMING MODEL")
    print("=" * 70)
    print("📊 Based on F8_D3 performance: K=1 ~12h, K=10 ~18h")
    print("📈 Sublinear scaling: complexity^0.7 (more efficient for larger models)")
    print("🛡️  40% safety buffer for all experiments")
    print()
    
    scripts, experiments = create_conservative_slurm_scripts()
    
    # Analysis by priority
    by_priority = {}
    for exp in experiments:
        priority = exp["priority"]
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(exp)
    
    print("🎯 CONSERVATIVE EXECUTION PLAN:")
    
    priority_names = {
        "P1_CORE": "🚨 CORE COMPLEXITY (camera-ready critical)",
        "P2_DEPTH": "📊 DEPTH SCALING (comprehensive analysis)", 
        "P3_MAX": "📈 MAX COMPLEXITY (complete grid)"
    }
    
    for priority, exps in by_priority.items():
        total_time = sum(exp["conservative_hours"] for exp in exps)
        configs = set(exp.get('config', f"F{exp['features']}_D{exp['depth']}") for exp in exps)
        slurm_allocation = min(int(total_time * 1.2) + 2, 48)
        
        print(f"   {priority_names[priority]}")
        print(f"   📦 {len(exps)} experiments, {total_time:.1f}h estimated, {slurm_allocation}h allocated")
        print(f"   🎯 Configs: {', '.join(configs)}")
        print()
    
    # Save files
    os.makedirs('conservative_mlp_execution', exist_ok=True)
    
    for script_name, content in scripts.items():
        with open(f'conservative_mlp_execution/{script_name}', 'w') as f:
            f.write(content)
    
    # Summary
    with open('conservative_mlp_execution/conservative_summary.json', 'w') as f:
        summary = {
            "timing_model": "Sublinear scaling (complexity^0.7) with 40% safety buffer",
            "base_times": "F8_D3: K=1 ~12h, K=10 ~18h for 200 epochs",
            "priorities": {}
        }
        
        for priority, exps in by_priority.items():
            total_time = sum(exp["conservative_hours"] for exp in exps)
            summary["priorities"][priority] = {
                "experiments": len(exps),
                "estimated_hours": round(total_time, 1),
                "slurm_allocation": min(int(total_time * 1.2) + 2, 48)
            }
        
        json.dump(summary, f, indent=2)
    
    print("✅ Conservative execution plan saved!")
    print("📁 Files in conservative_mlp_execution/:")
    for script_name in scripts.keys():
        print(f"   - {script_name}")
    print("   - conservative_summary.json")
    
    print(f"\n🚀 DELLA EXECUTION ORDER:")
    print(f"1. sbatch conservative_mlp_execution/run_mlp_core_conservative.slurm")
    print(f"2. sbatch conservative_mlp_execution/run_mlp_depth_conservative.slurm") 
    print(f"3. sbatch conservative_mlp_execution/run_mlp_max_conservative.slurm")
    
    return scripts, experiments

if __name__ == "__main__":
    main() 