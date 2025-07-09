#!/usr/bin/env python3
"""
Realistic MLP Timing Calibration for Della

Based on actual F8_D3 performance:
- K=1 experiments: ~12-15 hours for 200 epochs  
- K=10 experiments: ~18-24 hours for 200 epochs
- Scale with complexity (features × depth)
"""

import json
import os

def calibrate_realistic_timing():
    """Calculate realistic timing based on actual F8_D3 performance."""
    
    # Observed F8_D3 performance (from della analysis)
    f8d3_k1_observed = 15  # hours for 200 epochs
    f8d3_k10_observed = 24  # hours for 200 epochs
    
    # Base complexity units
    f8d3_complexity = 8 * 3  # features × depth = 24 units
    
    def estimate_time(features, depth, k_steps, epochs=200):
        """Estimate time based on complexity scaling."""
        complexity = features * depth
        complexity_factor = complexity / f8d3_complexity
        
        if k_steps == 1:
            base_time = f8d3_k1_observed
        else:  # k_steps == 10
            base_time = f8d3_k10_observed
        
        # Scale by complexity and epochs
        estimated_time = base_time * complexity_factor * (epochs / 200)
        
        # Add safety buffer (50% for complex experiments)
        safety_buffer = 1.5 if complexity > f8d3_complexity else 1.3
        
        return estimated_time * safety_buffer
    
    return estimate_time

def create_realistic_slurm_scripts():
    """Create SLURM scripts with realistic timing."""
    
    # Load experiments
    with open('mlp_grid_completion/mlp_experiments.json', 'r') as f:
        experiments = json.load(f)
    
    timing_func = calibrate_realistic_timing()
    
    # Update experiments with realistic timing
    for exp in experiments:
        realistic_time = timing_func(
            exp['features'], 
            exp['depth'], 
            exp['adaptation_steps'], 
            exp['epochs']
        )
        exp['realistic_timeout'] = int(realistic_time * 3600)  # Convert to seconds
        exp['realistic_hours'] = realistic_time
    
    # Group by priority
    by_priority = {}
    for exp in experiments:
        priority = exp["priority"]
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(exp)
    
    priority_info = {
        "P1_CORE": {
            "name": "core_complexity",
            "description": "Essential complexity scaling evidence",
            "urgent": True
        },
        "P2_DEPTH": {
            "name": "depth_scaling", 
            "description": "Logical complexity (depth) scaling",
            "urgent": False
        },
        "P3_MAX": {
            "name": "max_complexity",
            "description": "Maximum complexity validation", 
            "urgent": False
        }
    }
    
    slurm_scripts = {}
    
    for priority, exps in by_priority.items():
        info = priority_info[priority]
        script_name = f"run_realistic_mlp_{info['name']}.slurm"
        
        # Calculate total time with buffer
        total_realistic_time = sum(exp["realistic_hours"] for exp in exps)
        # Add 25% scheduling buffer
        slurm_time_hours = int(total_realistic_time * 1.25) + 2
        
        # Cap at 48 hours (della limit)
        slurm_time_hours = min(slurm_time_hours, 48)
        
        urgency_marker = "🚨 CAMERA-READY URGENT" if info['urgent'] else "📊 ANALYSIS EXTENSION"
        
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=mlp_{info['name']}
#SBATCH --output=mlp_{info['name']}_realistic_%j.out
#SBATCH --error=mlp_{info['name']}_realistic_%j.err
#SBATCH --time={slurm_time_hours:02d}:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "{urgency_marker}"
echo "MLP GRID COMPLETION - {priority} ({info['name'].upper()})"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Priority: {info['description']}"
echo "Experiments: {len(exps)}, Allocated time: {slurm_time_hours}h"
echo "Realistic estimate: {total_realistic_time:.1f}h (based on F8_D3 performance)"
echo "Configurations: {', '.join(set(exp['config'] for exp in exps))}"

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
echo "   Est. time: {exp['realistic_hours']:.1f}h (F8_D3 calibrated)"
echo "   Timeout: {exp['realistic_timeout']}s, Epochs: {exp['epochs']}"
echo "   Command: python main.py {' '.join(cmd_args)}"
echo "   Started at: $(date)"

timeout {exp['realistic_timeout']} python main.py {' '.join(cmd_args)}
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after {exp['realistic_hours']:.1f}h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="
"""
        
        slurm_content += f"""
echo "============================================"
echo "🎯 {info['name'].upper()} PRIORITY COMPLETED"
echo "Job finished at: $(date)"
echo "Total experiments: {len(exps)}"
echo "Realistic time used vs allocated: {total_realistic_time:.1f}h / {slurm_time_hours}h"
echo "Statistical coverage: {', '.join(set(exp['config'] for exp in exps))}"
echo "============================================"
"""
        
        slurm_scripts[script_name] = slurm_content
    
    return slurm_scripts, experiments

def create_execution_summary():
    """Create execution summary with realistic estimates."""
    
    scripts, experiments = create_realistic_slurm_scripts()
    
    timing_func = calibrate_realistic_timing()
    
    # Summary by priority
    summary = {
        "calibration_basis": "F8_D3 observed: K=1 ~15h, K=10 ~24h for 200 epochs",
        "scaling_method": "Linear with features×depth complexity",
        "safety_buffer": "30-50% depending on complexity",
        "priorities": {}
    }
    
    by_priority = {}
    for exp in experiments:
        priority = exp["priority"]
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(exp)
    
    for priority, exps in by_priority.items():
        total_time = sum(exp["realistic_hours"] for exp in exps)
        configs = list(set(exp["config"] for exp in exps))
        
        summary["priorities"][priority] = {
            "experiments": len(exps),
            "realistic_hours": round(total_time, 1),
            "slurm_allocation": min(int(total_time * 1.25) + 2, 48),
            "configurations": configs,
            "urgency": "CAMERA-READY CRITICAL" if priority == "P1_CORE" else "ANALYSIS ENHANCEMENT"
        }
    
    return summary, scripts

def main():
    """Generate realistic MLP timing and execution plan."""
    
    print("🕒 REALISTIC MLP TIMING CALIBRATION")
    print("=" * 80)
    print("📊 Calibration basis: F8_D3 observed performance")
    print("   K=1: ~15 hours for 200 epochs")
    print("   K=10: ~24 hours for 200 epochs")
    print("   Scaling: Linear with features×depth complexity")
    print("   Safety buffer: 30-50% for complex experiments")
    print()
    
    # Generate realistic timing
    summary, scripts = create_execution_summary()
    
    print("🎯 REALISTIC EXECUTION PLAN:")
    for priority, data in summary["priorities"].items():
        urgency = "🚨" if priority == "P1_CORE" else "📊"
        print(f"   {urgency} {priority}: {data['experiments']} exp, {data['realistic_hours']}h realistic, {data['slurm_allocation']}h allocated")
        print(f"      Configs: {', '.join(data['configurations'])}")
        print(f"      Status: {data['urgency']}")
        print()
    
    # Save everything
    os.makedirs('realistic_mlp_execution', exist_ok=True)
    
    # Save scripts
    for script_name, content in scripts.items():
        with open(f'realistic_mlp_execution/{script_name}', 'w') as f:
            f.write(content)
    
    # Save summary
    with open('realistic_mlp_execution/timing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Realistic execution plan saved to realistic_mlp_execution/")
    print("📁 Generated files:")
    for script_name in scripts.keys():
        print(f"   - {script_name}")
    print("   - timing_summary.json")
    
    print(f"\n🚀 READY TO QUEUE ON DELLA:")
    print(f"1. git pull origin master")
    print(f"2. sbatch realistic_mlp_execution/run_realistic_mlp_core_complexity.slurm  # START HERE")
    print(f"3. sbatch realistic_mlp_execution/run_realistic_mlp_depth_scaling.slurm   # AFTER P1")  
    print(f"4. sbatch realistic_mlp_execution/run_realistic_mlp_max_complexity.slurm  # FINAL")
    
    return summary, scripts

if __name__ == "__main__":
    main() 