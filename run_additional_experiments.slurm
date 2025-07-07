#!/bin/bash
#SBATCH --job-name=additional_experiments_camera_ready
#SBATCH --output=additional_experiments_%j.out
#SBATCH --error=additional_experiments_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

# Print job info
echo "SLURM SCRIPT: Starting additional experiments for camera-ready submission"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"

# Load modules
module load anaconda3/2023.3
module load cudatoolkit/11.8

# Activate conda environment
conda activate tensorflow
echo "Loaded anaconda3/2023.3 "
echo "Loaded cudatoolkit/11.8"
echo "Activated conda env: tensorflow (Python: $(which python))"

# Set up environment
export CUDA_VISIBLE_DEVICES=0

# Job information
echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

# Run the additional experiments
echo "🚀 STARTING ADDITIONAL EXPERIMENTS FOR CAMERA-READY"
echo "============================================"

# Check if we have the script
if [ ! -f "della_additional_experiments.py" ]; then
    echo "❌ Error: della_additional_experiments.py not found"
    exit 1
fi

# Run experiments with timeout and error handling
timeout 86400 python della_additional_experiments.py

# Check exit status
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "✅ Additional experiments completed successfully"
elif [ $exit_code -eq 124 ]; then
    echo "⏰ Additional experiments timed out after 24 hours"
    exit 1
else
    echo "❌ Additional experiments failed with exit code $exit_code"
    exit 1
fi

# Final summary
echo "============================================"
echo "🎯 ADDITIONAL EXPERIMENTS SUMMARY"
echo "============================================"
echo "Job completed at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Count results if possible
if [ -f "additional_experiments_final_results.json" ]; then
    echo "📊 Results summary:"
    python -c "
import json
try:
    with open('additional_experiments_final_results.json', 'r') as f:
        results = json.load(f)
    completed = len(results.get('completed', []))
    failed = len(results.get('failed', []))
    total = results.get('total_experiments', 0)
    print(f'✅ Completed: {completed}/{total}')
    print(f'❌ Failed: {failed}/{total}')
    
    # Priority breakdown
    for priority in ['URGENT', 'HIGH', 'MEDIUM', 'LOW']:
        completed_priority = len([r for r in results.get('completed', []) if r.get('priority') == priority])
        failed_priority = len([r for r in results.get('failed', []) if r.get('priority') == priority])
        total_priority = completed_priority + failed_priority
        if total_priority > 0:
            print(f'  {priority}: {completed_priority}/{total_priority}')
except Exception as e:
    print(f'Error reading results: {e}')
"
else
    echo "⚠️  No results file found"
fi

echo "============================================"
echo "📁 Files created:"
ls -la *additional_experiments* 2>/dev/null || echo "No additional experiment files found"
echo "============================================"

exit 0 