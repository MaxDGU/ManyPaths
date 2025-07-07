#!/bin/bash
#SBATCH --job-name=grad_align_experiments
#SBATCH --output=gradient_alignment_experiments_%j.out
#SBATCH --error=gradient_alignment_experiments_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load necessary modules
module purge
module load anaconda3/2023.9
module load cudatoolkit/11.8
module load cudnn/cuda-11.x/8.6.0

# Activate conda environment
echo "Activating conda environment..."
conda activate /scratch/gpfs/mg7411/envs/pytorch_env

# Set up environment
echo "Setting up environment..."
export PYTHONPATH=/scratch/gpfs/mg7411/ManyPaths:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Change to working directory
cd /scratch/gpfs/mg7411/ManyPaths

# Print system information
echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

# First run the test to ensure everything works
echo "Running gradient alignment test..."
python della_test_gradient_alignment.py

# Check if test passed
if [ $? -eq 0 ]; then
    echo "Test passed! Running full experiments..."
    python della_full_gradient_experiments.py
else
    echo "Test failed! Skipping full experiments."
    exit 1
fi

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================" 