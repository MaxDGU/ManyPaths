#!/bin/bash
#SBATCH --job-name=grad_align_experiments
#SBATCH --output=gradient_alignment_experiments_%j.out
#SBATCH --error=gradient_alignment_experiments_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mg7411@princeton.edu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam

# Load necessary modules (matching working baseline configuration)
echo "SLURM SCRIPT: Starting gradient alignment experiments"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
mkdir -p logs
module purge
module load anaconda3/2023.3  # Using the version confirmed to be available
if [ $? -ne 0 ]; then echo "ERROR: Failed to load anaconda3/2023.3 "; exit 1; fi
echo "Loaded anaconda3/2023.3 "
module load cudatoolkit/11.8
if [ $? -ne 0 ]; then echo "ERROR: Failed to load cudatoolkit/11.8"; exit 1; fi
echo "Loaded cudatoolkit/11.8"

# Activate conda environment (matching baseline configuration)
CONDA_ENV_NAME="tensorflow"
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then echo "ERROR: Failed to activate conda env ${CONDA_ENV_NAME}"; exit 1; fi
echo "Activated conda env: ${CONDA_ENV_NAME} (Python: $(which python))"

# Change to working directory
PROJECT_DIR="/scratch/gpfs/mg7411/ManyPaths"
cd $PROJECT_DIR || exit 1

# Print system information
echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
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
