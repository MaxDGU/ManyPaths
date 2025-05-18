#!/bin/bash

#SBATCH --job-name=mp_concept_f8_paired
#SBATCH --output=logs/mp_concept_f8_paired_%A_%a.out # Standard output and error log
#SBATCH --error=logs/mp_concept_f8_paired_%A_%a.err
#SBATCH --mail-type=FAIL,END          # Mail events (FAIL, END)
#SBATCH --mail-user=mg7411@princeton.edu # Your email address
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=16G                  # Job memory request
#SBATCH --time=03:00:00            # Time limit hrs:min:sec (Adjust as needed for 50k epochs)
#SBATCH --gres=gpu:1               # Request 1 GPU 
#SBATCH --array=0-1                # Job array for two experiments (1st order, 2nd order)
#SBATCH --partition=pli
#SBATCH --account=nam

# --- Environment Setup ---
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist (for SLURM output)
# main.py will create 'results/' for trajectories and 'saved_models/' for models.
mkdir -p logs

# Load necessary modules for DELLA
echo "Loading modules..."
module purge # Start with a clean environment
module load anaconda3/2023.09 # Or your specific anaconda/miniconda module
module load cudatoolkit/11.8 # Or your specific CUDA toolkit version

# Activate Conda environment
CONDA_ENV_NAME="tensorflow" # As per your previous script
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment: ${CONDA_ENV_NAME}"
    exit 1
fi
echo "Conda environment activated: ${CONDA_ENV_NAME}"

# Project directory on the cluster
PROJECT_DIR="/scratch/gpfs/mg7411/ManyPaths" # From your script
cd $PROJECT_DIR
if [ $? -ne 0 ]; then
    echo "Failed to change to project directory: ${PROJECT_DIR}"
    exit 1
fi
echo "Changed to project directory: $(pwd)"

# --- Experiment Parameters ---
# For this paired experiment, features are fixed.
CURRENT_FEATURES=8
ORDERS_LIST=(0 1) # 0 for 2nd order (MAML-like), 1 for 1st order MetaSGD

# Determine order based on SLURM_ARRAY_TASK_ID
CURRENT_ORDER_VAL=${ORDERS_LIST[$SLURM_ARRAY_TASK_ID]}

FIRST_ORDER_FLAG=""
ORDER_STR="2ndOrd"
if [ "$CURRENT_ORDER_VAL" -eq 1 ]; then
    FIRST_ORDER_FLAG="--first-order"
    ORDER_STR="1stOrd"
fi

# Shared parameters for main.py
ADAPTATION_STEPS=1
SEED_VAL=0
HYPER_INDEX_VAL=14
EPOCHS=50000 # As per your run_manypaths_concept.slurm
TASKS_PER_META_BATCH=4
OUTER_LR=1e-3
INNER_LR=1e-2

echo "--- Experiment Configuration --- "
echo "Task ID (SLURM): ${SLURM_ARRAY_TASK_ID}"
echo "Features: ${CURRENT_FEATURES}"
echo "Order: ${ORDER_STR} (Flag: ${FIRST_ORDER_FLAG})"
echo "Epochs: ${EPOCHS}"
echo "Seed: ${SEED_VAL}"
echo "---------------------------------"

# --- Run the Python script --- 
echo "Starting Python script for ${ORDER_STR}..."

python main.py \
    --experiment concept \
    --m mlp \
    --data-type bits \
    --num-concept-features ${CURRENT_FEATURES} \
    --adaptation-steps ${ADAPTATION_STEPS} \
    ${FIRST_ORDER_FLAG} \
    --epochs ${EPOCHS} \
    --tasks_per_meta_batch ${TASKS_PER_META_BATCH} \
    --outer_lr ${OUTER_LR} \
    --inner_lr ${INNER_LR} \
    --save \
    --seed ${SEED_VAL} \
    --no-hyper-search \
    --hyper-index ${HYPER_INDEX_VAL}
    # --plot flag is removed for cluster runs

EXIT_CODE=$?
echo "Python script for ${ORDER_STR} finished with exit code ${EXIT_CODE}"

# Deactivate Conda (optional, good practice)
# conda deactivate

exit $EXIT_CODE 