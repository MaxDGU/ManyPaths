#!/bin/bash

#SBATCH --job-name=mp_concept_feat_ord
#SBATCH --output=logs/mp_concept_%A_%a.out # Standard output and error log
#SBATCH --error=logs/mp_concept_%A_%a.err
#SBATCH --mail-type=FAIL,END          # Mail events (FAIL, END)
#SBATCH --mail-user=mg7411@princeton.edu # Your email address
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=4          # Number of CPU cores per task (adjust if needed)
#SBATCH --mem=16G                  # Job memory request (e.g., 16G, adjust as needed)
#SBATCH --time=03:00:00            # Time limit hrs:min:sec (adjust based on expected runtime)
#SBATCH --gres=gpu:1               # Request 1 GPU (specify type if necessary, e.g., gpu:a100:1)
#SBATCH --array=0                # MODIFIED: Job array for a single test task (task ID 0)
#SBATCH --partition=pli
#SBATCH --account=nam

# --- Environment Setup ---
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p state_dicts # Ensure state_dicts directory exists for saving models

# Load necessary modules for DELLA (replace with actual DELLA module commands)
# These are placeholders - update them for your DELLA environment
echo "Loading modules..."
module purge # Start with a clean environment
module load anaconda3/2023.3 # Or your specific anaconda/miniconda module
module load cudatoolkit/11.8 # Or your specific CUDA toolkit version
conda activate tensorflow # Activate your conda environment, e.g., tensorflow


# Project directory on the cluster
PROJECT_DIR="/scratch/gpfs/mg7411/ManyPaths" # Ensure this matches where you rsync'd
cd $PROJECT_DIR
echo "Changed to project directory: $(pwd)"

# --- Experiment Parameters ---
FEATURES_LIST=(8 16 32 64 128)
ORDERS_LIST=(0 1) # 0 for 2nd order (MAML-like), 1 for 1st order MetaSGD

# Calculate parameters for the current task
NUM_PARAM_COMBOS_PER_FEATURE=${#ORDERS_LIST[@]}

FEATURE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_PARAM_COMBOS_PER_FEATURE))
ORDER_IDX=$((SLURM_ARRAY_TASK_ID % NUM_PARAM_COMBOS_PER_FEATURE))

CURRENT_FEATURES=${FEATURES_LIST[$FEATURE_IDX]}
CURRENT_ORDER_VAL=${ORDERS_LIST[$ORDER_IDX]}

FIRST_ORDER_FLAG=""
ORDER_STR="2ndOrd"
if [ "$CURRENT_ORDER_VAL" -eq 1 ]; then
    FIRST_ORDER_FLAG="--first-order"
    ORDER_STR="1stOrd"
fi

# Shared parameters
ADAPTATION_STEPS=1
SEED_VAL=0 # Using a single seed for this batch
HYPER_INDEX_VAL=14 # Use the hyperparameter index found from local runs
EPOCHS=50000 # Max epochs, rely on early stopping
TASKS_PER_META_BATCH=4
OUTER_LR=1e-3
INNER_LR=1e-2 # Used by MetaSGD

echo "--- Experiment Configuration ---"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Features: ${CURRENT_FEATURES}"
echo "Order: ${ORDER_STR}"
echo "Adaptation Steps: ${ADAPTATION_STEPS}"
echo "Seed: ${SEED_VAL}"
echo "Hyperparameter Index: ${HYPER_INDEX_VAL}"
echo "Epochs: ${EPOCHS}"
echo "---------------------------------"

# --- Run the Python script ---
echo "Starting Python script..."
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

EXIT_CODE=$?
echo "Python script finished with exit code ${EXIT_CODE}"
exit $EXIT_CODE 