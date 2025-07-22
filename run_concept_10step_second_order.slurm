#!/bin/bash

#SBATCH --job-name=mp_concept_k10_2nd
#SBATCH --output=logs/mp_concept_k10_2nd_%A_%a.out
#SBATCH --error=logs/mp_concept_k10_2nd_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mg7411@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G         # K=10 with second-order might need more memory
#SBATCH --time=12:00:00   # Increased time for second-order K=10 (more compute intensive)
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --array=0-8      # 3 Features * 3 Depths * 1 Order (2nd only) = 9 tasks

# --- Parameter Lists ---
FEATURES_LIST=(8 16 32)
DEPTHS_LIST=(3 5 7)
ORDERS_LIST=(0) # Force 2nd Order only for this K=10 sweep (0 = second-order)

NUM_FEATURES=${#FEATURES_LIST[@]}
NUM_DEPTHS=${#DEPTHS_LIST[@]}
NUM_ORDERS=${#ORDERS_LIST[@]} # Will be 1

# --- Environment Setup ---
echo "SLURM SCRIPT: Starting job setup"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

mkdir -p logs
echo "SLURM SCRIPT: Created logs directory if it didn't exist."

echo "SLURM SCRIPT: Loading modules..."
module purge
echo "SLURM SCRIPT: Purged modules."
module load Anaconda3/2023.9 
echo "SLURM SCRIPT: Loaded Anaconda module (Anaconda3/2023.9)."
module load cudatoolkit/11.8
echo "SLURM SCRIPT: Loaded CUDA toolkit."

CONDA_ENV_NAME="tensorflow"
echo "SLURM SCRIPT: Attempting to activate conda environment: ${CONDA_ENV_NAME}"

__conda_setup="$('/usr/licensed/anaconda3/2023.9/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/usr/licensed/anaconda3/2023.9/etc/profile.d/conda.sh" ]; then
        . "/usr/licensed/anaconda3/2023.9/etc/profile.d/conda.sh"
    else
        echo "SLURM SCRIPT ERROR: conda.sh not found! Conda initialization failed."
        exit 1
    fi
fi
unset __conda_setup

conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "SLURM SCRIPT ERROR: Failed to activate conda environment: ${CONDA_ENV_NAME}"
    exit 1
fi
echo "SLURM SCRIPT: Conda environment activated: ${CONDA_ENV_NAME}"

PROJECT_DIR="/scratch/gpfs/mg7411/ManyPaths"
cd $PROJECT_DIR
if [ $? -ne 0 ]; then echo "SLURM SCRIPT ERROR: Failed to cd to ${PROJECT_DIR}"; exit 1; fi
echo "SLURM SCRIPT: Changed to project directory: $(pwd)"

# --- Determine Parameters ---
# Since NUM_ORDERS is 1, ORDER_IDX will always be 0, and CURRENT_ORDER_VAL will be ORDERS_LIST[0] which is 0.
ORDER_IDX=$((SLURM_ARRAY_TASK_ID % NUM_ORDERS)) # This will be 0
TEMP_IDX=$((SLURM_ARRAY_TASK_ID / NUM_ORDERS))  # This effectively becomes SLURM_ARRAY_TASK_ID
DEPTH_IDX=$((TEMP_IDX % NUM_DEPTHS))
FEATURE_IDX=$((TEMP_IDX / NUM_DEPTHS))

CURRENT_FEATURES=${FEATURES_LIST[$FEATURE_IDX]}
CURRENT_DEPTH=${DEPTHS_LIST[$DEPTH_IDX]}
CURRENT_ORDER_VAL=${ORDERS_LIST[$ORDER_IDX]} # This will be 0 (2nd order)

# No first-order flag for second-order experiments
FIRST_ORDER_FLAG_STR=""
ORDER_STR="2ndOrd"

# Shared parameters
ADAPTATION_STEPS=10 # <--- SET TO 10
SEED_VAL=0
HYPER_INDEX_VAL=14 
TASKS_PER_META_BATCH=4
OUTER_LR=1e-3
INNER_LR=1e-2
EPOCHS_VAL=1000 
CHECKPOINT_INTERVAL_VAL=40
SAVE_FLAG_STR="--save"

echo "--- Experiment Configuration for Task ID ${SLURM_ARRAY_TASK_ID} (2nd Order K=10 RUN) --- "
echo "Features: ${CURRENT_FEATURES}"
echo "PCFG Max Depth: ${CURRENT_DEPTH}"
echo "Order: ${ORDER_STR} (Second-Order - no flag needed)"
echo "Adaptation Steps (K): ${ADAPTATION_STEPS}" # Echo the K value
echo "Epochs: ${EPOCHS_VAL}"
echo "Checkpoint Interval: ${CHECKPOINT_INTERVAL_VAL}"
echo "Save Flag: ${SAVE_FLAG_STR}"
echo "Seed: ${SEED_VAL}"
echo "---------------------------------"

echo "SLURM SCRIPT: Starting Python script (main.py) for 2nd Order K=${ADAPTATION_STEPS}..."
python main.py \
    --experiment concept \
    --m mlp \
    --data-type bits \
    --num-concept-features ${CURRENT_FEATURES} \
    --pcfg-max-depth ${CURRENT_DEPTH} \
    --adaptation-steps ${ADAPTATION_STEPS} \
    --epochs ${EPOCHS_VAL} \
    --checkpoint-interval ${CHECKPOINT_INTERVAL_VAL} \
    --tasks_per_meta_batch ${TASKS_PER_META_BATCH} \
    --outer_lr ${OUTER_LR} \
    --inner_lr ${INNER_LR} \
    ${SAVE_FLAG_STR} \
    --seed ${SEED_VAL} \
    --no_hyper_search \
    --hyper-index ${HYPER_INDEX_VAL}

EXIT_CODE=$?
echo "SLURM SCRIPT: Python script finished with exit code ${EXIT_CODE}"
echo "SLURM SCRIPT: Job completed."

exit $EXIT_CODE 