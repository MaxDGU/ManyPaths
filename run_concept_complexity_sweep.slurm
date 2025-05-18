#!/bin/bash

#SBATCH --job-name=mp_concept_sweep
#SBATCH --output=logs/mp_concept_sweep_%A_%a.out # Standard output and error log
#SBATCH --error=logs/mp_concept_sweep_%A_%a.err
#SBATCH --mail-type=FAIL,END          # Mail events (FAIL, END)
#SBATCH --mail-user=mg7411@princeton.edu # Your email address
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=16G                  # Job memory request
#SBATCH --time=06:00:00            # Time limit hrs:min:sec (Adjust if 20k epochs is faster)
#SBATCH --gres=gpu:1               # Request 1 GPU 
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --array=0-17                # MODIFIED: Hardcoded for 3*3*2 = 18 jobs (indices 0-17)

# --- Parameter Lists for Sweep (still used for indexing logic below) ---
FEATURES_LIST=(8 16 32)
DEPTHS_LIST=(3 5 7)
ORDERS_LIST=(0 1)

NUM_FEATURES=${#FEATURES_LIST[@]}
NUM_DEPTHS=${#DEPTHS_LIST[@]}
NUM_ORDERS=${#ORDERS_LIST[@]}
TOTAL_JOBS=$((NUM_FEATURES * NUM_DEPTHS * NUM_ORDERS)) # Should be 18, used for echo

# --- Environment Setup ---
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}" # Should now be populated
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"  # Should now be populated
echo "Total expected jobs based on lists: $TOTAL_JOBS"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

mkdir -p logs

echo "Loading modules..."
module purge 
module load Anaconda3/2023.3 
module load cudatoolkit/11.8 

CONDA_ENV_NAME="tensorflow" 
echo "Attempting to activate conda environment: ${CONDA_ENV_NAME}"

__conda_setup="$('/usr/licensed/anaconda3/2023.9/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" 
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/usr/licensed/anaconda3/2023.9/etc/profile.d/conda.sh" ]; then # Added specific Della path as fallback for sourcing
        . "/usr/licensed/anaconda3/2023.9/etc/profile.d/conda.sh"
    else
        echo "Error: conda.sh not found! Conda initialization failed."
        exit 1
    fi
fi
unset __conda_setup

conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then 
    echo "Failed to activate conda environment: ${CONDA_ENV_NAME} using 'conda activate'"
    exit 1
fi
echo "Conda environment activated: ${CONDA_ENV_NAME} (using conda activate)"

PROJECT_DIR="/scratch/gpfs/mg7411/ManyPaths" 
cd $PROJECT_DIR
if [ $? -ne 0 ]; then echo "Failed to cd to ${PROJECT_DIR}"; exit 1; fi
echo "Changed to project directory: $(pwd)"

# --- Determine Parameters for Current Array Task ---
ORDER_IDX=$((SLURM_ARRAY_TASK_ID % NUM_ORDERS))
TEMP_IDX=$((SLURM_ARRAY_TASK_ID / NUM_ORDERS))
DEPTH_IDX=$((TEMP_IDX % NUM_DEPTHS))
FEATURE_IDX=$((TEMP_IDX / NUM_DEPTHS))

CURRENT_FEATURES=${FEATURES_LIST[$FEATURE_IDX]}
CURRENT_DEPTH=${DEPTHS_LIST[$DEPTH_IDX]}
CURRENT_ORDER_VAL=${ORDERS_LIST[$ORDER_IDX]}

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
EPOCHS=10000
TASKS_PER_META_BATCH=4
OUTER_LR=1e-3
INNER_LR=1e-2

echo "--- Experiment Configuration for Task ID ${SLURM_ARRAY_TASK_ID} --- "
echo "Features: ${CURRENT_FEATURES}"
echo "PCFG Max Depth: ${CURRENT_DEPTH}"
echo "Order: ${ORDER_STR} (Flag: ${FIRST_ORDER_FLAG})"
echo "Epochs: ${EPOCHS}"
echo "Seed: ${SEED_VAL}"
echo "---------------------------------"

# --- Run the Python script --- 
echo "Starting Python script..."

python main.py \
    --experiment concept \
    --m mlp \
    --data-type bits \
    --num-concept-features ${CURRENT_FEATURES} \
    --pcfg-max-depth ${CURRENT_DEPTH} \
    --adaptation-steps ${ADAPTATION_STEPS} \
    ${FIRST_ORDER_FLAG} \
    --epochs ${EPOCHS} \
    --tasks_per_meta_batch ${TASKS_PER_META_BATCH} \
    --outer_lr ${OUTER_LR} \
    --inner_lr ${INNER_LR} \
    --save \
    --seed ${SEED_VAL} \
    --no_hyper_search \
    --hyper-index ${HYPER_INDEX_VAL}
    --patience 100000

EXIT_CODE=$?
echo "Python script finished with exit code ${EXIT_CODE}"

exit $EXIT_CODE 