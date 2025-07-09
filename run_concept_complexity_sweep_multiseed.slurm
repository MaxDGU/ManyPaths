#!/bin/bash

#SBATCH --job-name=mp_concept_ms
#SBATCH --output=logs/mp_concept_ms_%A_%a.out # ms for multi-seed
#SBATCH --error=logs/mp_concept_ms_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mg7411@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00 # Keep as per original successful single run
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --array=71-74,75-79,80-84,85-89


# Define Parameter Lists for Array Job Calculation
FEATURES_LIST=(8 16 32)
DEPTHS_LIST=(3 5 7)
ORDERS_LIST=(0 1) # 0 for 2nd Order (MAML), 1 for 1st Order (Meta-SGD)
SEEDS_LIST=(0 1 2 3 4) # 5 NEW Seeds 

NUM_FEATURES=${#FEATURES_LIST[@]}
NUM_DEPTHS=${#DEPTHS_LIST[@]}
NUM_ORDERS=${#ORDERS_LIST[@]}
NUM_SEEDS=${#SEEDS_LIST[@]} 

TOTAL_JOBS=$((NUM_FEATURES * NUM_DEPTHS * NUM_ORDERS * NUM_SEEDS))
MAX_ARRAY_IDX=$((TOTAL_JOBS - 1))


# --- Parameter Lists ---
FEATURES_LIST=(8 16 32)
DEPTHS_LIST=(3 5 7)
ORDERS_LIST=(0 1) # 0 for 2nd Order (MAML), 1 for 1st Order (Meta-SGD)
SEEDS_LIST=(0 1 2 3 4) # 5 Seeds

NUM_FEATURES=${#FEATURES_LIST[@]}
NUM_DEPTHS=${#DEPTHS_LIST[@]}
NUM_ORDERS=${#ORDERS_LIST[@]}
NUM_SEEDS=${#SEEDS_LIST[@]}

TOTAL_JOBS=$((NUM_FEATURES * NUM_DEPTHS * NUM_ORDERS * NUM_SEEDS))
MAX_ARRAY_IDX=$((TOTAL_JOBS - 1))


# --- Sanity Checks and Debug Info ---
echo "=== SLURM Job Information ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}" # Single job ID for the whole array
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}" # Array master job ID
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" # Current task ID in the array
echo "Total array tasks: ${TOTAL_JOBS}"

# --- Critical Array Task ID Validation ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ] && [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not properly set or is empty. Value: '${SLURM_ARRAY_TASK_ID}'"
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -gt "${MAX_ARRAY_IDX}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) is greater than MAX_ARRAY_IDX (${MAX_ARRAY_IDX})."
    exit 1
fi
echo "SLURM_ARRAY_TASK_ID check passed. Value: ${SLURM_ARRAY_TASK_ID}"

# --- Determine Parameters ---
# Order of unrolling: Seed varies fastest, then Order, then Depth, then Feature
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
TEMP_IDX_AFTER_SEED=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

ORDER_IDX=$((TEMP_IDX_AFTER_SEED % NUM_ORDERS))
TEMP_IDX_AFTER_ORDER=$((TEMP_IDX_AFTER_SEED / NUM_ORDERS))

DEPTH_IDX=$((TEMP_IDX_AFTER_ORDER % NUM_DEPTHS))
FEATURE_IDX=$((TEMP_IDX_AFTER_ORDER / NUM_DEPTHS))

CURRENT_FEATURES=${FEATURES_LIST[$FEATURE_IDX]}
CURRENT_DEPTH=${DEPTHS_LIST[$DEPTH_IDX]}
CURRENT_ORDER_VAL=${ORDERS_LIST[$ORDER_IDX]}
CURRENT_SEED=${SEEDS_LIST[$SEED_IDX]}

FIRST_ORDER_FLAG_STR=""
ORDER_STR="2ndOrd"
if [ "$CURRENT_ORDER_VAL" -eq 1 ]; then
    FIRST_ORDER_FLAG_STR="--first-order"
    ORDER_STR="1stOrd"
fi

echo "--- Parameter Calculation Debug ---"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SEED_IDX: ${SEED_IDX} -> CURRENT_SEED: ${CURRENT_SEED}"
echo "ORDER_IDX: ${ORDER_IDX} -> CURRENT_ORDER_VAL: ${CURRENT_ORDER_VAL}"
echo "DEPTH_IDX: ${DEPTH_IDX} -> CURRENT_DEPTH: ${CURRENT_DEPTH}"
echo "FEATURE_IDX: ${FEATURE_IDX} -> CURRENT_FEATURES: ${CURRENT_FEATURES}"

# --- Environment Setup ---
echo "SLURM SCRIPT: Starting job setup for Task ID ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
mkdir -p logs

module purge
module load anaconda3/2023.3  # Using the version confirmed to be available
if [ $? -ne 0 ]; then echo "ERROR: Failed to load anaconda3/2023.3 "; exit 1; fi
echo "Loaded anaconda3/2023.3 "
module load cudatoolkit/11.8
if [ $? -ne 0 ]; then echo "ERROR: Failed to load cudatoolkit/11.8"; exit 1; fi
echo "Loaded cudatoolkit/11.8"

CONDA_ENV_NAME="tensorflow"
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then echo "ERROR: Failed to activate conda env ${CONDA_ENV_NAME}"; exit 1; fi
echo "Activated conda env: ${CONDA_ENV_NAME} (Python: $(which python))"

PROJECT_DIR="/scratch/gpfs/mg7411/ManyPaths"
cd $PROJECT_DIR || exit 1

# --- Shared Python script parameters ---
ADAPTATION_STEPS=1
HYPER_INDEX_VAL=14
TASKS_PER_META_BATCH=4
OUTER_LR=1e-3
INNER_LR=1e-2
EPOCHS_VAL=100 # As per previous successful runs
CHECKPOINT_INTERVAL_VAL=40
SAVE_FLAG_STR="--save"

RESULTS_SUBDIR="results/concept_multiseed" # Specific subdir for these aggregated runs
mkdir -p ${RESULTS_SUBDIR}
SAVED_MODELS_SUBDIR="saved_models/concept_multiseed"
mkdir -p ${SAVED_MODELS_SUBDIR}

echo "--- Experiment Configuration --- "
echo "Features: ${CURRENT_FEATURES}, Depth: ${CURRENT_DEPTH}, Order: ${ORDER_STR}, Seed: ${CURRENT_SEED}"
echo "Epochs: ${EPOCHS_VAL}, Checkpoint Interval: ${CHECKPOINT_INTERVAL_VAL}"
echo "Results will be in: ${RESULTS_SUBDIR}"

python main.py \
    --experiment concept \
    --m mlp \
    --data-type bits \
    --num-concept-features ${CURRENT_FEATURES} \
    --pcfg-max-depth ${CURRENT_DEPTH} \
    --adaptation-steps ${ADAPTATION_STEPS} \
    ${FIRST_ORDER_FLAG_STR} \
    --epochs ${EPOCHS_VAL} \
    --checkpoint-interval ${CHECKPOINT_INTERVAL_VAL} \
    --tasks_per_meta_batch ${TASKS_PER_META_BATCH} \
    --outer_lr ${OUTER_LR} \
    --inner_lr ${INNER_LR} \
    ${SAVE_FLAG_STR} \
    --seed ${CURRENT_SEED} \
    --no_hyper_search \
    --hyper-index ${HYPER_INDEX_VAL} \
    --results_dir ${RESULTS_SUBDIR} \
    --saved_models_dir ${SAVED_MODELS_SUBDIR} # Pass new models dir

EXIT_CODE=$?
echo "SLURM SCRIPT: Python script finished for Task ${SLURM_ARRAY_TASK_ID} with exit code ${EXIT_CODE}"
exit $EXIT_CODE 