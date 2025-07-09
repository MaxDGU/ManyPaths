#!/bin/bash

#SBATCH --job-name=mp_landscape_logging
#SBATCH --output=logs/mp_landscape_logging_%A_%a.out
#SBATCH --error=logs/mp_landscape_logging_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mg7411@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --array=0-8

# --- Sanity Checks and Debug Info ---
echo "=== SLURM Job Information ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_ARRAY_TASK_COUNT: ${SLURM_ARRAY_TASK_COUNT}"
echo "SLURM_ARRAY_TASK_MIN: ${SLURM_ARRAY_TASK_MIN}"
echo "SLURM_ARRAY_TASK_MAX: ${SLURM_ARRAY_TASK_MAX}"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Date: $(date)"
echo "============================="

# --- Critical Array Task ID Validation ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ] && [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not properly set or is empty."
    echo "This script is intended to be run as a SLURM job array."
    echo "SLURM_ARRAY_TASK_ID value: '${SLURM_ARRAY_TASK_ID}'"
    exit 1
fi
echo "SLURM_ARRAY_TASK_ID check passed. Value: ${SLURM_ARRAY_TASK_ID}"

# --- Parameter Lists for 3 Complexity Tiers ---
# Easy = F8 D3, Medium = F16 D5, Complex = F32 D7
FEATURES_LIST=(8 16 32)
DEPTHS_LIST=(3 5 7)
SEEDS_LIST=(1 2 3)

NUM_FEATURES=${#FEATURES_LIST[@]}
NUM_DEPTHS=${#DEPTHS_LIST[@]}
NUM_SEEDS=${#SEEDS_LIST[@]}

TOTAL_JOBS_EXPECTED=$((NUM_FEATURES * NUM_SEEDS))
echo "Expected total jobs based on FEATURES_LIST and SEEDS_LIST: ${TOTAL_JOBS_EXPECTED}"
if [ "${SLURM_ARRAY_TASK_ID}" -gt "${SLURM_ARRAY_TASK_MAX}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) is greater than SLURM_ARRAY_TASK_MAX (${SLURM_ARRAY_TASK_MAX})."
    exit 1
fi

# --- Calculate Indices ---
# Map task ID to (features, depth, seed) combination
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
FEATURE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

echo "--- Index Calculation ---"
echo "Calculated SEED_IDX: ${SEED_IDX}"
echo "Calculated FEATURE_IDX: ${FEATURE_IDX}"
echo "-------------------------"

if [ "${FEATURE_IDX}" -ge "${NUM_FEATURES}" ]; then
    echo "ERROR: Calculated FEATURE_IDX (${FEATURE_IDX}) is out of bounds for FEATURES_LIST (size ${NUM_FEATURES})."
    exit 1
fi
if [ "${SEED_IDX}" -ge "${NUM_SEEDS}" ]; then
    echo "ERROR: Calculated SEED_IDX (${SEED_IDX}) is out of bounds for SEEDS_LIST (size ${NUM_SEEDS})."
    exit 1
fi

CURRENT_FEATURES=${FEATURES_LIST[${FEATURE_IDX}]}
CURRENT_DEPTH=${DEPTHS_LIST[${FEATURE_IDX}]}  # Use same index for depth (F8D3, F16D5, F32D7)
CURRENT_SEED=${SEEDS_LIST[${SEED_IDX}]}

if [ -z "${CURRENT_FEATURES}" ] || [ -z "${CURRENT_DEPTH}" ] || [ -z "${CURRENT_SEED}" ]; then
    echo "ERROR: Failed to determine CURRENT_FEATURES, CURRENT_DEPTH, or CURRENT_SEED."
    echo "CURRENT_FEATURES: '${CURRENT_FEATURES}'"
    echo "CURRENT_DEPTH: '${CURRENT_DEPTH}'"
    echo "CURRENT_SEED: '${CURRENT_SEED}'"
    exit 1
fi
echo "Successfully determined parameters: FEATURES=${CURRENT_FEATURES}, DEPTH=${CURRENT_DEPTH}, SEED=${CURRENT_SEED}"

# --- Environment Setup ---
echo "SLURM SCRIPT: Setting up environment..."
mkdir -p logs
echo "SLURM SCRIPT: Created logs directory if it didn't exist."

echo "SLURM SCRIPT: Loading modules..."
module purge
echo "SLURM SCRIPT: Purged modules."
module load anaconda3/2023.3 
if [ $? -ne 0 ]; then echo "SLURM SCRIPT ERROR: Failed to load anaconda3/2023.3 module."; exit 1; fi
echo "SLURM SCRIPT: Loaded Anaconda module (anaconda3/2023.3)."
module load cudatoolkit/11.8
if [ $? -ne 0 ]; then echo "SLURM SCRIPT ERROR: Failed to load cudatoolkit/11.8 module."; exit 1; fi
echo "SLURM SCRIPT: Loaded CUDA toolkit."

CONDA_ENV_NAME="tensorflow"
echo "SLURM SCRIPT: Attempting to activate conda environment: ${CONDA_ENV_NAME}"

__conda_setup="$('/usr/licensed/anaconda3/2023.3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/licensed/anaconda3/2023.3/etc/profile.d/conda.sh" ]; then
        . "/usr/licensed/anaconda3/2023.3/etc/profile.d/conda.sh"
    else
        echo "SLURM SCRIPT ERROR: conda.sh not found for anaconda3/2023.3. Conda initialization failed."
        exit 1
    fi
fi
unset __conda_setup

conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "SLURM SCRIPT ERROR: Failed to activate conda environment: ${CONDA_ENV_NAME}"
    exit 1
fi
echo "SLURM SCRIPT: Conda environment activated: ${CONDA_ENV_NAME} (verified by which python: $(which python))"

PROJECT_DIR="/scratch/gpfs/mg7411/ManyPaths"
cd $PROJECT_DIR
if [ $? -ne 0 ]; then echo "SLURM SCRIPT ERROR: Failed to cd to ${PROJECT_DIR}"; exit 1; fi
echo "SLURM SCRIPT: Changed to project directory: $(pwd)"

# --- Experiment Configuration ---
EPOCHS=200
ADAPTATION_STEPS=10
CHECKPOINT_EVERY=10

echo "--- Landscape Logging Experiment Configuration for SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} ---"
echo "Features: ${CURRENT_FEATURES}"
echo "PCFG Max Depth: ${CURRENT_DEPTH}"
echo "Seed: ${CURRENT_SEED}"
echo "Epochs: ${EPOCHS}"
echo "Adaptation Steps: ${ADAPTATION_STEPS}"
echo "Checkpoint Every: ${CHECKPOINT_EVERY} steps"
echo "--------------------------------"

# --- Run Meta-SGD with Landscape Logging ---
echo "SLURM SCRIPT: Starting Meta-SGD with landscape logging..."
python main.py \
    --experiment concept --m mlp --data-type bits \
    --num-concept-features ${CURRENT_FEATURES} --pcfg-max-depth ${CURRENT_DEPTH} \
    --adaptation-steps ${ADAPTATION_STEPS} --epochs ${EPOCHS} \
    --seed ${CURRENT_SEED} \
    --no_hyper_search --hyper-index 14 \
    --log_landscape --checkpoint_every ${CHECKPOINT_EVERY} \
    --save --verbose-debug

META_SGD_EXIT_CODE=$?
echo "SLURM SCRIPT: Meta-SGD finished with exit code ${META_SGD_EXIT_CODE}"

if [ ${META_SGD_EXIT_CODE} -ne 0 ]; then
    echo "SLURM SCRIPT ERROR: Meta-SGD failed. Check error log."
    exit ${META_SGD_EXIT_CODE}
fi

# --- Run SGD Baseline with Landscape Logging ---
echo "SLURM SCRIPT: Starting SGD baseline with landscape logging..."

# Use cache if available for faster execution
CACHE_PATH="data/concept_cache/pcfg_tasks_f${CURRENT_FEATURES}_d${CURRENT_DEPTH}_s2p3n_q5p5n_t10000.pt"

python main_baseline_sgd.py \
    --cache ${CACHE_PATH} \
    --epochs ${EPOCHS} --lr 0.01 --momentum 0.9 \
    --seed ${CURRENT_SEED} \
    --no_hyper_search --hyper-index 14 \
    --log_landscape --checkpoint_every ${CHECKPOINT_EVERY} --save-checkpoints

SGD_EXIT_CODE=$?
echo "SLURM SCRIPT: SGD baseline finished with exit code ${SGD_EXIT_CODE}"

if [ ${SGD_EXIT_CODE} -ne 0 ]; then
    echo "SLURM SCRIPT ERROR: SGD baseline failed. Check error log."
    exit ${SGD_EXIT_CODE}
fi

echo "SLURM SCRIPT: Job ${SLURM_JOB_ID} (Array Task ${SLURM_ARRAY_TASK_ID}) completed successfully."
echo "SLURM SCRIPT: Both Meta-SGD and SGD completed for F${CURRENT_FEATURES}D${CURRENT_DEPTH} with seed ${CURRENT_SEED}"
exit 0 