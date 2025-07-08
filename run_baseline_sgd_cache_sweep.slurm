#!/bin/bash

#SBATCH --job-name=mp_baseline_cache_sweep
#SBATCH --output=logs/mp_baseline_cache_%A_%a.out
#SBATCH --error=logs/mp_baseline_cache_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mg7411@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --array=0-8

# --- Parameter Lists for Cache-Based Baseline SGD ---
CACHE_CONFIGS=(
    "data/concept_cache/pcfg_tasks_f8_d3_s2p3n_q5p5n_t10000.pt 8 3"
    "data/concept_cache/pcfg_tasks_f8_d5_s2p3n_q5p5n_t10000.pt 8 5"
    "data/concept_cache/pcfg_tasks_f32_d3_s2p3n_q5p5n_t10000.pt 32 3"
)
SEEDS_LIST=(1 2 3)

NUM_CACHES=${#CACHE_CONFIGS[@]}
NUM_SEEDS=${#SEEDS_LIST[@]}
TOTAL_JOBS=$((NUM_CACHES * NUM_SEEDS))

echo "🚀 SGD BASELINE EXPERIMENTS"
echo "Running vanilla SGD baseline for concept learning tasks"
echo "Host: $(hostname), Date: $(date)"
echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}, Node: $(hostname)"
echo "============================================"

# --- Array Task ID Validation ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ] && [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not properly set. Value: '${SLURM_ARRAY_TASK_ID}'"
    exit 1
fi

MAX_ARRAY_IDX=$((TOTAL_JOBS - 1))
if [ "${SLURM_ARRAY_TASK_ID}" -gt "${MAX_ARRAY_IDX}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) > MAX_ARRAY_IDX (${MAX_ARRAY_IDX})."
    exit 1
fi

# --- Determine Parameters ---
# Order: Seed varies fastest, then Cache
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
CACHE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

CURRENT_SEED=${SEEDS_LIST[$SEED_IDX]}
CACHE_CONFIG=${CACHE_CONFIGS[$CACHE_IDX]}

# Parse cache configuration
read -r CACHE_PATH FEATURES DEPTH <<< "$CACHE_CONFIG"

echo "--- Task Configuration ---"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Cache: ${CACHE_PATH}"
echo "Features: ${FEATURES}, Depth: ${DEPTH}, Seed: ${CURRENT_SEED}"
echo "-------------------------"

# --- Environment Setup ---
echo "Loading modules..."
module purge
module load anaconda3/2023.3 
if [ $? -ne 0 ]; then
    echo "ERROR: Unable to locate a modulefile for 'Anaconda3/2023.3'"
    echo "Attempting to activate conda environment: tensorflow"
    conda activate tensorflow
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate conda environment: tensorflow"
        exit 1
    fi
    echo "Conda environment activated: tensorflow"
else
    echo "Loading modules..."
    module load cudatoolkit/11.8
    if [ $? -ne 0 ]; then echo "ERROR: Failed to load cudatoolkit/11.8"; exit 1; fi
    
    echo "Attempting to activate conda environment: tensorflow"
    conda activate tensorflow
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate conda environment: tensorflow"
        exit 1
    fi
    echo "Conda environment activated: tensorflow"
fi

PROJECT_DIR="/scratch/gpfs/mg7411/ManyPaths"
cd $PROJECT_DIR || exit 1
echo "Changed to project directory: $(pwd)"

# --- Cache-specific Configuration ---
if [[ "$CACHE_PATH" == *"f8_d3"* ]]; then
    CACHE_NAME="F8D3"
    EXPECTED_TIME="2h"
elif [[ "$CACHE_PATH" == *"f8_d5"* ]]; then
    CACHE_NAME="F8D5"
    EXPECTED_TIME="2h"
elif [[ "$CACHE_PATH" == *"f32_d3"* ]]; then
    CACHE_NAME="F32D3"
    EXPECTED_TIME="6h"
else
    CACHE_NAME="UNKNOWN"
    EXPECTED_TIME="unknown"
fi

echo "🧠 STARTING ${CACHE_NAME} CACHE EXPERIMENTS"
echo "${CACHE_NAME} Cache - Est: ${EXPECTED_TIME}"
echo ""

# --- Parameters for main_baseline_sgd.py ---
HYPER_INDEX_VAL=14
NUM_TASKS_TO_EVALUATE=100  # Use cache size
NUM_SGD_STEPS_PER_TASK=32
LR_BASELINE=0.01
RUN_NAME_PREFIX="baseline_run1"

echo "🌱 Running with seed ${CURRENT_SEED}"

python main_baseline_sgd.py \
    --cache "${CACHE_PATH}" \
    --num-concept-features ${FEATURES} \
    --pcfg-max-depth ${DEPTH} \
    --seed ${CURRENT_SEED} \
    --hyper-index ${HYPER_INDEX_VAL} \
    --num-tasks-to-evaluate ${NUM_TASKS_TO_EVALUATE} \
    --epochs ${NUM_SGD_STEPS_PER_TASK} \
    --lr ${LR_BASELINE} \
    --run-name "${RUN_NAME_PREFIX}" \
    --no-save-task-models \
    --verbose

EXIT_CODE=$?
echo "${CACHE_NAME} Cache completed at $(date)"

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Python script failed for ${CACHE_NAME} seed ${CURRENT_SEED}"
    exit ${EXIT_CODE}
fi

echo "============================================"
echo "🎉 SGD BASELINE EXPERIMENTS COMPLETED"
echo "Completed at: $(date)"
echo "============================================" 