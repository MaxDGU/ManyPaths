#!/bin/bash

#SBATCH --job-name=mp_baseline_sgd_sweep_v2
#SBATCH --output=logs/mp_baseline_sgd_sweep_v2_%A_%a.out
#SBATCH --error=logs/mp_baseline_sgd_sweep_v2_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mg7411@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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

# --- Parameter Lists ---
FEATURES_LIST=(8 16 32)
DEPTHS_LIST=(3 5 7)
NUM_FEATURES=${#FEATURES_LIST[@]}
NUM_DEPTHS=${#DEPTHS_LIST[@]}

TOTAL_JOBS_EXPECTED=$((NUM_FEATURES * NUM_DEPTHS))
echo "Expected total jobs based on FEATURES_LIST and DEPTHS_LIST: ${TOTAL_JOBS_EXPECTED}"
if [ "${SLURM_ARRAY_TASK_ID}" -gt "${SLURM_ARRAY_TASK_MAX}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) is greater than SLURM_ARRAY_TASK_MAX (${SLURM_ARRAY_TASK_MAX})."
    exit 1
fi

# --- Calculate Indices (using bash arithmetic, should be safe with validated SLURM_ARRAY_TASK_ID) ---
DEPTH_IDX=$((SLURM_ARRAY_TASK_ID % NUM_DEPTHS))
FEATURE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_DEPTHS))

echo "--- Index Calculation ---"
echo "Calculated DEPTH_IDX: ${DEPTH_IDX}"
echo "Calculated FEATURE_IDX: ${FEATURE_IDX}"
echo "-------------------------"

if [ "${FEATURE_IDX}" -ge "${NUM_FEATURES}" ]; then
    echo "ERROR: Calculated FEATURE_IDX (${FEATURE_IDX}) is out of bounds for FEATURES_LIST (size ${NUM_FEATURES})."
    exit 1
fi
if [ "${DEPTH_IDX}" -ge "${NUM_DEPTHS}" ]; then
    echo "ERROR: Calculated DEPTH_IDX (${DEPTH_IDX}) is out of bounds for DEPTHS_LIST (size ${NUM_DEPTHS})."
    exit 1
fi

CURRENT_FEATURES=${FEATURES_LIST[${FEATURE_IDX}]}
CURRENT_DEPTH=${DEPTHS_LIST[${DEPTH_IDX}]}

if [ -z "${CURRENT_FEATURES}" ] || [ -z "${CURRENT_DEPTH}" ]; then
    echo "ERROR: Failed to determine CURRENT_FEATURES or CURRENT_DEPTH."
    echo "CURRENT_FEATURES: '${CURRENT_FEATURES}'"
    echo "CURRENT_DEPTH: '${CURRENT_DEPTH}'"
    exit 1
fi
echo "Successfully determined parameters: FEATURES=${CURRENT_FEATURES}, DEPTH=${CURRENT_DEPTH}"

# --- Environment Setup ---
echo "SLURM SCRIPT: Setting up environment..."
mkdir -p logs
echo "SLURM SCRIPT: Created logs directory if it didn't exist."

echo "SLURM SCRIPT: Loading modules..."
module purge
echo "SLURM SCRIPT: Purged modules."
module load anaconda3/2023.3 
if [ $? -ne 0 ]; then echo "SLURM SCRIPT ERROR: Failed to load anaconda3/2023.3  module."; exit 1; fi
echo "SLURM SCRIPT: Loaded Anaconda module (anaconda3/2023.3 )."
module load cudatoolkit/11.8
if [ $? -ne 0 ]; then echo "SLURM SCRIPT ERROR: Failed to load cudatoolkit/11.8 module."; exit 1; fi
echo "SLURM SCRIPT: Loaded CUDA toolkit."

CONDA_ENV_NAME="tensorflow"
echo "SLURM SCRIPT: Attempting to activate conda environment: ${CONDA_ENV_NAME}"

__conda_setup="$('/usr/licensed/anaconda3/2023.3 /bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/licensed/anaconda3/2023.3 /etc/profile.d/conda.sh" ]; then
        . "/usr/licensed/anaconda3/2023.3 /etc/profile.d/conda.sh"
    else
        echo "SLURM SCRIPT ERROR: conda.sh not found for anaconda3/2023.3 . Conda initialization failed."
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

# --- Parameters for main_baseline_sgd.py ---
SEED_VAL=0
HYPER_INDEX_VAL=14
NUM_TASKS_TO_EVALUATE=200
NUM_SGD_STEPS_PER_TASK=100
LR_BASELINE=1e-3
RUN_NAME_BASELINE="baseline_sgd_run_v2"

echo "--- Baseline SGD Experiment Configuration for SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} --- "
echo "Features: ${CURRENT_FEATURES}"
echo "PCFG Max Depth: ${CURRENT_DEPTH}"
echo "Seed: ${SEED_VAL}"
echo "Hyper Index (for model init): ${HYPER_INDEX_VAL}"
echo "Num Tasks to Evaluate: ${NUM_TASKS_TO_EVALUATE}"
echo "SGD Steps per Task: ${NUM_SGD_STEPS_PER_TASK}"
echo "Learning Rate: ${LR_BASELINE}"
echo "Run Name: ${RUN_NAME_BASELINE}"
echo "---------------------------------"

echo "SLURM SCRIPT: Starting Python script (main_baseline_sgd.py)..."
python main_baseline_sgd.py \
    --num-concept-features ${CURRENT_FEATURES} \
    --pcfg-max-depth ${CURRENT_DEPTH} \
    --seed ${SEED_VAL} \
    --hyper-index ${HYPER_INDEX_VAL} \
    --num-tasks-to-evaluate ${NUM_TASKS_TO_EVALUATE} \
    --num-sgd-steps-per-task ${NUM_SGD_STEPS_PER_TASK} \
    --lr ${LR_BASELINE} \
    --run-name ${RUN_NAME_BASELINE} \
    --verbose

EXIT_CODE=$?
echo "SLURM SCRIPT: Python script (main_baseline_sgd.py) finished with exit code ${EXIT_CODE}"

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "SLURM SCRIPT ERROR: Python script failed. Check error log: logs/mp_baseline_sgd_sweep_v2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
    exit ${EXIT_CODE}
fi

echo "SLURM SCRIPT: Job ${SLURM_JOB_ID} (Array Task ${SLURM_ARRAY_TASK_ID}) completed successfully."
exit 0 