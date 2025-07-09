#!/bin/bash

#SBATCH --job-name=mp_baseline_ms # ms for multi-seed
#SBATCH --output=logs/mp_baseline_ms_%A_%a.out
#SBATCH --error=logs/mp_baseline_ms_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mg7411@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00 # Baseline SGD is quick per combo
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --array=0-36


# Define Parameter Lists for Array Job Calculation
FEATURES_LIST=(8 16 32)
DEPTHS_LIST=(3 5 7)
SEEDS_LIST=(1 2 3 4) # 4 NEW Seeds (seed 0 is assumed to exist separately)

NUM_FEATURES=${#FEATURES_LIST[@]}
NUM_DEPTHS=${#DEPTHS_LIST[@]}
NUM_SEEDS=${#SEEDS_LIST[@]} # Should be 4 for the new runs

TOTAL_JOBS=$((NUM_FEATURES * NUM_DEPTHS * NUM_SEEDS))
MAX_ARRAY_IDX=$((TOTAL_JOBS - 1))


# --- Sanity Checks and Debug Info ---
echo "=== SLURM Job Information ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total array tasks: ${TOTAL_JOBS}"

# --- Critical Array Task ID Validation ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ] && [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not properly set. Value: '${SLURM_ARRAY_TASK_ID}'"
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -gt "${MAX_ARRAY_IDX}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) > MAX_ARRAY_IDX (${MAX_ARRAY_IDX})."
    exit 1
fi
echo "SLURM_ARRAY_TASK_ID check passed. Value: ${SLURM_ARRAY_TASK_ID}"

# --- Determine Parameters ---
# Order of unrolling: Seed varies fastest, then Depth, then Feature
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
TEMP_IDX_AFTER_SEED=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

DEPTH_IDX=$((TEMP_IDX_AFTER_SEED % NUM_DEPTHS))
FEATURE_IDX=$((TEMP_IDX_AFTER_SEED / NUM_DEPTHS))

CURRENT_FEATURES=${FEATURES_LIST[$FEATURE_IDX]}
CURRENT_DEPTH=${DEPTHS_LIST[$DEPTH_IDX]}
CURRENT_SEED=${SEEDS_LIST[$SEED_IDX]}

echo "--- Parameter Calculation Debug ---"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SEED_IDX: ${SEED_IDX} -> CURRENT_SEED: ${CURRENT_SEED}"
echo "DEPTH_IDX: ${DEPTH_IDX} -> CURRENT_DEPTH: ${CURRENT_DEPTH}"
echo "FEATURE_IDX: ${FEATURE_IDX} -> CURRENT_FEATURES: ${CURRENT_FEATURES}"

# --- Environment Setup ---
# (Identical to the MetaSGD multi-seed script)
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

# --- Parameters for main_baseline_sgd.py ---
HYPER_INDEX_VAL=14
NUM_TASKS_TO_EVALUATE=200
NUM_SGD_STEPS_PER_TASK=100
LR_BASELINE=1e-3

# Construct a run name that includes -seed{S} to ensure results go into unique subdirs per seed
# main_baseline_sgd.py already puts results into results/baseline_sgd/{args.run_name}/
# So, the run_name itself should be unique per seed for directory separation.
RUN_NAME_PREFIX="baseline_ms"
CURRENT_RUN_NAME="${RUN_NAME_PREFIX}_feat${CURRENT_FEATURES}_dep${CURRENT_DEPTH}_seed${CURRENT_SEED}"

echo "--- Baseline SGD Experiment Configuration --- "
echo "Features: ${CURRENT_FEATURES}, Depth: ${CURRENT_DEPTH}, Seed: ${CURRENT_SEED}"
echo "Run Name for output: ${CURRENT_RUN_NAME}"

python main_baseline_sgd.py \
    --num-concept-features ${CURRENT_FEATURES} \
    --pcfg-max-depth ${CURRENT_DEPTH} \
    --seed ${CURRENT_SEED} \
    --hyper-index ${HYPER_INDEX_VAL} \
    --num-tasks-to-evaluate ${NUM_TASKS_TO_EVALUATE} \
    --num-sgd-steps-per-task ${NUM_SGD_STEPS_PER_TASK} \
    --lr ${LR_BASELINE} \
    --run-name "${CURRENT_RUN_NAME}" \
    --verbose

EXIT_CODE=$?
echo "SLURM SCRIPT: Python script finished for Task ${SLURM_ARRAY_TASK_ID} with exit code ${EXIT_CODE}"
exit $EXIT_CODE 