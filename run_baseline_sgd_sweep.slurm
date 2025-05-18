#!/bin/bash

#SBATCH --job-name=mp_baseline_sgd_sweep
#SBATCH --output=logs/mp_baseline_sgd_%A_%a.out
#SBATCH --error=logs/mp_baseline_sgd_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mg7411@princeton.edu # Your email address
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 
#SBATCH --time=02:00:00 # Adjust time based on num_tasks_to_evaluate and num_sgd_steps_per_task
#SBATCH --gres=gpu:1
#SBATCH --partition=pli # Or your default partition
#SBATCH --account=nam # Or your default account

# --- Parameter Lists for Sweep ---
FEATURES_LIST=(8 16 32)
DEPTHS_LIST=(3 5 7)
# Order is not applicable for baseline SGD

NUM_FEATURES=${#FEATURES_LIST[@]}
NUM_DEPTHS=${#DEPTHS_LIST[@]}
TOTAL_JOBS=$((NUM_FEATURES * NUM_DEPTHS)) # 3 features * 3 depths = 9 jobs

#SBATCH --array=0-$((TOTAL_JOBS - 1)) # Array indices from 0 to (TOTAL_JOBS - 1)

# --- Environment Setup (Same as your MAML script) ---
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total expected jobs: $TOTAL_JOBS"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

mkdir -p logs

echo "Loading modules..."
module purge
module load Anaconda3/2023.3 # Or your preferred Anaconda module
module load cudatoolkit/11.8 # Or your preferred CUDA toolkit

CONDA_ENV_NAME="tensorflow" # Or your conda environment name
echo "Attempting to activate conda environment: ${CONDA_ENV_NAME}"

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
        echo "Error: conda.sh not found! Conda initialization failed."
        exit 1
    fi
fi
unset __conda_setup

conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then 
    echo "Failed to activate conda environment: ${CONDA_ENV_NAME}"
    exit 1
fi
echo "Conda environment activated: ${CONDA_ENV_NAME}"

PROJECT_DIR="/scratch/gpfs/mg7411/ManyPaths" # Your project directory on the cluster
cd $PROJECT_DIR
if [ $? -ne 0 ]; then echo "Failed to cd to ${PROJECT_DIR}"; exit 1; fi
echo "Changed to project directory: $(pwd)"

# --- Determine Parameters for Current Array Task ---
# Since there's no ORDER_LIST, indexing is simpler
DEPTH_IDX=$((SLURM_ARRAY_TASK_ID % NUM_DEPTHS))
FEATURE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_DEPTHS))

CURRENT_FEATURES=${FEATURES_LIST[$FEATURE_IDX]}
CURRENT_DEPTH=${DEPTHS_LIST[$DEPTH_IDX]}

# --- Shared parameters for main_baseline_sgd.py ---
SEED_VAL=0
HYPER_INDEX_VAL=14 # Corresponds to the MAML runs for model structure consistency
NUM_TASKS_TO_EVALUATE=200 # Number of random tasks to average over per setting
NUM_SGD_STEPS_PER_TASK=100 # Number of training steps on each task's support set
LR_BASELINE=1e-3
RUN_NAME_BASELINE="baseline_sgd_run1" # Consistent run name for this sweep

echo "--- Baseline SGD Experiment Configuration for Task ID ${SLURM_ARRAY_TASK_ID} --- "
echo "Features: ${CURRENT_FEATURES}"
echo "PCFG Max Depth: ${CURRENT_DEPTH}"
echo "Seed: ${SEED_VAL}"
echo "Num Tasks to Evaluate: ${NUM_TASKS_TO_EVALUATE}"
echo "SGD Steps per Task: ${NUM_SGD_STEPS_PER_TASK}"
echo "Learning Rate: ${LR_BASELINE}"
echo "Run Name: ${RUN_NAME_BASELINE}"
echo "Hyper Index (for model init): ${HYPER_INDEX_VAL}"
echo "---------------------------------"

# --- Run the Python script --- 
echo "Starting Python script for baseline SGD..."

python main_baseline_sgd.py \
    --experiment concept \
    --m mlp \
    --data-type bits \
    --num-concept-features ${CURRENT_FEATURES} \
    --pcfg-max-depth ${CURRENT_DEPTH} \
    --num-tasks-to-evaluate ${NUM_TASKS_TO_EVALUATE} \
    --num-sgd-steps-per-task ${NUM_SGD_STEPS_PER_TASK} \
    --lr ${LR_BASELINE} \
    --seed ${SEED_VAL} \
    --hyper-index ${HYPER_INDEX_VAL} \
    --run-name ${RUN_NAME_BASELINE} \
    --verbose # Optional: add if you want detailed per-task printouts in logs
    # --no-save-task-models # Uncomment if you don't want to save all task-specific models

EXIT_CODE=$?
echo "Python script finished with exit code ${EXIT_CODE}"

exit $EXIT_CODE 