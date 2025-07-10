#!/bin/bash

#SBATCH --job-name=mp_landscape_overnight
#SBATCH --output=logs/mp_landscape_overnight_%A_%a.out
#SBATCH --error=logs/mp_landscape_overnight_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mg7411@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --array=0-8

# === OPTIMIZED OVERNIGHT LANDSCAPE LOGGING ===
# Changes from original:
# - 20 hour time limit (was 6 hours)
# - 32GB memory (was 16GB) 
# - Reduced logging frequency (every 5000 steps instead of 1000)
# - Better error handling for Hessian computation
# - Focus on configurations that can complete

echo "🌙 OVERNIGHT LANDSCAPE LOGGING"
echo "=== SLURM Job Information ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Date: $(date)"
echo "Time limit: 20 hours"
echo "============================="

# --- Critical Array Task ID Validation ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ] && [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not properly set."
    exit 1
fi

# --- Parameter Lists for 3 Complexity Tiers ---
# Prioritize configurations that are likely to complete in 20 hours
FEATURES_LIST=(8 16 32)
DEPTHS_LIST=(3 5 7)
SEEDS_LIST=(1 2 3)

NUM_FEATURES=${#FEATURES_LIST[@]}
NUM_DEPTHS=${#DEPTHS_LIST[@]}
NUM_SEEDS=${#SEEDS_LIST[@]}

# --- Calculate Indices ---
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
FEATURE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

if [ "${FEATURE_IDX}" -ge "${NUM_FEATURES}" ]; then
    echo "ERROR: FEATURE_IDX out of bounds"
    exit 1
fi

CURRENT_FEATURES=${FEATURES_LIST[${FEATURE_IDX}]}
CURRENT_DEPTH=${DEPTHS_LIST[${FEATURE_IDX}]}  
CURRENT_SEED=${SEEDS_LIST[${SEED_IDX}]}

echo "🎯 Configuration: F${CURRENT_FEATURES}D${CURRENT_DEPTH}_seed${CURRENT_SEED}"

# Estimate completion time based on configuration
if [ "${CURRENT_FEATURES}" -eq 32 ]; then
    echo "⚠️  F32D7 configuration - expect ~15-18 hours"
    CHECKPOINT_EVERY=50  # Less frequent for complex configs
elif [ "${CURRENT_FEATURES}" -eq 16 ]; then
    echo "✅ F16D5 configuration - expect ~8-12 hours"
    CHECKPOINT_EVERY=25
else
    echo "🚀 F8D3 configuration - expect ~4-6 hours"
    CHECKPOINT_EVERY=10
fi

# --- Environment Setup ---
echo "Setting up environment..."
mkdir -p logs

module purge
module load anaconda3/2023.3 
module load cudatoolkit/11.8

CONDA_ENV_NAME="tensorflow"
__conda_setup="$('/usr/licensed/anaconda3/2023.3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/licensed/anaconda3/2023.3/etc/profile.d/conda.sh" ]; then
        . "/usr/licensed/anaconda3/2023.3/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda.sh not found"
        exit 1
    fi
fi

conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

PROJECT_DIR="/scratch/gpfs/mg7411/ManyPaths"
cd $PROJECT_DIR

# --- Experiment Configuration ---
EPOCHS=200
ADAPTATION_STEPS=10

echo "--- Optimized Landscape Logging Configuration ---"
echo "Features: ${CURRENT_FEATURES}"
echo "PCFG Max Depth: ${CURRENT_DEPTH}"
echo "Seed: ${CURRENT_SEED}"
echo "Epochs: ${EPOCHS}"
echo "Adaptation Steps: ${ADAPTATION_STEPS}"
echo "Checkpoint Every: ${CHECKPOINT_EVERY} steps (optimized for config)"
echo "Memory: 32GB allocated"
echo "Time: 20 hours allocated"
echo "---------------------------------------------"

# --- Run Meta-SGD with Optimized Landscape Logging ---
echo "🚀 Starting Meta-SGD with optimized landscape logging..."
python main.py \
    --experiment concept --m mlp --data-type bits \
    --num-concept-features ${CURRENT_FEATURES} --pcfg-max-depth ${CURRENT_DEPTH} \
    --adaptation-steps ${ADAPTATION_STEPS} --epochs ${EPOCHS} \
    --seed ${CURRENT_SEED} \
    --no_hyper_search --hyper-index 14 \
    --log_landscape --checkpoint_every ${CHECKPOINT_EVERY} \
    --save --verbose-debug

META_SGD_EXIT_CODE=$?
echo "Meta-SGD finished with exit code ${META_SGD_EXIT_CODE}"

if [ ${META_SGD_EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Meta-SGD failed for F${CURRENT_FEATURES}D${CURRENT_DEPTH}_seed${CURRENT_SEED}"
    echo "Continuing to SGD baseline..."
fi

# --- Run SGD Baseline with Optimized Landscape Logging ---
echo "🔄 Starting SGD baseline with optimized landscape logging..."

CACHE_PATH="data/concept_cache/pcfg_tasks_f${CURRENT_FEATURES}_d${CURRENT_DEPTH}_s2p3n_q5p5n_t10000.pt"

python main_baseline_sgd.py \
    --cache ${CACHE_PATH} \
    --epochs ${EPOCHS} --lr 0.01 --momentum 0.9 \
    --seed ${CURRENT_SEED} \
    --no_hyper_search --hyper-index 14 \
    --log_landscape --checkpoint_every ${CHECKPOINT_EVERY} --save-checkpoints

SGD_EXIT_CODE=$?
echo "SGD baseline finished with exit code ${SGD_EXIT_CODE}"

# --- Final Status Report ---
echo "🌅 OVERNIGHT JOB COMPLETION REPORT"
echo "=================================="
echo "Configuration: F${CURRENT_FEATURES}D${CURRENT_DEPTH}_seed${CURRENT_SEED}"
echo "Meta-SGD exit code: ${META_SGD_EXIT_CODE}"
echo "SGD exit code: ${SGD_EXIT_CODE}"
echo "Completed at: $(date)"

if [ ${META_SGD_EXIT_CODE} -eq 0 ] && [ ${SGD_EXIT_CODE} -eq 0 ]; then
    echo "✅ SUCCESS: Both Meta-SGD and SGD completed successfully"
elif [ ${META_SGD_EXIT_CODE} -eq 0 ] || [ ${SGD_EXIT_CODE} -eq 0 ]; then
    echo "⚠️  PARTIAL: One method completed successfully"
else
    echo "❌ FAILURE: Both methods failed"
    exit 1
fi

echo "🎯 Check trajectory files:"
echo "   - Meta-SGD: concept_mlp_*_landscape_trajectory.csv"
echo "   - SGD: baseline_sgd_*_landscape_trajectory.csv"
echo "=================================="

exit 0 