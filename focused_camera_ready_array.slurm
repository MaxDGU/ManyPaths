#!/bin/bash
#SBATCH --job-name=camera_ready_array
#SBATCH --output=camera_ready_array_%A_%a.out
#SBATCH --error=camera_ready_array_%A_%a.err
#SBATCH --time=80:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu
#SBATCH --array=0-17

echo "🚀 CAMERA-READY ARRAY JOB"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname), Date: $(date)"

module load anaconda3/2023.3
module load cudatoolkit/11.8
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=0

# Define experimental configurations
# Array mapping: 0-5=F8D3, 6-11=F16D3, 12-17=F32D3
# Within each group: 0,1,2=K1_S1,S2,S3 and 3,4,5=K10_S1,S2,S3

declare -a CONFIGS=(
    # F8D3 experiments (indices 0-5)
    "8 3 1 1"   # F8D3_K1_S1
    "8 3 1 2"   # F8D3_K1_S2  
    "8 3 1 3"   # F8D3_K1_S3
    "8 3 10 1"  # F8D3_K10_S1
    "8 3 10 2"  # F8D3_K10_S2
    "8 3 10 3"  # F8D3_K10_S3
    
    # F16D3 experiments (indices 6-11)
    "16 3 1 1"  # F16D3_K1_S1
    "16 3 1 2"  # F16D3_K1_S2
    "16 3 1 3"  # F16D3_K1_S3
    "16 3 10 1" # F16D3_K10_S1
    "16 3 10 2" # F16D3_K10_S2
    "16 3 10 3" # F16D3_K10_S3
    
    # F32D3 experiments (indices 12-17)
    "32 3 1 1"  # F32D3_K1_S1
    "32 3 1 2"  # F32D3_K1_S2
    "32 3 1 3"  # F32D3_K1_S3
    "32 3 10 1" # F32D3_K10_S1
    "32 3 10 2" # F32D3_K10_S2
    "32 3 10 3" # F32D3_K10_S3
)

# Get configuration for this array task
CONFIG_STR=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read -r FEATURES DEPTH ADAPT_STEPS SEED <<< "$CONFIG_STR"

# Set timeout based on complexity and adaptation steps
if [ "$FEATURES" -eq 8 ]; then
    if [ "$ADAPT_STEPS" -eq 1 ]; then
        TIMEOUT=72000  # 20h for F8D3_K1
    else
        TIMEOUT=108000 # 30h for F8D3_K10
    fi
elif [ "$FEATURES" -eq 16 ]; then
    if [ "$ADAPT_STEPS" -eq 1 ]; then
        TIMEOUT=144000 # 40h for F16D3_K1
    else
        TIMEOUT=216000 # 60h for F16D3_K10
    fi
else  # F32
    if [ "$ADAPT_STEPS" -eq 1 ]; then
        TIMEOUT=216000 # 60h for F32D3_K1
    else
        TIMEOUT=252000 # 70h for F32D3_K10 (may timeout but better than sequential)
    fi
fi

echo "============================================"
echo "🧠 EXPERIMENT: F${FEATURES}D${DEPTH}_K${ADAPT_STEPS}_S${SEED}"
echo "Expected time: $(($TIMEOUT / 3600))h"
echo "Started at: $(date)"
echo "============================================"

# Run the experiment
timeout $TIMEOUT python main.py \
    --experiment concept \
    --m mlp \
    --data-type bits \
    --num-concept-features $FEATURES \
    --pcfg-max-depth $DEPTH \
    --adaptation-steps $ADAPT_STEPS \
    --epochs 200 \
    --seed $SEED \
    --save \
    --no_hyper_search \
    --first-order

EXIT_CODE=$?

echo "============================================"
if [ $EXIT_CODE -eq 124 ]; then
    echo "⚠️  TIMEOUT after $(($TIMEOUT / 3600))h - F${FEATURES}D${DEPTH}_K${ADAPT_STEPS}_S${SEED}"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS - F${FEATURES}D${DEPTH}_K${ADAPT_STEPS}_S${SEED} completed at $(date)"
else
    echo "❌ ERROR code $EXIT_CODE - F${FEATURES}D${DEPTH}_K${ADAPT_STEPS}_S${SEED}"
fi
echo "============================================"
