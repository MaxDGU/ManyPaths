#!/bin/bash
#SBATCH --job-name=mlp_core_complexity
#SBATCH --output=mlp_core_complexity_realistic_%j.out
#SBATCH --error=mlp_core_complexity_realistic_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "🚨 CAMERA-READY URGENT"
echo "MLP GRID COMPLETION - P1_CORE (CORE_COMPLEXITY)"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Priority: Essential complexity scaling evidence"
echo "Experiments: 12, Allocated time: 48h"
echo "Realistic estimate: 1053.0h (based on F8_D3 performance)"
echo "Configurations: F16_D3, F32_D3"

# Load modules
module load anaconda3/2023.3
module load cudatoolkit/11.8

# Activate conda environment
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"


echo "🧠 EXPERIMENT 1/12: k1_f16d3_s1"
echo "   Config: F16_D3, K=1, Seed=1"
echo "   Est. time: 45.0h (F8_D3 calibrated)"
echo "   Timeout: 162000s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 162000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 45.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 2/12: k1_f16d3_s2"
echo "   Config: F16_D3, K=1, Seed=2"
echo "   Est. time: 45.0h (F8_D3 calibrated)"
echo "   Timeout: 162000s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 162000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 45.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 3/12: k1_f16d3_s3"
echo "   Config: F16_D3, K=1, Seed=3"
echo "   Est. time: 45.0h (F8_D3 calibrated)"
echo "   Timeout: 162000s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 162000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 45.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 4/12: k10_f16d3_s1"
echo "   Config: F16_D3, K=10, Seed=1"
echo "   Est. time: 72.0h (F8_D3 calibrated)"
echo "   Timeout: 259200s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 259200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 72.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 5/12: k10_f16d3_s2"
echo "   Config: F16_D3, K=10, Seed=2"
echo "   Est. time: 72.0h (F8_D3 calibrated)"
echo "   Timeout: 259200s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 259200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 72.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 6/12: k10_f16d3_s3"
echo "   Config: F16_D3, K=10, Seed=3"
echo "   Est. time: 72.0h (F8_D3 calibrated)"
echo "   Timeout: 259200s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 259200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 72.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 7/12: k1_f32d3_s1"
echo "   Config: F32_D3, K=1, Seed=1"
echo "   Est. time: 90.0h (F8_D3 calibrated)"
echo "   Timeout: 324000s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 324000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 90.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 8/12: k1_f32d3_s2"
echo "   Config: F32_D3, K=1, Seed=2"
echo "   Est. time: 90.0h (F8_D3 calibrated)"
echo "   Timeout: 324000s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 324000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 90.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 9/12: k1_f32d3_s3"
echo "   Config: F32_D3, K=1, Seed=3"
echo "   Est. time: 90.0h (F8_D3 calibrated)"
echo "   Timeout: 324000s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 324000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 90.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 10/12: k10_f32d3_s1"
echo "   Config: F32_D3, K=10, Seed=1"
echo "   Est. time: 144.0h (F8_D3 calibrated)"
echo "   Timeout: 518400s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 518400 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 144.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 11/12: k10_f32d3_s2"
echo "   Config: F32_D3, K=10, Seed=2"
echo "   Est. time: 144.0h (F8_D3 calibrated)"
echo "   Timeout: 518400s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 518400 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 144.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 12/12: k10_f32d3_s3"
echo "   Config: F32_D3, K=10, Seed=3"
echo "   Est. time: 144.0h (F8_D3 calibrated)"
echo "   Timeout: 518400s, Epochs: 200"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 518400 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 144.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "============================================"
echo "🎯 CORE_COMPLEXITY PRIORITY COMPLETED"
echo "Job finished at: $(date)"
echo "Total experiments: 12"
echo "Realistic time used vs allocated: 1053.0h / 48h"
echo "Statistical coverage: F16_D3, F32_D3"
echo "============================================"
