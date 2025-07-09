#!/bin/bash
#SBATCH --job-name=mlp_depth_scaling
#SBATCH --output=mlp_depth_scaling_realistic_%j.out
#SBATCH --error=mlp_depth_scaling_realistic_%j.err
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

echo "📊 ANALYSIS EXTENSION"
echo "MLP GRID COMPLETION - P2_DEPTH (DEPTH_SCALING)"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Priority: Logical complexity (depth) scaling"
echo "Experiments: 18, Allocated time: 48h"
echo "Realistic estimate: 1535.6h (based on F8_D3 performance)"
echo "Configurations: F8_D5, F16_D5, F32_D5"

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


echo "🧠 EXPERIMENT 1/18: k1_f8d5_s1"
echo "   Config: F8_D5, K=1, Seed=1"
echo "   Est. time: 28.1h (F8_D3 calibrated)"
echo "   Timeout: 101250s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 101250 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 28.1h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 2/18: k1_f8d5_s2"
echo "   Config: F8_D5, K=1, Seed=2"
echo "   Est. time: 28.1h (F8_D3 calibrated)"
echo "   Timeout: 101250s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 101250 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 28.1h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 3/18: k1_f8d5_s3"
echo "   Config: F8_D5, K=1, Seed=3"
echo "   Est. time: 28.1h (F8_D3 calibrated)"
echo "   Timeout: 101250s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 101250 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 28.1h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 4/18: k10_f8d5_s1"
echo "   Config: F8_D5, K=10, Seed=1"
echo "   Est. time: 45.0h (F8_D3 calibrated)"
echo "   Timeout: 162000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 162000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 45.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 5/18: k10_f8d5_s2"
echo "   Config: F8_D5, K=10, Seed=2"
echo "   Est. time: 45.0h (F8_D3 calibrated)"
echo "   Timeout: 162000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 162000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 45.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 6/18: k10_f8d5_s3"
echo "   Config: F8_D5, K=10, Seed=3"
echo "   Est. time: 45.0h (F8_D3 calibrated)"
echo "   Timeout: 162000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 162000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 45.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 7/18: k1_f16d5_s1"
echo "   Config: F16_D5, K=1, Seed=1"
echo "   Est. time: 56.2h (F8_D3 calibrated)"
echo "   Timeout: 202500s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 202500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 56.2h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 8/18: k1_f16d5_s2"
echo "   Config: F16_D5, K=1, Seed=2"
echo "   Est. time: 56.2h (F8_D3 calibrated)"
echo "   Timeout: 202500s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 202500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 56.2h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 9/18: k1_f16d5_s3"
echo "   Config: F16_D5, K=1, Seed=3"
echo "   Est. time: 56.2h (F8_D3 calibrated)"
echo "   Timeout: 202500s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 202500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 56.2h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 10/18: k10_f16d5_s1"
echo "   Config: F16_D5, K=10, Seed=1"
echo "   Est. time: 90.0h (F8_D3 calibrated)"
echo "   Timeout: 324000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 324000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 90.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 11/18: k10_f16d5_s2"
echo "   Config: F16_D5, K=10, Seed=2"
echo "   Est. time: 90.0h (F8_D3 calibrated)"
echo "   Timeout: 324000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 324000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 90.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 12/18: k10_f16d5_s3"
echo "   Config: F16_D5, K=10, Seed=3"
echo "   Est. time: 90.0h (F8_D3 calibrated)"
echo "   Timeout: 324000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 324000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 90.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 13/18: k1_f32d5_s1"
echo "   Config: F32_D5, K=1, Seed=1"
echo "   Est. time: 112.5h (F8_D3 calibrated)"
echo "   Timeout: 405000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 405000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 112.5h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 14/18: k1_f32d5_s2"
echo "   Config: F32_D5, K=1, Seed=2"
echo "   Est. time: 112.5h (F8_D3 calibrated)"
echo "   Timeout: 405000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 405000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 112.5h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 15/18: k1_f32d5_s3"
echo "   Config: F32_D5, K=1, Seed=3"
echo "   Est. time: 112.5h (F8_D3 calibrated)"
echo "   Timeout: 405000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 405000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 112.5h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 16/18: k10_f32d5_s1"
echo "   Config: F32_D5, K=10, Seed=1"
echo "   Est. time: 180.0h (F8_D3 calibrated)"
echo "   Timeout: 648000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 648000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 180.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 17/18: k10_f32d5_s2"
echo "   Config: F32_D5, K=10, Seed=2"
echo "   Est. time: 180.0h (F8_D3 calibrated)"
echo "   Timeout: 648000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 648000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 180.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 18/18: k10_f32d5_s3"
echo "   Config: F32_D5, K=10, Seed=3"
echo "   Est. time: 180.0h (F8_D3 calibrated)"
echo "   Timeout: 648000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 648000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 180.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "============================================"
echo "🎯 DEPTH_SCALING PRIORITY COMPLETED"
echo "Job finished at: $(date)"
echo "Total experiments: 18"
echo "Realistic time used vs allocated: 1535.6h / 48h"
echo "Statistical coverage: F8_D5, F16_D5, F32_D5"
echo "============================================"
