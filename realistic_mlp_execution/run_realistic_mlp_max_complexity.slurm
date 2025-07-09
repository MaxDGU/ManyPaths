#!/bin/bash
#SBATCH --job-name=mlp_max_complexity
#SBATCH --output=mlp_max_complexity_realistic_%j.out
#SBATCH --error=mlp_max_complexity_realistic_%j.err
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
echo "MLP GRID COMPLETION - P3_MAX (MAX_COMPLEXITY)"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Priority: Maximum complexity validation"
echo "Experiments: 18, Allocated time: 48h"
echo "Realistic estimate: 2149.9h (based on F8_D3 performance)"
echo "Configurations: F8_D7, F32_D7, F16_D7"

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


echo "🧠 EXPERIMENT 1/18: k1_f8d7_s1"
echo "   Config: F8_D7, K=1, Seed=1"
echo "   Est. time: 39.4h (F8_D3 calibrated)"
echo "   Timeout: 141750s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 141750 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 39.4h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 2/18: k1_f8d7_s2"
echo "   Config: F8_D7, K=1, Seed=2"
echo "   Est. time: 39.4h (F8_D3 calibrated)"
echo "   Timeout: 141750s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 141750 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 39.4h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 3/18: k1_f8d7_s3"
echo "   Config: F8_D7, K=1, Seed=3"
echo "   Est. time: 39.4h (F8_D3 calibrated)"
echo "   Timeout: 141750s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 141750 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 39.4h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 4/18: k10_f8d7_s1"
echo "   Config: F8_D7, K=10, Seed=1"
echo "   Est. time: 63.0h (F8_D3 calibrated)"
echo "   Timeout: 226800s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 226800 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 63.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 5/18: k10_f8d7_s2"
echo "   Config: F8_D7, K=10, Seed=2"
echo "   Est. time: 63.0h (F8_D3 calibrated)"
echo "   Timeout: 226800s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 226800 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 63.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 6/18: k10_f8d7_s3"
echo "   Config: F8_D7, K=10, Seed=3"
echo "   Est. time: 63.0h (F8_D3 calibrated)"
echo "   Timeout: 226800s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 226800 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 63.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 7/18: k1_f16d7_s1"
echo "   Config: F16_D7, K=1, Seed=1"
echo "   Est. time: 78.8h (F8_D3 calibrated)"
echo "   Timeout: 283500s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 283500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 78.8h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 8/18: k1_f16d7_s2"
echo "   Config: F16_D7, K=1, Seed=2"
echo "   Est. time: 78.8h (F8_D3 calibrated)"
echo "   Timeout: 283500s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 283500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 78.8h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 9/18: k1_f16d7_s3"
echo "   Config: F16_D7, K=1, Seed=3"
echo "   Est. time: 78.8h (F8_D3 calibrated)"
echo "   Timeout: 283500s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 283500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 78.8h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 10/18: k10_f16d7_s1"
echo "   Config: F16_D7, K=10, Seed=1"
echo "   Est. time: 126.0h (F8_D3 calibrated)"
echo "   Timeout: 453600s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 453600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 126.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 11/18: k10_f16d7_s2"
echo "   Config: F16_D7, K=10, Seed=2"
echo "   Est. time: 126.0h (F8_D3 calibrated)"
echo "   Timeout: 453600s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 453600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 126.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 12/18: k10_f16d7_s3"
echo "   Config: F16_D7, K=10, Seed=3"
echo "   Est. time: 126.0h (F8_D3 calibrated)"
echo "   Timeout: 453600s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 453600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 126.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 13/18: k1_f32d7_s1"
echo "   Config: F32_D7, K=1, Seed=1"
echo "   Est. time: 157.5h (F8_D3 calibrated)"
echo "   Timeout: 567000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 567000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 157.5h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 14/18: k1_f32d7_s2"
echo "   Config: F32_D7, K=1, Seed=2"
echo "   Est. time: 157.5h (F8_D3 calibrated)"
echo "   Timeout: 567000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 567000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 157.5h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 15/18: k1_f32d7_s3"
echo "   Config: F32_D7, K=1, Seed=3"
echo "   Est. time: 157.5h (F8_D3 calibrated)"
echo "   Timeout: 567000s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 567000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 157.5h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 16/18: k10_f32d7_s1"
echo "   Config: F32_D7, K=10, Seed=1"
echo "   Est. time: 252.0h (F8_D3 calibrated)"
echo "   Timeout: 907200s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 907200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 252.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 17/18: k10_f32d7_s2"
echo "   Config: F32_D7, K=10, Seed=2"
echo "   Est. time: 252.0h (F8_D3 calibrated)"
echo "   Timeout: 907200s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 907200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 252.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "🧠 EXPERIMENT 18/18: k10_f32d7_s3"
echo "   Config: F32_D7, K=10, Seed=3"
echo "   Est. time: 252.0h (F8_D3 calibrated)"
echo "   Timeout: 907200s, Epochs: 150"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
echo "   Started at: $(date)"

timeout 907200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "   ⚠️  TIMEOUT after 252.0h - experiment may need more time"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ SUCCESS completed at $(date)"
else
    echo "   ❌ ERROR with exit code $EXIT_CODE"
fi
echo "   =========================================="

echo "============================================"
echo "🎯 MAX_COMPLEXITY PRIORITY COMPLETED"
echo "Job finished at: $(date)"
echo "Total experiments: 18"
echo "Realistic time used vs allocated: 2149.9h / 48h"
echo "Statistical coverage: F8_D7, F32_D7, F16_D7"
echo "============================================"
