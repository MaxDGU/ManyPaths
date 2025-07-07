#!/bin/bash
#SBATCH --job-name=mlp_max_complexity
#SBATCH --output=mlp_max_complexity_%j.out
#SBATCH --error=mlp_max_complexity_%j.err
#SBATCH --time=76:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "MLP GRID COMPLETION - P3_MAX (MAX_COMPLEXITY)"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Priority: Maximum complexity validation"
echo "Experiments: 18, Est. time: 76h"
echo "Configurations: F16_D7, F32_D7, F8_D7"

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
echo "   Timeout: 2.4h, Justification: Simple features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
timeout 8600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 1/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 2/18: k1_f8d7_s2"
echo "   Config: F8_D7, K=1, Seed=2"
echo "   Timeout: 2.4h, Justification: Simple features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
timeout 8600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 2/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 3/18: k1_f8d7_s3"
echo "   Config: F8_D7, K=1, Seed=3"
echo "   Timeout: 2.4h, Justification: Simple features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
timeout 8600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 3/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 4/18: k10_f8d7_s1"
echo "   Config: F8_D7, K=10, Seed=1"
echo "   Timeout: 3.6h, Justification: Simple features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
timeout 12900 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 4/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 5/18: k10_f8d7_s2"
echo "   Config: F8_D7, K=10, Seed=2"
echo "   Timeout: 3.6h, Justification: Simple features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
timeout 12900 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 5/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 6/18: k10_f8d7_s3"
echo "   Config: F8_D7, K=10, Seed=3"
echo "   Timeout: 3.6h, Justification: Simple features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
timeout 12900 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 6/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 7/18: k1_f16d7_s1"
echo "   Config: F16_D7, K=1, Seed=1"
echo "   Timeout: 2.6h, Justification: Medium features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
timeout 9400 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 7/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 8/18: k1_f16d7_s2"
echo "   Config: F16_D7, K=1, Seed=2"
echo "   Timeout: 2.6h, Justification: Medium features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
timeout 9400 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 8/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 9/18: k1_f16d7_s3"
echo "   Config: F16_D7, K=1, Seed=3"
echo "   Timeout: 2.6h, Justification: Medium features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
timeout 9400 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 9/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 10/18: k10_f16d7_s1"
echo "   Config: F16_D7, K=10, Seed=1"
echo "   Timeout: 3.9h, Justification: Medium features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
timeout 14100 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 10/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 11/18: k10_f16d7_s2"
echo "   Config: F16_D7, K=10, Seed=2"
echo "   Timeout: 3.9h, Justification: Medium features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
timeout 14100 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 11/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 12/18: k10_f16d7_s3"
echo "   Config: F16_D7, K=10, Seed=3"
echo "   Timeout: 3.9h, Justification: Medium features, max depth"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
timeout 14100 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 12/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 13/18: k1_f32d7_s1"
echo "   Config: F32_D7, K=1, Seed=1"
echo "   Timeout: 3.1h, Justification: Maximum complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
timeout 11000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 13/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 14/18: k1_f32d7_s2"
echo "   Config: F32_D7, K=1, Seed=2"
echo "   Timeout: 3.1h, Justification: Maximum complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
timeout 11000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 14/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 15/18: k1_f32d7_s3"
echo "   Config: F32_D7, K=1, Seed=3"
echo "   Timeout: 3.1h, Justification: Maximum complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
timeout 11000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 15/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 16/18: k10_f32d7_s1"
echo "   Config: F32_D7, K=10, Seed=1"
echo "   Timeout: 4.6h, Justification: Maximum complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order"
timeout 16500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 16/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 17/18: k10_f32d7_s2"
echo "   Config: F32_D7, K=10, Seed=2"
echo "   Timeout: 4.6h, Justification: Maximum complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order"
timeout 16500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 17/18 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 18/18: k10_f32d7_s3"
echo "   Config: F32_D7, K=10, Seed=3"
echo "   Timeout: 4.6h, Justification: Maximum complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order"
timeout 16500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 18/18 at $(date)"
echo "   =========================================="

echo "============================================"
echo "🎯 MAX_COMPLEXITY PRIORITY COMPLETED"
echo "Job finished at: $(date)"
echo "Experiments completed: 18"
echo "Statistical coverage: F16_D7, F32_D7, F8_D7"
echo "============================================"
