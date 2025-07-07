#!/bin/bash
#SBATCH --job-name=mlp_core_complexity
#SBATCH --output=mlp_core_complexity_%j.out
#SBATCH --error=mlp_core_complexity_%j.err
#SBATCH --time=41:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "MLP GRID COMPLETION - P1_CORE (CORE_COMPLEXITY)"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Priority: Essential complexity scaling evidence"
echo "Experiments: 12, Est. time: 41h"
echo "Configurations: F32_D3, F16_D3"

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
echo "   Timeout: 1.9h, Justification: Feature complexity scaling"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order"
timeout 7000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 1/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 2/12: k1_f16d3_s2"
echo "   Config: F16_D3, K=1, Seed=2"
echo "   Timeout: 1.9h, Justification: Feature complexity scaling"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order"
timeout 7000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 2/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 3/12: k1_f16d3_s3"
echo "   Config: F16_D3, K=1, Seed=3"
echo "   Timeout: 1.9h, Justification: Feature complexity scaling"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order"
timeout 7000 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 3/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 4/12: k10_f16d3_s1"
echo "   Config: F16_D3, K=10, Seed=1"
echo "   Timeout: 2.9h, Justification: Feature complexity scaling"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order"
timeout 10500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 4/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 5/12: k10_f16d3_s2"
echo "   Config: F16_D3, K=10, Seed=2"
echo "   Timeout: 2.9h, Justification: Feature complexity scaling"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order"
timeout 10500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 5/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 6/12: k10_f16d3_s3"
echo "   Config: F16_D3, K=10, Seed=3"
echo "   Timeout: 2.9h, Justification: Feature complexity scaling"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order"
timeout 10500 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 6/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 7/12: k1_f32d3_s1"
echo "   Config: F32_D3, K=1, Seed=1"
echo "   Timeout: 2.4h, Justification: High feature complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order"
timeout 8600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 1 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 7/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 8/12: k1_f32d3_s2"
echo "   Config: F32_D3, K=1, Seed=2"
echo "   Timeout: 2.4h, Justification: High feature complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order"
timeout 8600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 8/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 9/12: k1_f32d3_s3"
echo "   Config: F32_D3, K=1, Seed=3"
echo "   Timeout: 2.4h, Justification: High feature complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order"
timeout 8600 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 9/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 10/12: k10_f32d3_s1"
echo "   Config: F32_D3, K=10, Seed=1"
echo "   Timeout: 3.6h, Justification: High feature complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order"
timeout 12900 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 1 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 10/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 11/12: k10_f32d3_s2"
echo "   Config: F32_D3, K=10, Seed=2"
echo "   Timeout: 3.6h, Justification: High feature complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order"
timeout 12900 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 11/12 at $(date)"
echo "   =========================================="

echo "🧠 EXPERIMENT 12/12: k10_f32d3_s3"
echo "   Config: F32_D3, K=10, Seed=3"
echo "   Timeout: 3.6h, Justification: High feature complexity"
echo "   Command: python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order"
timeout 12900 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "   ✅ Completed experiment 12/12 at $(date)"
echo "   =========================================="

echo "============================================"
echo "🎯 CORE_COMPLEXITY PRIORITY COMPLETED"
echo "Job finished at: $(date)"
echo "Experiments completed: 12"
echo "Statistical coverage: F32_D3, F16_D3"
echo "============================================"
