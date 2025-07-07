#!/bin/bash
#SBATCH --job-name=extension_urgent
#SBATCH --output=extension_urgent_%j.out
#SBATCH --error=extension_urgent_%j.err
#SBATCH --time=37:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "EXTENSION EXPERIMENTS - URGENT PRIORITY"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Total experiments: 12"
echo "Estimated time: 37 hours"

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

# Run experiments

echo "🚀 EXPERIMENT 1/12: k1_f16_d3_s2"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: Core complexity scaling"
timeout 7200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "   Completed experiment 1/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 2/12: k1_f16_d3_s3"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: Core complexity scaling"
timeout 7200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "   Completed experiment 2/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 3/12: k1_f16_d3_s4"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: Core complexity scaling"
timeout 7200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 4 --save --no_hyper_search --first-order
echo "   Completed experiment 3/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 4/12: k10_f16_d3_s2"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: Core complexity scaling"
timeout 10800 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "   Completed experiment 4/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 5/12: k10_f16_d3_s3"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: Core complexity scaling"
timeout 10800 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "   Completed experiment 5/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 6/12: k10_f16_d3_s4"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: Core complexity scaling"
timeout 10800 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 4 --save --no_hyper_search --first-order
echo "   Completed experiment 6/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 7/12: k1_f32_d3_s2"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: High complexity baseline"
timeout 7200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "   Completed experiment 7/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 8/12: k1_f32_d3_s3"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: High complexity baseline"
timeout 7200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "   Completed experiment 8/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 9/12: k1_f32_d3_s4"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: High complexity baseline"
timeout 7200 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 200 --seed 4 --save --no_hyper_search --first-order
echo "   Completed experiment 9/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 10/12: k10_f32_d3_s2"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: High complexity baseline"
timeout 10800 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 2 --save --no_hyper_search --first-order
echo "   Completed experiment 10/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 11/12: k10_f32_d3_s3"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: High complexity baseline"
timeout 10800 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 3 --save --no_hyper_search --first-order
echo "   Completed experiment 11/12 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 12/12: k10_f32_d3_s4"
echo "   Priority: URGENT, Batch: core_complexity"
echo "   Justification: High complexity baseline"
timeout 10800 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 200 --seed 4 --save --no_hyper_search --first-order
echo "   Completed experiment 12/12 at $(date)"
echo "   ----------------------------------------"

echo "============================================"
echo "URGENT PRIORITY EXPERIMENTS COMPLETED"
echo "Job finished at: $(date)"
echo "Total experiments run: 12"
echo "============================================"
