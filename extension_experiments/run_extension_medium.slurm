#!/bin/bash
#SBATCH --job-name=extension_medium
#SBATCH --output=extension_medium_%j.out
#SBATCH --error=extension_medium_%j.err
#SBATCH --time=49:00:00
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mg7411@princeton.edu

echo "EXTENSION EXPERIMENTS - MEDIUM PRIORITY"
echo "Host: $(hostname), Working Dir: $(pwd), Date: $(date)"
echo "Total experiments: 16"
echo "Estimated time: 49 hours"

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

echo "🚀 EXPERIMENT 1/16: k1_f8d3_cnn_s0"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: CNN vs MLP comparison"
timeout 7200 python main.py --experiment concept --m cnn --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 1/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 2/16: k1_f8d3_cnn_s1"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: CNN vs MLP comparison"
timeout 7200 python main.py --experiment concept --m cnn --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 2/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 3/16: k10_f8d3_cnn_s0"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: CNN vs MLP comparison"
timeout 10800 python main.py --experiment concept --m cnn --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 3/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 4/16: k10_f8d3_cnn_s1"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: CNN vs MLP comparison"
timeout 10800 python main.py --experiment concept --m cnn --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 4/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 5/16: k1_f16d3_cnn_s0"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: CNN scaling"
timeout 7200 python main.py --experiment concept --m cnn --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 5/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 6/16: k1_f16d3_cnn_s1"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: CNN scaling"
timeout 7200 python main.py --experiment concept --m cnn --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 6/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 7/16: k10_f16d3_cnn_s0"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: CNN scaling"
timeout 10800 python main.py --experiment concept --m cnn --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 7/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 8/16: k10_f16d3_cnn_s1"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: CNN scaling"
timeout 10800 python main.py --experiment concept --m cnn --data-type bits --num-concept-features 16 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 8/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 9/16: k1_f8d3_lstm_s0"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: Sequential processing"
timeout 7200 python main.py --experiment concept --m lstm --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 9/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 10/16: k1_f8d3_lstm_s1"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: Sequential processing"
timeout 7200 python main.py --experiment concept --m lstm --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 10/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 11/16: k10_f8d3_lstm_s0"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: Sequential processing"
timeout 10800 python main.py --experiment concept --m lstm --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 11/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 12/16: k10_f8d3_lstm_s1"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: Sequential processing"
timeout 10800 python main.py --experiment concept --m lstm --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 12/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 13/16: k1_f8d3_transformer_s0"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: Attention mechanism"
timeout 7200 python main.py --experiment concept --m transformer --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 13/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 14/16: k1_f8d3_transformer_s1"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: Attention mechanism"
timeout 7200 python main.py --experiment concept --m transformer --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 1 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 14/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 15/16: k10_f8d3_transformer_s0"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: Attention mechanism"
timeout 10800 python main.py --experiment concept --m transformer --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 0 --save --no_hyper_search --first-order
echo "   Completed experiment 15/16 at $(date)"
echo "   ----------------------------------------"

echo "🚀 EXPERIMENT 16/16: k10_f8d3_transformer_s1"
echo "   Priority: MEDIUM, Batch: architecture_comparison"
echo "   Justification: Attention mechanism"
timeout 10800 python main.py --experiment concept --m transformer --data-type bits --num-concept-features 8 --pcfg-max-depth 3 --adaptation-steps 10 --epochs 100 --seed 1 --save --no_hyper_search --first-order
echo "   Completed experiment 16/16 at $(date)"
echo "   ----------------------------------------"

echo "============================================"
echo "MEDIUM PRIORITY EXPERIMENTS COMPLETED"
echo "Job finished at: $(date)"
echo "Total experiments run: 16"
echo "============================================"
