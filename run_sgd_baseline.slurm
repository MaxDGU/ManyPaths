#!/bin/bash
#SBATCH --job-name=sgd_baseline
#SBATCH --output=sgd_baseline_%j.out
#SBATCH --error=sgd_baseline_%j.err
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

echo "🚀 SGD BASELINE EXPERIMENTS"
echo "Running vanilla SGD baseline for concept learning tasks"
echo "Host: $(hostname), Date: $(date)"

module load anaconda3/2023.3
module load cudatoolkit/11.8
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname)"
echo "============================================"

# F8D3 Cache
echo "🧠 STARTING F8D3 CACHE EXPERIMENTS"
echo "F8D3 Cache - Est: 2h"
timeout 7200 python main_baseline_sgd.py \
  --cache data/concept_cache/pcfg_tasks_f8_d3_s2p3n_q5p5n_t10000.pt \
  --num-concept-features 8 --pcfg-max-depth 3 \
  --epochs 32 --lr 0.01 --momentum 0.9 \
  --seeds 1 2 3 --save-checkpoints --verbose
echo "F8D3 Cache completed at $(date)"

# F8D5 Cache
echo "🧠 STARTING F8D5 CACHE EXPERIMENTS"
echo "F8D5 Cache - Est: 2h"
timeout 7200 python main_baseline_sgd.py \
  --cache data/concept_cache/pcfg_tasks_f8_d5_s2p3n_q5p5n_t10000.pt \
  --num-concept-features 8 --pcfg-max-depth 5 \
  --epochs 32 --lr 0.01 --momentum 0.9 \
  --seeds 1 2 3 --save-checkpoints --verbose
echo "F8D5 Cache completed at $(date)"

# F32D3 Cache
echo "🧠 STARTING F32D3 CACHE EXPERIMENTS"
echo "F32D3 Cache - Est: 6h"
timeout 21600 python main_baseline_sgd.py \
  --cache data/concept_cache/pcfg_tasks_f32_d3_s2p3n_q5p5n_t10000.pt \
  --num-concept-features 32 --pcfg-max-depth 3 \
  --epochs 32 --lr 0.01 --momentum 0.9 \
  --seeds 1 2 3 --save-checkpoints --verbose
echo "F32D3 Cache completed at $(date)"

echo "============================================"
echo "🎉 SGD BASELINE EXPERIMENTS COMPLETED"
echo "Completed at: $(date)"
echo "============================================" 