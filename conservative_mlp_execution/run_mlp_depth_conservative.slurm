#!/bin/bash
#SBATCH --job-name=mlp_depth_grid
#SBATCH --output=mlp_depth_grid_%j.out
#SBATCH --error=mlp_depth_grid_%j.err
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

echo "MLP GRID COMPLETION - P2_DEPTH"
echo "Conservative timing model (sublinear scaling)"
echo "Host: $(hostname), Date: $(date)"
echo "Experiments: 18, Allocated: 48h, Est: 711.2h"
echo "Configs: F8_D5, F16_D5, F32_D5"

module load anaconda3/2023.3
module load cudatoolkit/11.8
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname)"
echo "============================================"


echo "Experiment 1/18: F8_D5_K1_S1"
echo "  Est: 18.0h, Timeout: 64858s"
echo "  Started: $(date)"

timeout 64858 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 18.0h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 2/18: F8_D5_K1_S2"
echo "  Est: 18.0h, Timeout: 64858s"
echo "  Started: $(date)"

timeout 64858 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 18.0h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 3/18: F8_D5_K1_S3"
echo "  Est: 18.0h, Timeout: 64858s"
echo "  Started: $(date)"

timeout 64858 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 18.0h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 4/18: F8_D5_K10_S1"
echo "  Est: 27.0h, Timeout: 97287s"
echo "  Started: $(date)"

timeout 97287 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 27.0h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 5/18: F8_D5_K10_S2"
echo "  Est: 27.0h, Timeout: 97287s"
echo "  Started: $(date)"

timeout 97287 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 27.0h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 6/18: F8_D5_K10_S3"
echo "  Est: 27.0h, Timeout: 97287s"
echo "  Started: $(date)"

timeout 97287 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 27.0h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 7/18: F16_D5_K1_S1"
echo "  Est: 29.3h, Timeout: 105363s"
echo "  Started: $(date)"

timeout 105363 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 29.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 8/18: F16_D5_K1_S2"
echo "  Est: 29.3h, Timeout: 105363s"
echo "  Started: $(date)"

timeout 105363 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 29.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 9/18: F16_D5_K1_S3"
echo "  Est: 29.3h, Timeout: 105363s"
echo "  Started: $(date)"

timeout 105363 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 29.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 10/18: F16_D5_K10_S1"
echo "  Est: 43.9h, Timeout: 158044s"
echo "  Started: $(date)"

timeout 158044 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 43.9h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 11/18: F16_D5_K10_S2"
echo "  Est: 43.9h, Timeout: 158044s"
echo "  Started: $(date)"

timeout 158044 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 43.9h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 12/18: F16_D5_K10_S3"
echo "  Est: 43.9h, Timeout: 158044s"
echo "  Started: $(date)"

timeout 158044 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 43.9h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 13/18: F32_D5_K1_S1"
echo "  Est: 47.5h, Timeout: 171162s"
echo "  Started: $(date)"

timeout 171162 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 47.5h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 14/18: F32_D5_K1_S2"
echo "  Est: 47.5h, Timeout: 171162s"
echo "  Started: $(date)"

timeout 171162 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 47.5h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 15/18: F32_D5_K1_S3"
echo "  Est: 47.5h, Timeout: 171162s"
echo "  Started: $(date)"

timeout 171162 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 47.5h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 16/18: F32_D5_K10_S1"
echo "  Est: 71.3h, Timeout: 256744s"
echo "  Started: $(date)"

timeout 256744 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 71.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 17/18: F32_D5_K10_S2"
echo "  Est: 71.3h, Timeout: 256744s"
echo "  Started: $(date)"

timeout 256744 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 71.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 18/18: F32_D5_K10_S3"
echo "  Est: 71.3h, Timeout: 256744s"
echo "  Started: $(date)"

timeout 256744 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 5 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 71.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "============================================"
echo "P2_DEPTH COMPLETED at $(date)"
echo "Time allocation: 48h, Estimated: 711.2h"
echo "============================================"
