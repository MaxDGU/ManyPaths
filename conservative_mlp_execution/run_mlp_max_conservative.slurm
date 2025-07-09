#!/bin/bash
#SBATCH --job-name=mlp_max_grid
#SBATCH --output=mlp_max_grid_%j.out
#SBATCH --error=mlp_max_grid_%j.err
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

echo "MLP GRID COMPLETION - P3_MAX"
echo "Conservative timing model (sublinear scaling)"
echo "Host: $(hostname), Date: $(date)"
echo "Experiments: 18, Allocated: 48h, Est: 900.1h"
echo "Configs: F16_D7, F8_D7, F32_D7"

module load anaconda3/2023.3
module load cudatoolkit/11.8
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=0

echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname)"
echo "============================================"


echo "Experiment 1/18: F8_D7_K1_S1"
echo "  Est: 22.8h, Timeout: 82083s"
echo "  Started: $(date)"

timeout 82083 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 22.8h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 2/18: F8_D7_K1_S2"
echo "  Est: 22.8h, Timeout: 82083s"
echo "  Started: $(date)"

timeout 82083 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 22.8h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 3/18: F8_D7_K1_S3"
echo "  Est: 22.8h, Timeout: 82083s"
echo "  Started: $(date)"

timeout 82083 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 22.8h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 4/18: F8_D7_K10_S1"
echo "  Est: 34.2h, Timeout: 123125s"
echo "  Started: $(date)"

timeout 123125 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 34.2h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 5/18: F8_D7_K10_S2"
echo "  Est: 34.2h, Timeout: 123125s"
echo "  Started: $(date)"

timeout 123125 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 34.2h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 6/18: F8_D7_K10_S3"
echo "  Est: 34.2h, Timeout: 123125s"
echo "  Started: $(date)"

timeout 123125 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 8 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 34.2h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 7/18: F16_D7_K1_S1"
echo "  Est: 37.0h, Timeout: 133345s"
echo "  Started: $(date)"

timeout 133345 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 37.0h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 8/18: F16_D7_K1_S2"
echo "  Est: 37.0h, Timeout: 133345s"
echo "  Started: $(date)"

timeout 133345 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 37.0h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 9/18: F16_D7_K1_S3"
echo "  Est: 37.0h, Timeout: 133345s"
echo "  Started: $(date)"

timeout 133345 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 37.0h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 10/18: F16_D7_K10_S1"
echo "  Est: 55.6h, Timeout: 200017s"
echo "  Started: $(date)"

timeout 200017 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 55.6h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 11/18: F16_D7_K10_S2"
echo "  Est: 55.6h, Timeout: 200017s"
echo "  Started: $(date)"

timeout 200017 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 55.6h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 12/18: F16_D7_K10_S3"
echo "  Est: 55.6h, Timeout: 200017s"
echo "  Started: $(date)"

timeout 200017 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 16 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 55.6h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 13/18: F32_D7_K1_S1"
echo "  Est: 60.2h, Timeout: 216620s"
echo "  Started: $(date)"

timeout 216620 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 60.2h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 14/18: F32_D7_K1_S2"
echo "  Est: 60.2h, Timeout: 216620s"
echo "  Started: $(date)"

timeout 216620 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 60.2h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 15/18: F32_D7_K1_S3"
echo "  Est: 60.2h, Timeout: 216620s"
echo "  Started: $(date)"

timeout 216620 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 1 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 60.2h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 16/18: F32_D7_K10_S1"
echo "  Est: 90.3h, Timeout: 324930s"
echo "  Started: $(date)"

timeout 324930 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 1 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 90.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 17/18: F32_D7_K10_S2"
echo "  Est: 90.3h, Timeout: 324930s"
echo "  Started: $(date)"

timeout 324930 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 2 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 90.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "Experiment 18/18: F32_D7_K10_S3"
echo "  Est: 90.3h, Timeout: 324930s"
echo "  Started: $(date)"

timeout 324930 python main.py --experiment concept --m mlp --data-type bits --num-concept-features 32 --pcfg-max-depth 7 --adaptation-steps 10 --epochs 150 --seed 3 --save --no_hyper_search --first-order
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "  TIMEOUT after 90.3h"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "  SUCCESS at $(date)"
else
    echo "  ERROR code $EXIT_CODE"
fi
echo "  ------------------"

echo "============================================"
echo "P3_MAX COMPLETED at $(date)"
echo "Time allocation: 48h, Estimated: 900.1h"
echo "============================================"
