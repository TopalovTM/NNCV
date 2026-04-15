#!/bin/bash
#SBATCH --job-name=eval_r50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_eval.out
#SBATCH --error=logs/%j_eval.err

set -e

PROJECT_ROOT="$HOME/5LSH0/NNCV/final_assignment"
cd "$PROJECT_ROOT"

mkdir -p logs

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source .venv/bin/activate

python evaluate.py \
  --config configs/peak_deeplabv3plus_r50_metrics.yaml \
  --checkpoint checkpoints/peak-deeplabv3plus-r50-metrics/best_model_loss.pt \
  --save-dir evaluation_outputs/run2_r50_loss \
  --num-samples 8
