#!/bin/bash
#SBATCH --job-name=ood_coco_r101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_ood_coco.out
#SBATCH --error=logs/%j_ood_coco.err

set -e

PROJECT_ROOT="$HOME/5LSH0/NNCV/final_assignment"
cd "$PROJECT_ROOT"

mkdir -p logs

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source .venv/bin/activate

python ood_score_folder.py \
  --checkpoint checkpoints/peak-deeplabv3plus-r101-tuned/best_model_dice.pt \
  --encoder-name resnet101 \
  --image-dir ../data/coco_ood/val2017 \
  --save-dir ood_outputs/coco_r101 \
  --num-samples 0
