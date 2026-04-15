#!/bin/bash
#SBATCH --job-name=cityscapes_r101_tuned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -e

PROJECT_ROOT="$HOME/5LSH0/NNCV/final_assignment"
cd "$PROJECT_ROOT"

mkdir -p logs

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source .venv/bin/activate

set -a
source .env
set +a

python train.py --config configs/peak_deeplabv3plus_r101_tuned.yaml