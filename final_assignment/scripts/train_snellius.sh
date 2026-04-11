#!/bin/bash
#SBATCH --job-name=cityscapes_dlvp_r50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -e

mkdir -p logs

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python train.py --config configs/peak_deeplabv3plus_r50.yaml