#!/bin/bash
#SBATCH --job-name=download_cityscapes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j_download.out
#SBATCH --error=logs/%j_download.err

set -e

PROJECT_ROOT="$HOME/5LSH0/NNCV"
cd "$PROJECT_ROOT"

mkdir -p logs
mkdir -p data

if [ ! -f container.sif ]; then
    apptainer pull container.sif docker://cclaess/5lsm0:v1
fi

apptainer exec --env-file final_assignment/.env container.sif /bin/bash -lc '
    huggingface-cli download TimJaspersTue/5LSM0 \
        --local-dir ./data \
        --repo-type dataset
'
