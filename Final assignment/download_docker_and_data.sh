#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=3:00:00

apptainer pull container.sif docker://cclaess/5lsm0:v1

mkdir -p data

apptainer exec --env-file .env container.sif /bin/bash -lc '
    huggingface-cli download TimJaspersTue/5LSM0 \
        --local-dir ./data \
        --repo-type dataset
'
