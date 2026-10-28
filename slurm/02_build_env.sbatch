#!/bin/bash
#SBATCH --job-name=rank-build-env
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Build a venv (system-site-packages over the NGC container torch) with robo_jepa deps + robosuite/robocasa git pins.
set -uo pipefail
R=/projects/u6xn/rankability; SIF=/projects/u6xn/sif-images/pytorch_26.09-py3.sif
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ) robo_jepa_sha=$(git -C $R/robo_jepa rev-parse HEAD)"
apptainer exec --nv --bind $R:$R ${SIF} bash $R/slurm/build_env_inner.sh
echo "end=$(date -u +%FT%TZ)"
