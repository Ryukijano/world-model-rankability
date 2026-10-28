#!/bin/bash
#SBATCH --job-name=rank-fix-env
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
set -uo pipefail
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ)"
apptainer exec --nv --bind /projects/u6xn/rankability:/projects/u6xn/rankability /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash /projects/u6xn/rankability/slurm/fix_env_inner.sh
