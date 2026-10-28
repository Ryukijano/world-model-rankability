#!/bin/bash
#SBATCH --job-name=rank-scaling
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
set -uo pipefail
R=/projects/u6xn/rankability; ANCH=${ANCH:-anchors_pilot_v2}; DIR=${DIR:-scores}; TAG=${TAG:-}
echo "job=${SLURM_JOB_ID} start=$(date -u +%FT%TZ) anchors=$ANCH dir=$DIR"
apptainer exec --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R && mkdir -p results && python3 probe/scaling_plot.py data/$DIR $ANCH results/scaling_${ANCH}${TAG}" 2>&1 | grep -vE "WARNING|INFO"
