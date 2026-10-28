#!/bin/bash
#SBATCH --job-name=rank-analyze
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Metrics table over data/scores/<ANCH>__*.npz (login python is 3.6 without scipy, so this runs in the container).
set -uo pipefail
R=/projects/u6xn/rankability; ANCH=${ANCH:-anchors_pilot_v0}; DIR=${DIR:-scores}
echo "job=${SLURM_JOB_ID} start=$(date -u +%FT%TZ) anchors=$ANCH"
apptainer exec --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R && python3 probe/analyze.py data/${DIR}/${ANCH}__*.npz" 2>&1 | grep -vE "WARNING|INFO"
