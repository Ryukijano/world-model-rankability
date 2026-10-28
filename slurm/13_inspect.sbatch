#!/bin/bash
#SBATCH --job-name=rank-inspect
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
set -uo pipefail
R=/projects/u6xn/rankability; ANCH=${ANCH:-anchors_pilot_v1}
echo "job=${SLURM_JOB_ID} start=$(date -u +%FT%TZ) anchors=$ANCH"
apptainer exec --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R && python3 probe/inspect_anchors.py data/$ANCH data/${ANCH}_montage.png" 2>&1 | grep -vE "WARNING|INFO"
