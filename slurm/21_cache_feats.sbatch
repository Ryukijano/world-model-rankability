#!/bin/bash
#SBATCH --job-name=rank-feats
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Frozen-encoder feature cache for the cooldown pilot (inference only, no network). Args: <anchors_dir> <out_dir> <shard_i> <shard_n>
set -uo pipefail
R=/projects/u6xn/rankability
echo "job=${SLURM_JOB_ID} start=$(date -u +%FT%TZ) sha=$(git -C $R/robo_jepa rev-parse HEAD) args=$*"
apptainer exec --nv --bind $R:$R --bind /scratch/u6xn:/scratch/u6xn /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/cache_feats.py --anchors $1 --out $2 --shard $3 $4"
echo "end=$(date -u +%FT%TZ)"
