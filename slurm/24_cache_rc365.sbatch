#!/bin/bash
#SBATCH --job-name=rank-rc365-feats
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Frozen-encoder feature cache for RoboCasa365 pretrain-atomic demos (inference only, no network, data pre-staged by 23_fetch_rc365).
# Args: <out_dir> <n_episodes> <shard_i> <shard_n> [extra cache_rc365.py args]. Resumable; resubmit until ALL_CACHED.
set -uo pipefail
R=/projects/u6xn/rankability; OUT=$1; N=$2; SI=$3; SN=$4; shift 4
echo "job=${SLURM_JOB_ID} start=$(date -u +%FT%TZ) sha=$(git -C $R/robo_jepa rev-parse HEAD) script_sha=$(sha256sum $R/probe/cache_rc365.py | cut -c1-16) args=$OUT $N $SI $SN $*"
apptainer exec --nv --bind $R:$R --bind /scratch/u6xn:/scratch/u6xn /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/cache_rc365.py --out $OUT --n_episodes $N --shard $SI $SN $*"
echo "end=$(date -u +%FT%TZ)"
