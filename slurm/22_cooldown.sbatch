#!/bin/bash
#SBATCH --job-name=rank-cool
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Cooldown pilot. Args: <config.json> [--dry_run]. Provenance (config, sha, seed, env) is printed by the script.
set -uo pipefail
R=/projects/u6xn/rankability
echo "job=${SLURM_JOB_ID} start=$(date -u +%FT%TZ) config=$1 sha=$(git -C $R/robo_jepa rev-parse HEAD)"
apptainer exec --nv --bind $R:$R --bind /scratch/u6xn:/scratch/u6xn /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/cooldown_rank.py --config $*"
echo "end=$(date -u +%FT%TZ)"
