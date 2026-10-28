#!/bin/bash
#SBATCH --job-name=rank-flow
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Flow-matching / deterministic predictor in frozen V-JEPA space on cached features (no network). Args: <config.json> [--dry_run|--eval_only]
# Resumable: resubmit the same config until the log prints CHECKPOINT .../predictor_final.pt.
set -uo pipefail
R=/projects/u6xn/rankability; case "$1" in /*) CFG=$1;; *) CFG=$R/$1;; esac; shift  # keep the /projects logical path (bound in the container)
echo "job=${SLURM_JOB_ID} start=$(date -u +%FT%TZ) robo_jepa_sha=$(git -C $R/robo_jepa rev-parse HEAD) config=$CFG config_sha=$(sha256sum $CFG | cut -c1-16) script_sha=$(sha256sum $R/probe/flow_predictor.py | cut -c1-16) args=$*"
apptainer exec --nv --bind $R:$R --bind /scratch/u6xn:/scratch/u6xn /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/flow_predictor.py --config $CFG $*"
echo "end=$(date -u +%FT%TZ)"
