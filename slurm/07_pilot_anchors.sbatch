#!/bin/bash
#SBATCH --job-name=rank-pilot-anchors
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Pilot (<=500 samples rule): 1 scene x 5 reach task seeds x K=64 candidates, simulator ground truth only. No downloads.
set -uo pipefail
R=/projects/u6xn/rankability
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ) robo_jepa_sha=$(git -C $R/robo_jepa rev-parse HEAD) seed=0"
apptainer exec --nv --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/collect_anchors.py --out $R/data/anchors_pilot_v0 --scene_seeds 0 --task_seeds 0 1 2 3 4 --k 64"
echo "end=$(date -u +%FT%TZ)"
