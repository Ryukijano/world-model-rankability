#!/bin/bash
#SBATCH --job-name=rank-score-smoke
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Smoke: score pilot_v0 anchors with the 22M final checkpoint (frozen; no training; no downloads).
set -uo pipefail
R=/projects/u6xn/rankability; SZ=${SZ:-22m}; W=${W:-cooldown_22m_150k.pth.tar}; ANCH=${ANCH:-anchors_pilot_v0}
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ) robo_jepa_sha=$(git -C $R/robo_jepa rev-parse HEAD) model=wm-$SZ/$W anchors=$ANCH"
apptainer exec --nv --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/score_anchors.py --model_dir $R/ckpt/wm-$SZ --weights $W --anchors $R/data/$ANCH --out $R/data/scores/${ANCH}__wm-${SZ}__${W%.pth.tar}.npz"
echo "end=$(date -u +%FT%TZ)"
