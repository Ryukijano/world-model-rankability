#!/bin/bash
#SBATCH --job-name=rank-bias
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
set -uo pipefail
R=/projects/u6xn/rankability; A=anchors_pilot_v3
apptainer exec --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R && for f in data/scores_hist/${A}__wm-22m__cooldown_22m_150k__h-last__s-init_pose.npz data/scores_hist/${A}__wm-1b__cooldown_1b_300k__h-last__s-init_pose.npz; do python3 probe/bias_check.py \$f data/$A; done" 2>&1 | grep -vE "WARNING|INFO"
