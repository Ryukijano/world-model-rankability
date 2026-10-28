#!/bin/bash
#SBATCH --job-name=droid-action-id
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
set -euo pipefail
R=/projects/u6xn/rankability; DATA=/projects/u6xn/datasets/droid-raw-1.0.1-robojepa; SIF=/projects/u6xn/sif-images/pytorch_26.09-py3.sif
OURS=$R/repro22m/data/models-ours
echo "job=${SLURM_JOB_ID} host=$(hostname) utc=$(date -u +%FT%TZ)"
apptainer exec --nv --bind "$R:$R,/projects/u6xn/datasets:/projects/u6xn/datasets" "$SIF" bash -c "
  set -euo pipefail; source $R/venv/bin/activate; export PYTHONPATH=$R/robo_jepa; cd $R/robo_jepa
  python $R/probe/eval_droid_action_id_v2.py --data $DATA --encoder $R/ckpt/encoder/vjepa2_1_vitG_384.pt \
    --out-dir $R/results/droid_action_id_v2 --n-clips 256 \
    --ckpts ours_22m_10k=$OURS:22m:$OURS/wm-22m/cooldown_22m_10k.pth.tar \
            meta_22m_10k=$R/ckpt:22m:$R/ckpt/wm-22m/cooldown_22m_10k.pth.tar \
            meta_22m_150k=$R/ckpt:22m:$R/ckpt/wm-22m/cooldown_22m_150k.pth.tar \
            meta_1b_300k=$R/ckpt:1b:$R/ckpt/wm-1b/cooldown_1b_300k.pth.tar"
echo "end=$(date -u +%FT%TZ)"
