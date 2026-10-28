#!/bin/bash
#SBATCH --job-name=rank-fp32
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# fp16-autocast repeat determinism + fp32 comparison on the first 12 v3 anchors, 22m final ckpt.
set -uo pipefail
R=/projects/u6xn/rankability; A=${ANCH:-anchors_pilot_v3}; SIF=/projects/u6xn/sif-images/pytorch_26.09-py3.sif
echo "job=${SLURM_JOB_ID} start=$(date -u +%FT%TZ) anchors=$A"
for tag in fp16a fp16b fp32; do
  F=""; [ $tag = fp32 ] && F="--fp32"
  apptainer exec --nv --bind $R:$R $SIF bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/score_anchors.py --model_dir $R/ckpt/wm-22m --weights cooldown_22m_150k.pth.tar --anchors $R/data/$A --max_anchors 12 --out $R/data/scores_fp/${A}__22m__$tag.npz $F" 2>&1 | grep -E "^wrote|Error|Traceback"
done
apptainer exec --bind $R:$R $SIF bash -c "source $R/venv/bin/activate && cd $R && python3 probe/analyze.py data/scores_fp/${A}__22m__*.npz && python3 probe/fp_check.py data/scores_fp/${A}__22m__fp16a.npz data/scores_fp/${A}__22m__fp16b.npz data/scores_fp/${A}__22m__fp32.npz" 2>&1 | grep -vE "WARNING|INFO"
