#!/bin/bash
#SBATCH --job-name=rank-score-conds
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Shift axes: score ANCH under each COND with the latest ckpt of each size in SIZES. Idempotent. No downloads.
set -uo pipefail
R=/projects/u6xn/rankability; SIZES=${SIZES:-22m 1b}; ANCH=${ANCH:-anchors_pilot_v0}; CONDS=${CONDS:-clean noise10 noise25 blur2 shift16 dark50}; VIEWS=${VIEWS:-}
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ) robo_jepa_sha=$(git -C $R/robo_jepa rev-parse HEAD) sizes=[$SIZES] anchors=$ANCH conds=[$CONDS] views=[$VIEWS]"
for SZ in $SIZES; do
  f=$(ls $R/ckpt/wm-$SZ/cooldown_${SZ}_*k.pth.tar | sed -E 's/.*_([0-9]+)k\.pth\.tar/\1 &/' | sort -n | awk '{print $2}' | tail -1); W=$(basename $f)
  for C in $CONDS; do
    TAG=$C; VARG=""; [ -n "$VIEWS" ] && { TAG="${C}_v${VIEWS// /}"; VARG="--views $VIEWS"; }
    OUT=$R/data/scores_cond/${ANCH}__wm-${SZ}__${W%.pth.tar}__${TAG}.npz
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    echo "=== wm-$SZ/$W cond=$TAG $(date -u +%T)"
    apptainer exec --nv --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/score_anchors.py --model_dir $R/ckpt/wm-$SZ --weights $W --anchors $R/data/$ANCH --out $OUT --cond $C $VARG" 2>&1 | grep -vE "WARNING|INFO|self.gen"
  done
done
apptainer exec --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R && python3 probe/analyze.py data/scores_cond/${ANCH}__*.npz" 2>&1 | grep -vE "WARNING|INFO"
echo "end=$(date -u +%FT%TZ)"
