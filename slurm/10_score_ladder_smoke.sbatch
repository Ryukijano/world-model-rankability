#!/bin/bash
#SBATCH --job-name=rank-score-ladder
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Score ANCH with every checkpoint present for SIZES (frozen; idempotent: skips existing score files). No downloads.
set -uo pipefail
R=/projects/u6xn/rankability; SIZES=${SIZES:-22m 50m 100m 300m 1b}; ANCH=${ANCH:-anchors_pilot_v0}; ONLY_LATEST=${ONLY_LATEST:-1}
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ) robo_jepa_sha=$(git -C $R/robo_jepa rev-parse HEAD) sizes=[$SIZES] anchors=$ANCH only_latest=$ONLY_LATEST"
for SZ in $SIZES; do
  files=$(ls $R/ckpt/wm-$SZ/cooldown_${SZ}_*k.pth.tar 2>/dev/null | sed -E 's/.*_([0-9]+)k\.pth\.tar/\1 &/' | sort -n | awk '{print $2}')
  [ "$ONLY_LATEST" = 1 ] && files=$(echo "$files" | tail -1)
  for f in $files; do
    W=$(basename $f); OUT=$R/data/scores/${ANCH}__wm-${SZ}__${W%.pth.tar}.npz
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    echo "=== wm-$SZ/$W $(date -u +%T)"
    apptainer exec --nv --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/score_anchors.py --model_dir $R/ckpt/wm-$SZ --weights $W --anchors $R/data/$ANCH --out $OUT" 2>&1 | grep -vE "WARNING|INFO|self.gen"
  done
done
apptainer exec --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R && python3 probe/analyze.py data/scores/${ANCH}__*.npz" 2>&1 | grep -vE "WARNING|INFO"
echo "end=$(date -u +%FT%TZ)"

