#!/bin/bash
#SBATCH --job-name=rank-score-hist
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# History / state-token axes: score ANCH with latest ckpt per size for each "<history>:<state_key>" in VARIANTS. Idempotent.
set -uo pipefail
R=/projects/u6xn/rankability; SIZES=${SIZES:-22m 1b}; ANCH=${ANCH:-anchors_pilot_v2}; VARIANTS=${VARIANTS:-last:init_pose full:init_pose last:init_proprio}; OUTDIR=${OUTDIR:-scores_hist}
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ) robo_jepa_sha=$(git -C $R/robo_jepa rev-parse HEAD) sizes=[$SIZES] anchors=$ANCH variants=[$VARIANTS] n_anchors=$(ls $R/data/$ANCH/anchor_*.npz | wc -l)"
for SZ in $SIZES; do
  f=$(ls $R/ckpt/wm-$SZ/cooldown_${SZ}_*k.pth.tar | sed -E 's/.*_([0-9]+)k\.pth\.tar/\1 &/' | sort -n | awk '{print $2}' | tail -1); W=$(basename $f)
  for V in $VARIANTS; do
    H=${V%%:*}; S=${V##*:}; OUT=$R/data/$OUTDIR/${ANCH}__wm-${SZ}__${W%.pth.tar}__h-${H}__s-${S}.npz
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    echo "=== wm-$SZ/$W history=$H state=$S $(date -u +%T)"
    apptainer exec --nv --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/score_anchors.py --model_dir $R/ckpt/wm-$SZ --weights $W --anchors $R/data/$ANCH --out $OUT --history $H --state_key $S" 2>&1 | grep -vE "WARNING|INFO|self.gen"
  done
done
apptainer exec --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R && python3 probe/analyze.py data/$OUTDIR/${ANCH}__*.npz && python3 -c \"import numpy as np,glob; f=sorted(glob.glob('data/$ANCH/anchor_*.npz'))[0]; z=np.load(f); print({k:(z[k].shape, str(z[k].dtype)) for k in z.files}); print('ctx_actions', z['ctx_actions']); print('hist_pose', z['hist_pose'])\"" 2>&1 | grep -vE "WARNING|INFO"
echo "end=$(date -u +%FT%TZ)"
