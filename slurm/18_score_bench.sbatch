#!/bin/bash
#SBATCH --job-name=rank-score-bench
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Bench scoring: one checkpoint (SZ, W) x one shard (SHARD_I of SHARD_N) of ANCH, optional COND/VIEWS/HIST. Idempotent.
# 4 python workers share the GPU (each ~1 GB + encoder 2 GB... ViT-G fp16 ~2 GB) to use the node better.
set -uo pipefail
R=/projects/u6xn/rankability; SZ=${SZ:-22m}; W=${W:-latest}; ANCH=${ANCH:-anchors_bench_v2}; SHARD_N=${SHARD_N:-8}; SHARDS=${SHARDS:-"0 1 2 3"}
COND=${COND:-clean}; HIST=${HIST:-last}; OUTDIR=${OUTDIR:-scores_bench}; EXTRA=${EXTRA:-}
[ "$W" = latest ] && W=$(basename $(ls $R/ckpt/wm-$SZ/cooldown_${SZ}_*k.pth.tar | sed -E 's/.*_([0-9]+)k\.pth\.tar/\1 &/' | sort -n | awk '{print $2}' | tail -1))
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ) sha=$(git -C $R/robo_jepa rev-parse HEAD) wm-$SZ/$W anchors=$ANCH n=$(ls $R/data/$ANCH/anchor_*.npz | wc -l) shards=[$SHARDS]/$SHARD_N cond=$COND hist=$HIST"
for i in $SHARDS; do
  OUT=$R/data/$OUTDIR/${ANCH}__wm-${SZ}__${W%.pth.tar}__${COND}_h-${HIST}.npz
  [ -s "${OUT%.npz}__shard${i}-${SHARD_N}.npz" ] && { echo "skip shard $i"; continue; }
  apptainer exec --nv --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/score_anchors.py --model_dir $R/ckpt/wm-$SZ --weights $W --anchors $R/data/$ANCH --out $OUT --cond $COND --history $HIST --shard $i $SHARD_N $EXTRA" 2>&1 | grep -E "^wrote|^loaded|Error|Traceback|out of memory" > $R/logs/${SLURM_JOB_ID}_shard$i.log &
done
wait; cat $R/logs/${SLURM_JOB_ID}_shard*.log; echo "end=$(date -u +%FT%TZ)"
