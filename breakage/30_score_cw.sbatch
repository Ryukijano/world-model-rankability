#!/bin/bash
#SBATCH --job-name=rank-score-cw
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
# Change-weighted rescoring: one checkpoint (SZ, W) x shards SHARDS of SHARD_N on ANCH, 4 workers share the GPU.
set -uo pipefail
R=/projects/u6xn/rankability; SZ=${SZ:-22m}; W=${W}; ANCH=${ANCH:-anchors_bench_v2}; SHARD_N=${SHARD_N:-8}; SHARDS=${SHARDS:-"0 1 2 3"}
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ) wm-$SZ/$W anchors=$ANCH n=$(ls $R/data/$ANCH/anchor_*.npz | wc -l) shards=[$SHARDS]/$SHARD_N"
for i in $SHARDS; do
  OUT=$R/data/scores_cw/${ANCH}__wm-${SZ}__${W%.pth.tar}.npz
  [ -s "${OUT%.npz}__shard${i}-${SHARD_N}.npz" ] && { echo "skip shard $i"; continue; }
  apptainer exec --nv --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/score_cw.py --model_dir $R/ckpt/wm-$SZ --weights $W --anchors $R/data/$ANCH --out $OUT --shard $i $SHARD_N" > $R/logs/${SLURM_JOB_ID}_cw_shard$i.log 2>&1 &
done
wait; grep -hE "^wrote|Error|Traceback|out of memory" $R/logs/${SLURM_JOB_ID}_cw_shard*.log; echo "end=$(date -u +%FT%TZ)"
