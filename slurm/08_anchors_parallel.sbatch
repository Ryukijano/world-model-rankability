#!/bin/bash
#SBATCH --job-name=rank-anchors
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Anchor collection, N CPU workers sharing one node (EGL on the GPU). Idempotent: skips existing anchors. No downloads.
# Usage: sbatch --export=ALL,OUT=anchors_pilot_v1,SCENES="0",TASKS="$(seq -s' ' 0 99)",NW=16 slurm/08_anchors_parallel.sbatch
set -uo pipefail
R=/projects/u6xn/rankability; OUT=${OUT:-anchors_pilot_v1}; SCENES=${SCENES:-0}; TASKS=${TASKS:-$(seq -s' ' 0 15)}; NW=${NW:-16}
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ) robo_jepa_sha=$(git -C $R/robo_jepa rev-parse HEAD) out=$OUT scenes=[$SCENES] ntasks=$(echo $TASKS | wc -w) nw=$NW rng_seed=0"
for w in $(seq 0 $((NW-1))); do
  apptainer exec --nv --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && cd $R/.rc && python3 $R/probe/collect_anchors.py --out $R/data/$OUT --scene_seeds $SCENES --task_seeds $TASKS --k 64 --worker $w --nworkers $NW ${EXTRA:-}" > $R/logs/${SLURM_JOB_ID}_w$w.log 2>&1 &
done
wait
grep -h "^anchor" $R/logs/${SLURM_JOB_ID}_w*.log | wc -l | xargs echo anchors_written=
grep -h "^anchor" $R/logs/${SLURM_JOB_ID}_w*.log | awk '{print $NF}' | sed 's/s//' | sort -n | awk '{a[NR]=$1} END{print "sec/anchor median", a[int(NR/2)+1]}'
du -sh $R/data/$OUT; echo "end=$(date -u +%FT%TZ)"
