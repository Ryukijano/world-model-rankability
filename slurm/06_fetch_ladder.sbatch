#!/bin/bash
#SBATCH --job-name=rank-fetch-ladder
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Transfer job (no training): every released RoboJEPA cooldown checkpoint (CC BY-NC-SA 4.0), small sizes first.
# Resumable with wget -c; resubmit (afterany chain) until "ALL_PRESENT".
set -uo pipefail
cd /projects/u6xn/rankability/ckpt
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ)"
declare -A STEPS=(
  [22m]="10 20 30 40 50 60 70 80 90 100 120 150" [50m]="10 20 30 40 50 60 70 80 90 120"
  [100m]="10 20 30 40 50 60 70 80 90 100 120 150" [300m]="10 50 100 150 200 250 300"
  [1b]="10 50 100 150 200 250 300" [2b]="10 50 100 150 200 250 300 350 400 450"
  [4b]="10 50 100 150 200 250 300 450" [8b]="50 100 150 200 250 300" )
missing=0
for sz in 22m 50m 100m 300m 1b 2b 4b 8b; do
  mkdir -p wm-$sz; cp -n ../robo_jepa/configs/released/wm-$sz/config.json wm-$sz/ 2>/dev/null
  for k in ${STEPS[$sz]}; do
    f=cooldown_${sz}_${k}k.pth.tar; url=https://dl.fbaipublicfiles.com/robojepa/wm/wm-$sz/$f
    len=$(curl -sI "$url" | awk 'tolower($1)=="content-length:"{print $2+0}')
    have=$(stat -c %s wm-$sz/$f 2>/dev/null || echo 0)
    if [ -n "$len" ] && [ "$have" = "$len" ]; then continue; fi
    missing=$((missing+1)); echo "fetch wm-$sz/$f have=$have len=$len"
    timeout 1500 wget -c -q -P wm-$sz "$url" || echo "PARTIAL wm-$sz/$f"
  done
done
[ "$missing" = 0 ] && echo ALL_PRESENT
du -sh /projects/u6xn/rankability/ckpt; echo "end=$(date -u +%FT%TZ)"
