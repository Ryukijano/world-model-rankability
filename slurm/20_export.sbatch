#!/bin/bash
#SBATCH --job-name=rank-export
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# CPU-only export of anchors + score files for the HF dataset (no network).
set -uo pipefail
R=/projects/u6xn/rankability; D=$R/data; OUT=$R/hf_export
echo "job=${SLURM_JOB_ID} start=$(date -u +%FT%TZ) sha=$(git -C $R/robo_jepa rev-parse HEAD) out=$OUT"
apptainer exec --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && python3 $R/probe/export_slim.py $OUT \
  pretrain:$D/anchors_bench_v2:$D/scores_bench/anchors_bench_v2__*.npz,$D/scores_bench_last4/anchors_bench_v2_last4__*.npz \
  target:$D/anchors_bench_target_v2:$D/scores_bench/anchors_bench_target_v2__*.npz"
ls -la $OUT; du -sh $OUT; echo "end=$(date -u +%FT%TZ)"
