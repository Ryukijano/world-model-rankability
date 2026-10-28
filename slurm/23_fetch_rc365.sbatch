#!/bin/bash
#SBATCH --job-name=rank-fetch-rc365
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Transfer job (no training): RoboCasa365 pretrain-atomic demos (ember-lab-berkeley, MIT, LeRobot v3), only the two
# benchmark cameras (agentview_left, eye_in_hand) + parquet + meta (~2.7 GB). Resumable; resubmit until ALL_PRESENT.
# The target-atomic set (10 held-out kitchens = our test set) is deliberately NOT fetched.
set -uo pipefail
R=/projects/u6xn/rankability; OUT=$R/data/rc365/pretrain-atomic; mkdir -p $OUT
echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date -u +%FT%TZ) out=$OUT"
apptainer exec --nv --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && timeout 1600 python3 - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download('ember-lab-berkeley/robocasa365-pretrain-atomic', repo_type='dataset', local_dir='$OUT', max_workers=8,
    allow_patterns=['meta/*', 'meta/**', 'data/**', 'videos/observation.images.robot0_agentview_left/**', 'videos/observation.images.robot0_eye_in_hand/**'])
print('ALL_PRESENT', p)
PY"
du -sh $OUT; find $OUT -name '*.mp4' | wc -l; echo "end=$(date -u +%FT%TZ)"
