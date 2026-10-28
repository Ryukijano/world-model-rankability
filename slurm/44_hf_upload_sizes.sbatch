#!/bin/bash
#SBATCH --job-name=hf-upload-sizes
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
R=/projects/u6xn/rankability
cd $R/hf_live_weights
apptainer exec --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && HF_HUB_DISABLE_PROGRESS_BARS=1 hf upload Ryukijano/robojepa-22m-rankability-live sizes sizes --repo-type model --commit-message 'Add bf16 predictors for all eight released RoboJEPA sizes'"
echo "exit=$?"
