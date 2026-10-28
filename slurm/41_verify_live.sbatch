#!/bin/bash
#SBATCH --job-name=verify-live-space
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
R=/projects/u6xn/rankability
apptainer exec --nv --bind $R:$R /projects/u6xn/sif-images/pytorch_26.09-py3.sif bash -c "source $R/venv/bin/activate && python3 $R/hf_live/verify_gpu.py"
