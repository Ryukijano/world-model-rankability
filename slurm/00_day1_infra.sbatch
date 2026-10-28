#!/bin/bash
#SBATCH --job-name=rank-day1-infra
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xn/rankability/logs/%x-%j.out
#SBATCH --error=/projects/u6xn/rankability/logs/%x-%j.err
# Day-1 infrastructure test for the RoboJEPA rankability probe + AeroLoc kill test.
# Checks: outbound HTTPS from compute, aarch64 wheel availability, SDPA throughput,
# RoboJEPA checkpoint load. No dataset download, no training.
set -uo pipefail
R=/projects/u6xn/rankability
SIF=/projects/u6xn/sif-images/pytorch_26.09-py3.sif
echo "job=${SLURM_JOB_ID} host=$(hostname) arch=$(uname -m) date=$(date -u +%FT%TZ)"
echo "sif=${SIF} sha256=$(cut -d' ' -f1 ${SIF}.sha256)"
echo "robo_jepa_sha=$(git -C $R/robo_jepa rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== outbound network from compute node"
timeout 20 curl -sI https://dl.fbaipublicfiles.com/robojepa/wm/wm-22m/cooldown_22m_150k.pth.tar | head -1 || echo "NO_OUTBOUND_HTTPS"
timeout 20 curl -sI https://pypi.org/simple/mujoco/ | head -1 || echo "NO_PYPI"
echo "== walltime limits"; sacctmgr show qos workq_qos format=Name,MaxWall -P 2>/dev/null
apptainer exec --nv --bind $R:$R ${SIF} bash -c '
set -uo pipefail
echo "== versions"; python3 -c "import torch,sys;print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
echo "== aarch64 wheel availability (dry run, no install)"
export PIP_CACHE_DIR=/tmp/pipcache
for p in mujoco robosuite robocasa gymnasium h5py einops timm scipy scikit-learn faiss-cpu av torchcodec xformers flash-attn decord; do
  if timeout 90 pip download --no-deps -q -d /tmp/wheels "$p" >/tmp/pip_$p.log 2>&1; then f=$(ls -t /tmp/wheels | grep -i "^${p//-/_}\|^$p" | head -1); echo "OK   $p  $f"; else echo "FAIL $p  $(tail -1 /tmp/pip_$p.log | cut -c1-120)"; fi
done
echo "== SDPA throughput: ViT-L/16-like predictor, bf16, 4 GPUs independent"
python3 - <<PY
import torch, time
torch.backends.cuda.matmul.allow_tf32=True
def bench(dev, d=1024, L=24, T=1024, B=16, heads=16):
    blk = torch.nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, norm_first=True, activation="gelu").to(dev, torch.bfloat16)
    x = torch.randn(B, T, d, device=dev, dtype=torch.bfloat16)
    with torch.no_grad(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
        for _ in range(3): y = blk(x)
        torch.cuda.synchronize(dev); t=time.time()
        for _ in range(20): y = blk(x)
        torch.cuda.synchronize(dev); dt=(time.time()-t)/20
    return B*T/dt*L  # tokens/s equivalent for full L-layer stack (inference)
for i in range(torch.cuda.device_count()):
    dev=f"cuda:{i}"; print(dev, torch.cuda.get_device_name(i), f"inference tokens/s (24-layer d=1024, seq 1024): {bench(dev):,.0f}")
PY
echo "== RoboJEPA checkpoint load (state dict only)"
python3 - <<PY
import torch, glob, os, json
for f in sorted(glob.glob("/projects/u6xn/rankability/ckpt/wm-*/*.pth.tar")):
    try:
        sd = torch.load(f, map_location="cpu", weights_only=False)
        keys = list(sd.keys()) if isinstance(sd, dict) else []
        inner = sd.get("model") or sd.get("predictor") or sd.get("state_dict") or sd
        n = sum(v.numel() for v in inner.values() if hasattr(v, "numel")) if isinstance(inner, dict) else -1
        print(os.path.basename(f), "top-level keys:", keys[:8], "params:", f"{n/1e6:.1f}M")
    except Exception as e: print(os.path.basename(f), "LOAD_FAIL", repr(e)[:200])
PY
'
echo "== done $(date -u +%FT%TZ)"
