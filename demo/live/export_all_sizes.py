"""Export bf16 predictor-only weights + config for every released RoboJEPA size (final checkpoint) for the live Space."""
import os, shutil, torch
from safetensors.torch import save_file
R = "/projects/u6xn/rankability"; OUT = f"{R}/hf_live_weights/sizes"
FINAL = {"22m": "150k", "50m": "120k", "100m": "150k", "300m": "300k", "1b": "300k", "2b": "450k", "4b": "450k", "8b": "300k"}
for size, k in FINAL.items():
    d = f"{OUT}/{size}"; os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{d}/predictor_bf16.safetensors"):
        print("skip", size); continue
    p = torch.load(f"{R}/ckpt/wm-{size}/cooldown_{size}_{k}.pth.tar", map_location="cpu", mmap=True, weights_only=False)
    key = "predictor" if "predictor" in p else "model"
    sd = {n.replace("module.", ""): v.to(torch.bfloat16).contiguous() for n, v in p[key].items()}
    save_file(sd, f"{d}/predictor_bf16.safetensors", metadata={"source": f"cooldown_{size}_{k}.pth.tar"})
    shutil.copy(f"{R}/ckpt/wm-{size}/config.json", f"{d}/config.json")
    print(size, k, "params %.1fM" % (sum(v.numel() for v in sd.values()) / 1e6), "bytes", os.path.getsize(f"{d}/predictor_bf16.safetensors"), flush=True)
    del p, sd
