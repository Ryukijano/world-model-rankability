"""Export Space-ready weights: V-JEPA 2.1 ViT-G/384 target encoder (bf16) + RoboJEPA-22M 150k predictor."""
import json, os, shutil, torch
from safetensors.torch import save_file
R = "/projects/u6xn/rankability"; OUT = f"{R}/hf_live_weights"; os.makedirs(OUT, exist_ok=True)
s = torch.load(f"{R}/ckpt/encoder/vjepa2_1_vitG_384.pt", map_location="cpu", mmap=True, weights_only=False)
print("encoder ckpt keys", list(s.keys()))
enc = {k.replace("module.backbone.", ""): v.to(torch.bfloat16).contiguous() for k, v in s["target_encoder"].items()}
print("encoder params %.1fM" % (sum(v.numel() for v in enc.values()) / 1e6))
save_file(enc, f"{OUT}/vjepa2_1_vitG_384_target_encoder_bf16.safetensors")
p = torch.load(f"{R}/ckpt/wm-22m/cooldown_22m_150k.pth.tar", map_location="cpu", weights_only=False)
k = "predictor" if "predictor" in p else "model"
pred = {kk.replace("module.", ""): v.float().contiguous() for kk, v in p[k].items()}
print("ckpt keys", list(p.keys()), "predictor params %.1fM" % (sum(v.numel() for v in pred.values()) / 1e6))
save_file(pred, f"{OUT}/robojepa_22m_150k_predictor.safetensors")
shutil.copy(f"{R}/ckpt/wm-22m/config.json", f"{OUT}/config.json")
print("wrote", os.listdir(OUT), {f: os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)})
