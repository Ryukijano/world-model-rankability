import sys, time, numpy as np, torch
sys.path.insert(0, "/projects/u6xn/rankability/hf_live/space")
import rjlive
from scipy.stats import spearmanr
W = "/projects/u6xn/rankability/hf_live_weights"
SIZES = ["22m", "50m", "100m", "300m", "1b", "2b", "4b", "8b"]
enc = rjlive.load_encoder(W, "cuda")
models = {}
for s in SIZES:
    t = time.time()
    models[s], info = rjlive.load_predictor(f"{W}/sizes/{s}/config.json", f"{W}/sizes/{s}/predictor_bf16.safetensors", enc, "cuda", torch.float16)
    print(s, "load", round(time.time() - t, 1), "s missing", len(info["missing"]), "unexpected", len(info["unexpected"]), flush=True)
print("all loaded, GPU mem %.1f GiB" % (torch.cuda.memory_allocated() / 2**30), flush=True)
d = np.load("/projects/u6xn/rankability/hf_live/space/data/anchors300.npz")
for s in SIZES:
    rh, am, secs = [], 0, []
    for i in range(10):
        torch.cuda.synchronize(); t = time.time()
        o, g = rjlive.encode_state(enc_wm := models[s], d["init_imgs"][i], d["goal_imgs"][i], "cuda")
        sc = rjlive.score_from_features(models[s], o, g, d["init_pose"][i], np.concatenate([d["cands"][i], d["cands"][i]]), "cuda")[:64]
        torch.cuda.synchronize(); secs.append(time.time() - t)
        p = d[f"l1_pre_{s}"][i]; rh.append(spearmanr(sc, p).correlation); am += int(sc.argmax() == p.argmax())
    print(f"{s}: spearman min={min(rh):.4f} median={np.median(rh):.5f} argmax={am}/10 sec_warm={np.median(secs[1:]):.2f} peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
