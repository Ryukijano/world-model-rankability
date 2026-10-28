import sys, time, numpy as np, torch
sys.path.insert(0, "/projects/u6xn/rankability/hf_live/space")
import rjlive
from scipy.stats import spearmanr
t = time.time(); wm, info = rjlive.load_world_model("/projects/u6xn/rankability/hf_live_weights", "cuda")
print("load", round(time.time() - t, 1), "s", info, flush=True)
d = np.load("/projects/u6xn/rankability/hf_live/space/data/anchors300.npz")
rhos, am, mx, secs = [], 0, 0.0, []
for i in range(30):
    torch.cuda.synchronize(); t = time.time()
    s = rjlive.score_candidates(wm, d["init_imgs"][i], d["goal_imgs"][i], d["init_pose"][i], d["cands"][i], "cuda")
    torch.cuda.synchronize(); secs.append(time.time() - t)
    p = d["l1_pre"][i]; rhos.append(spearmanr(s, p).correlation); am += int(s.argmax() == p.argmax()); mx = max(mx, float(np.abs(s - p).max()))
print(f"anchors=30 spearman min={min(rhos):.5f} median={np.median(rhos):.5f} argmax_agree={am}/30 maxabs={mx:.2e} "
      f"sec first={secs[0]:.2f} warm_median={np.median(secs[1:]):.3f} peak_mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
