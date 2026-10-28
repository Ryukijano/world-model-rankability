"""Cache frozen V-JEPA 2.1 ViT-G features (fp16) for every anchor: init, goal and the replayed post-action frames.
Same preprocessing as score_anchors.encode_views. Output: <out>/<anchor>.npz with init/goal [V,h,w,C], post [P,V,h,w,C]."""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from score_anchors import R, WorldModel, encode_views  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--anchors", required=True); p.add_argument("--out", required=True); p.add_argument("--shard", type=int, nargs=2, default=(0, 1))
p.add_argument("--model_dir", default=f"{R}/ckpt/wm-22m"); p.add_argument("--weights", default="cooldown_22m_150k.pth.tar")
p.add_argument("--encoder", default=f"{R}/ckpt/encoder/vjepa2_1_vitG_384.pt"); p.add_argument("--image_size", type=int, nargs=2, default=[240, 320])
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
files = sorted(glob.glob(f"{a.anchors}/anchor_*.npz"))[a.shard[0]::a.shard[1]]
wm = WorldModel.load_pretrained(a.model_dir, load_decoder=False, load_encoder=True, load_compiled=False, encoder_path=a.encoder, device="cuda", weights_filename=a.weights)
t0 = time.time(); n = 0
for f in files:
    o = f"{a.out}/{os.path.basename(f)}"
    if os.path.exists(o):
        continue
    d = np.load(f)
    enc = lambda im: encode_views(wm, im, a.image_size, "cuda")[0, 0].half().cpu().numpy()
    post = np.stack([enc(x) for x in d["post_imgs"]]) if "post_imgs" in d and d["post_imgs"].ndim == 5 else np.zeros((0,), np.float16)
    np.savez(o + ".tmp.npz", init=enc(d["init_imgs"]), goal=enc(d["goal_imgs"]), post=post)
    os.replace(o + ".tmp.npz", o); n += 1
print(f"cached {n}/{len(files)} anchors in {time.time()-t0:.0f}s -> {a.out}", flush=True)
