"""Cache frozen DINOv3 ViT-L/16 patch features (fp16) for every anchor: init, goal and the replayed post-action frames.
Encoder-swap twin of cache_feats.py: same output layout (<out>/<anchor>.npz with init/goal [V,h,w,C], post [P,V,h,w,C]),
same 240x320 resize, different encoder and 16px patches (grid 15x20x1024). No network access; weights pre-staged."""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from score_anchors import R  # noqa: E402
from dino_encode import load_dino, encode_dino  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--anchors", required=True); p.add_argument("--out", required=True); p.add_argument("--shard", type=int, nargs=2, default=(0, 1))
p.add_argument("--scenes", type=int, nargs="*", default=None, help="only encode anchors whose npz scene is in this list")
p.add_argument("--dino_dir", default=f"{R}/ckpt/dinov3-vitl16"); p.add_argument("--image_size", type=int, nargs=2, default=[240, 320])
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
files = sorted(glob.glob(f"{a.anchors}/anchor_*.npz"))[a.shard[0]::a.shard[1]]
model = load_dino(a.dino_dir)
t0 = time.time(); n = 0
for f in files:
    o = f"{a.out}/{os.path.basename(f)}"
    if os.path.exists(o):
        continue
    d = np.load(f)
    if a.scenes is not None and int(d["scene"]) not in a.scenes:
        continue
    enc = lambda im: encode_dino(model, im, a.image_size)
    post = np.stack([enc(x) for x in d["post_imgs"]]) if "post_imgs" in d and d["post_imgs"].ndim == 5 else np.zeros((0,), np.float16)
    np.savez(o + ".tmp.npz", init=enc(d["init_imgs"]), goal=enc(d["goal_imgs"]), post=post)
    os.replace(o + ".tmp.npz", o); n += 1
print(f"cached {n} anchors in {time.time()-t0:.0f}s -> {a.out}", flush=True)
