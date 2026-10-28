"""Hand-check helper for a pilot anchor set (runs in the container job): numeric sanity over all anchors + a PNG montage
(init views | goal views) for a few anchors so a human can look at the frames. Usage: inspect_anchors.py <dir> <out.png>"""
import glob, sys
import numpy as np
from PIL import Image, ImageDraw

d, outpng = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{d}/anchor_*.npz")); print(f"{len(files)} anchors in {d}")
rows = []
for f in files:
    z = np.load(f); dist = z["dist"]; src = z["cand_source"]
    mv = np.linalg.norm(z["eef_final"] - z["eef0"][None], axis=1); cmd = np.linalg.norm(z["cands"][:, :3], axis=1)
    ratio = mv / np.maximum(cmd, 1e-6)
    rows.append((float(z["d0"]), float(z["det_err"]), dist.min(), np.median(dist), dist.max(), (dist < 0.05).mean(),
                 (dist < 0.5 * float(z["d0"])).mean(), np.isnan(dist).sum(), len(dist), *(np.median(dist[src == s]) if (src == s).any() else np.nan for s in (0, 1, 2)),
                 *(np.median(ratio[src == s]) if (src == s).any() else np.nan for s in (0, 1, 2)), np.median(cmd[src == 2])))
r = np.array(rows, dtype=np.float64)
names = ["d0", "det_err", "dist_min", "dist_med", "dist_max", "frac<5cm", "frac<d0/2", "nan", "K", "med_src0", "med_src1", "med_src2", "mv/cmd_src0", "mv/cmd_src1", "mv/cmd_src2", "med_cmd_src2"]
print(" ".join(f"{n}:{np.nanmean(r[:, i]):.4f}[{np.nanmin(r[:, i]):.4f},{np.nanmax(r[:, i]):.4f}]" for i, n in enumerate(names)))
print("anchors with no candidate <5cm:", int((r[:, 5] == 0).sum()), "| max det_err:", r[:, 1].max(), "| K!=64:", int((r[:, 8] != 64).sum()),
      "| STUCK anchors (median moved/cmd of random cands < 0.5):", [files[i].split("/")[-1] for i in np.where(r[:, 12] < 0.5)[0]])
pick = files[:: max(1, len(files) // 6)][:6]
tiles = []
for f in pick:
    z = np.load(f); ims = [Image.fromarray(x) for x in list(z["init_imgs"]) + list(z["goal_imgs"])]
    w, h = ims[0].size; row = Image.new("RGB", (w * len(ims), h + 14), "white"); dr = ImageDraw.Draw(row)
    for i, im in enumerate(ims):
        row.paste(im, (i * w, 14))
    dr.text((2, 1), f"{f.split('/')[-1]} d0={float(z['d0']):.3f} best={z['dist'].min():.3f} frac<5cm={(z['dist']<0.05).mean():.2f} | init views .. goal views", fill="black")
    tiles.append(row)
W = max(t.size[0] for t in tiles); H = sum(t.size[1] for t in tiles); M = Image.new("RGB", (W, H), "white"); y = 0
for t in tiles:
    M.paste(t, (0, y)); y += t.size[1]
M.save(outpng); print("wrote", outpng)
