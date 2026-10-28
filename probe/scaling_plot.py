"""Ranking quality vs training compute. Reads data/<dir>/<anch>__wm-<size>__cooldown_<size>_<k>k.npz files, puts each on the
README compute axis C = pre*1000*k + cool*5000 FLOPs (pre/cool = FLOPs per iteration per size, k = pretraining step in
thousands), computes the analyze.py metrics (all / random / goal-directed subsets) and writes a CSV + a PNG.
Usage: scaling_plot.py <scores_dir> <anchors_name> <out_prefix>"""
import glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import metrics, ece_heldout

FLOPS = {"22m": (1.562e15, 6.860e15), "50m": (3.098e15, 1.311e16), "100m": (5.743e15, 2.353e16), "300m": (1.566e16, 4.190e16),
         "1b": (4.777e16, 1.761e17), "2b": (8.117e16, 2.864e17), "4b": (1.578e17, 5.344e17), "8b": (3.054e17, 1.003e18)}
PARAMS = {"22m": 22, "50m": 50, "100m": 100, "300m": 300, "1b": 1000, "2b": 2000, "4b": 4000, "8b": 8000}


def main(d, anch, out):
    rows = []; groups = {}
    for p in sorted(glob.glob(f"{d}/{anch}__wm-*__cooldown_*k*.npz")):
        groups.setdefault(re.sub(r"__shard\d+-\d+", "", p), []).append(p)  # merge shards of the same checkpoint/condition
    for key_p, parts in sorted(groups.items()):
        m = re.search(r"wm-(\w+?)__cooldown_\w+?_(\d+)k", key_p); size, k = m.group(1), int(m.group(2))
        pre, cool = FLOPS[size]; C = pre * 1000 * k + cool * 5000
        zs = [np.load(q, allow_pickle=True) for q in parts]
        z = {f: np.concatenate([zz[f] for zz in zs]) for f in ["anchor", "l1", "dist", "d0", "cand_source"] + [f for f in ("pred_err", "mv_ratio_med") if f in zs[0].files]}
        if os.environ.get("ANALYZE_DROP_STUCK") == "1" and "mv_ratio_med" in z:  # same stuck rule as analyze.py
            keep = z["mv_ratio_med"] >= 0.5; z = {f: v[keep] for f, v in z.items()}
        z["files"] = list(z.keys()); src = z["cand_source"]; n = len(z["anchor"])
        tail = os.path.basename(key_p).split("k__", 1); r = dict(size=size, step_k=k, compute=C, params_m=PARAMS[size], n=n, cond=(tail[1].replace(".npz", "") if len(tail) > 1 else "clean"))
        for sname, sel in [("all", None), ("rnd", 0), ("goal", 1)]:
            per = []
            for i in range(n):
                s = None if sel is None else ((src[i] == 0) if sel == 0 else (src[i] > 0))
                if s is not None and s.sum() < 4:
                    continue
                per.append(metrics(z["l1"][i], z["dist"][i], z["d0"][i], sel=s))
            for key in ["spearman", "auroc_5cm", "regret", "hit1", "hit3", "pairwise"]:
                v = np.array([q[key] for q in per], float); r[f"{sname}_{key}"] = np.nanmean(v); r[f"{sname}_{key}_se"] = np.nanstd(v) / np.sqrt(np.isfinite(v).sum())
        r["ece"], _ = ece_heldout(list(z["l1"]), list(z["dist"])) if n >= 10 else (np.nan, np.nan)
        r["pred_err"] = float(np.nanmean(z["pred_err"])) if "pred_err" in z and np.isfinite(z["pred_err"]).any() else np.nan  # imagination L1 on stored post-action frames
        rows.append(r)
    if not rows:
        print("no files"); return
    keys = list(rows[0].keys())
    with open(out + ".csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(f"wrote {out}.csv ({len(rows)} checkpoints)")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    panels = [("all_spearman", "Spearman(score, -dist), all 64"), ("goal_spearman", "Spearman within goal-directed"), ("all_auroc_5cm", "AUROC @5cm"),
              ("goal_pairwise", "pairwise acc. within goal-directed"), ("all_regret", "top-1 regret (m)"), ("ece", "held-out ECE"), ("pred_err", "imagination error (latent L1, post-action frames)")]
    panels = [pn for pn in panels if not (pn[0] == "pred_err" and all(np.isnan(r["pred_err"]) for r in rows))]
    fig, axes = plt.subplots(2, 4 if len(panels) > 6 else 3, figsize=(20 if len(panels) > 6 else 15, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        conds = sorted({r["cond"] for r in rows}, key=lambda c: (not c.startswith("clean"), c))
        for ci, cond in enumerate(conds):  # one line per (size, condition); base condition solid, others dashed
            for size in FLOPS:
                rs = sorted([r for r in rows if r["size"] == size and r["cond"] == cond], key=lambda r: r["compute"])
                if not rs:
                    continue
                x = [r["compute"] for r in rs]; y = [r[key] for r in rs]; se = [r.get(key + "_se", 0) for r in rs]
                ax.errorbar(x, y, yerr=se, marker="o" if ci == 0 else "x", ms=3, capsize=2, ls="-" if ci == 0 else "--", alpha=1 if ci == 0 else .6,
                            label=size if ci == 0 else f"{size} {cond}")
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("training compute C (FLOPs, README formula)"); ax.grid(alpha=.3)
    axes.flat[0].legend(fontsize=8); fig.suptitle(f"{anch}: n={rows[0]['n']} anchors x K=64, frozen RoboJEPA cooldown checkpoints, l1 objective")
    fig.tight_layout(); fig.savefig(out + ".png", dpi=120); print("wrote", out + ".png")


if __name__ == "__main__":
    main(*sys.argv[1:4])
