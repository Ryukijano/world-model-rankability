"""Slim export for the HF dataset: per split, anchor metadata + init/goal frames + candidate banks + replay outcomes,
and every score file merged by (size, step, cond). numpy only (login-node/compute safe). Usage:
python3 export_slim.py OUTDIR split:ANCHDIR:SCORE_GLOB[,SCORE_GLOB...] ..."""
import glob, json, os, re, sys
import numpy as np

META = ["scene", "task_seed", "d0", "det_err", "eef0", "goal_pos", "to_goal_base", "init_pose", "R_wb"]
CAND = ["cands", "cand_source", "dist", "reward", "success", "eef_final"]


def export(out, split, anch_dir, score_globs):
    files = sorted(glob.glob(f"{anch_dir}/anchor_*.npz")); ids = [os.path.basename(f) for f in files]
    acc = {k: [] for k in META + CAND + ["init_imgs", "goal_imgs"]}
    for f in files:
        z = np.load(f)
        for k in acc:
            acc[k].append(z[k])
    np.savez_compressed(f"{out}/anchors_{split}.npz", anchor=np.array(ids), **{k: np.stack(v) for k, v in acc.items()})
    print(f"{split}: {len(ids)} anchors", flush=True)
    idx = {a: i for i, a in enumerate(ids)}; groups = {}
    for g in score_globs:
        for p in glob.glob(g):
            m = re.search(r"wm-(\w+?)__cooldown_\w+?_(\d+)k__(\w+?)_h-last", os.path.basename(p))
            groups.setdefault((m.group(1), int(m.group(2)), m.group(3)), []).append(p)
    for (size, k, cond), parts in sorted(groups.items()):
        N = len(ids); l1 = np.full((N, 64), np.nan, np.float32); cos = l1.copy(); pe = l1.copy(); mv = np.full(N, np.nan); seen = np.zeros(N, int)
        for p in parts:
            z = np.load(p)
            for j, a in enumerate(z["anchor"]):
                i = idx[str(a)]; seen[i] += 1; l1[i] = z["l1"][j]; cos[i] = z["cos"][j]; mv[i] = z["mv_ratio_med"][j]
                if "pred_err" in z.files:
                    pe[i] = z["pred_err"][j]
                assert np.allclose(z["dist"][j], acc["dist"][i]), (p, a)
        assert seen.max() <= 1, f"duplicate anchors in {size} {k} {cond}"
        np.savez_compressed(f"{out}/scores_{split}__{size}__{k}k__{cond}.npz", l1=l1, cos=cos, pred_err=pe, mv_ratio_med=mv, scored=seen.astype(bool))
        print(f"  {size} {k}k {cond}: {seen.sum()}/{N} anchors from {len(parts)} files", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for spec in sys.argv[2:]:
        split, anch, globs = spec.split(":", 2); export(out, split, anch, globs.split(","))
