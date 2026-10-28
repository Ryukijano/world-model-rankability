"""Convert the slim Isambard export (hf_export/*.npz) into the HF dataset layout:
data/anchors-{split}-NNNNN.parquet (metadata + 4 PNG frames), data/candidates-{split}.parquet,
data/scores-{split}.parquet (long: one row per model checkpoint x condition x anchor x candidate)."""
import glob, io, os, re, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image

SRC, DST = sys.argv[1], sys.argv[2]
FLOPS = {"22m": (1.562e15, 6.860e15), "50m": (3.098e15, 1.311e16), "100m": (5.743e15, 2.353e16), "300m": (1.566e16, 4.190e16),
         "1b": (4.777e16, 1.761e17), "2b": (8.117e16, 2.864e17), "4b": (1.578e17, 5.344e17), "8b": (3.054e17, 1.003e18)}
SRCNAME = np.array(["random", "goal_coarse", "goal_fine"])
os.makedirs(f"{DST}/data", exist_ok=True)


def png(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG", optimize=True); return {"bytes": b.getvalue(), "path": None}


img_t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
for split in ["pretrain", "target"]:
    z = dict(np.load(f"{SRC}/anchors_{split}.npz")); ids = [a.replace(".npz", "") for a in z["anchor"]]; N = len(ids)
    mv = None
    for p in sorted(glob.glob(f"{SRC}/scores_{split}__*__clean.npz")):
        s = np.load(p)
        if s["scored"].all():
            mv = s["mv_ratio_med"]; break
    per = 300
    for c, lo in enumerate(range(0, N, per)):
        sl = range(lo, min(N, lo + per))
        cols = dict(anchor_id=[ids[i] for i in sl], split=[split] * len(sl), scene=z["scene"][lo:lo + per].astype(int),
                    task_seed=z["task_seed"][lo:lo + per].astype(int), d0_m=z["d0"][lo:lo + per], replay_det_err_m=z["det_err"][lo:lo + per],
                    stuck=(mv[lo:lo + per] < 0.5) if mv is not None else np.zeros(len(sl), bool),
                    eef0=list(z["eef0"][lo:lo + per]), goal_pos=list(z["goal_pos"][lo:lo + per]), to_goal_base=list(z["to_goal_base"][lo:lo + per]),
                    init_pose=list(z["init_pose"][lo:lo + per]), R_wb=[r.reshape(-1) for r in z["R_wb"][lo:lo + per]])
        t = pa.table({k: (pa.array([list(map(float, v)) for v in vals], pa.list_(pa.float64())) if k in ("eef0", "goal_pos", "to_goal_base", "init_pose", "R_wb") else vals) for k, vals in cols.items()})
        for name, arr, v in [("init_agentview", "init_imgs", 0), ("init_wrist", "init_imgs", 1), ("goal_agentview", "goal_imgs", 0), ("goal_wrist", "goal_imgs", 1)]:
            t = t.append_column(name, pa.array([png(z[arr][i, v]) for i in sl], img_t))
        pq.write_table(t, f"{DST}/data/anchors-{split}-{c:05d}.parquet")
        print(split, "anchors chunk", c, flush=True)
    K = z["cands"].shape[1]
    cand = pd.DataFrame(dict(anchor_id=np.repeat(ids, K), cand_idx=np.tile(np.arange(K), N), source=SRCNAME[z["cand_source"].reshape(-1)],
                             action=list(z["cands"].reshape(-1, 7).astype(np.float32)), final_dist_m=z["dist"].reshape(-1), reward=z["reward"].reshape(-1),
                             success_1mm=z["success"].reshape(-1), eef_final=list(z["eef_final"].reshape(-1, 3))))
    cand.to_parquet(f"{DST}/data/candidates-{split}.parquet", index=False)
    rows = []
    for p in sorted(glob.glob(f"{SRC}/scores_{split}__*.npz")):
        size, k, cond = re.search(r"__(\w+?)__(\d+)k__(\w+)\.npz", os.path.basename(p)).groups(); k = int(k); s = np.load(p)
        keep = np.repeat(s["scored"], K)
        if not keep.any():
            continue
        pre, cool = FLOPS[size]
        rows.append(pd.DataFrame(dict(model=f"robojepa-{size}", size=size, step_k=k, compute_flops=pre * 1000 * k + cool * 5000, cond=cond,
                                      anchor_id=np.repeat(ids, K)[keep], cand_idx=np.tile(np.arange(K), N)[keep],
                                      score_neg_l1=s["l1"].reshape(-1)[keep], score_cos=s["cos"].reshape(-1)[keep], pred_err_l1=s["pred_err"].reshape(-1)[keep])))
    sc = pd.concat(rows, ignore_index=True)
    sc.to_parquet(f"{DST}/data/scores-{split}.parquet", index=False)
    print(split, "scores", len(sc), "rows", sc.groupby(["size", "step_k", "cond"]).ngroups, "checkpoints x conds", flush=True)
