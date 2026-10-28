"""Compare whole-frame latent L1 with change-weighted variants on the pretrain bench (outputs of score_cw.py)."""
import glob, json, sys, numpy as np
from scipy.stats import spearmanr
D = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/rankability/breakage/scores_cw"
KEYS = ["l1", "cw_goal10", "cw_goal25", "cw_goal50", "cw_pred25"]
res = {}
for sz in ["22m", "1b", "8b"]:
    fs = sorted(glob.glob(f"{D}/anchors_bench_v2__wm-{sz}__*__shard*-8.npz"))
    if len(fs) < 8:
        print(sz, "shards", len(fs)); continue
    S = [np.load(f) for f in fs]
    cat = {k: np.concatenate([s[k] for s in S]) for k in ["anchor", "dist", "cand_source"] + KEYS}
    o = np.argsort(cat["anchor"]); cat = {k: v[o] for k, v in cat.items()}
    ref = np.load(f"/home/ubuntu/rankability/hf_export/scores_pretrain__{sz}__{'150k' if sz=='22m' else '300k'}__clean.npz")["l1"]
    src = cat["cand_source"]; print(sz, "n", len(o), "src counts", {int(v): int((src[0] == v).sum()) for v in np.unique(src[0])},
                                    "max|l1-ref|", float(np.abs(cat["l1"] - ref).max()))
    dist = cat["dist"]; r = {}
    for k in KEYS:
        sc = cat[k]
        ov = np.nanmean([spearmanr(sc[i], -dist[i])[0] for i in range(len(sc))])
        fi = np.nanmean([spearmanr(sc[i][src[i] != 0], -dist[i][src[i] != 0])[0] for i in range(len(sc))])
        best = sc.argmax(1); regret = dist[np.arange(len(sc)), best] - dist.min(1)
        bins = [0, .005, .01, .02, .04, .08, 1]; hit = np.zeros(6); tot = np.zeros(6)
        for i in range(0, len(sc), 3):
            dd = dist[i][:, None] - dist[i][None]; vv = sc[i][:, None] - sc[i][None]
            m = np.triu(np.ones_like(dd, bool), 1) & (np.abs(dd) > 1e-9)
            c = (np.sign(-dd[m]) == np.sign(vv[m])) + 0.5 * (vv[m] == 0); b = np.digitize(np.abs(dd[m]), bins) - 1
            np.add.at(hit, b, c); np.add.at(tot, b, 1)
        r[k] = dict(overall_rho=round(float(ov), 4), fine_rho=round(float(fi), 4), regret_cm=round(float(100 * regret.mean()), 2),
                    pairwise_by_gap=[round(float(x), 3) for x in hit / tot])
        print(f"  {k:10s} overall {ov:.3f} fine {fi:.3f} regret {100*regret.mean():.2f}cm pair-by-gap {r[k]['pairwise_by_gap']}")
    res[sz] = r
json.dump(res, open(f"{D}/../cw_results.json", "w"), indent=1)
