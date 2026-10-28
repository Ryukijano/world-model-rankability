"""leaderboard.csv: metrics for every (split, size, step, cond) in data/scores-*.parquet, using space/rankability_metrics.py."""
import sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "space"); from rankability_metrics import evaluate
D = sys.argv[1]


def one(args):
    split, key, g, cand = args
    g = g.sort_values(["anchor_id", "cand_idx"]); n = g.anchor_id.nunique(); c = cand[cand.anchor_id.isin(g.anchor_id.unique())].sort_values(["anchor_id", "cand_idx"])
    S = g.score_neg_l1.values.reshape(n, 64).astype(float); Dd = c.final_dist_m.values.reshape(n, 64); R = c.source.values.reshape(n, 64)
    m = evaluate(S, Dd, R); size, k, cond, model, C = key
    return dict(split=split, model=model, size=size, step_k=k, compute_flops=C, cond=cond, pred_err_l1=float(np.nanmean(g.pred_err_l1)), **m)


jobs = []
for split in ("pretrain", "target"):
    cand = pd.read_parquet(f"{D}/data/candidates-{split}.parquet", columns=["anchor_id", "cand_idx", "final_dist_m", "source"])
    sc = pd.read_parquet(f"{D}/data/scores-{split}.parquet")
    for key, g in sc.groupby(["size", "step_k", "cond", "model", "compute_flops"]):
        jobs.append((split, key, g, cand))
with ProcessPoolExecutor(8) as ex:
    rows = list(ex.map(one, jobs))
lb = pd.DataFrame(rows).sort_values(["split", "cond", "compute_flops"]); lb.to_csv(f"{D}/leaderboard.csv", index=False)
print(lb[lb.cond == "clean"].sort_values("step_k").groupby(["split", "size"]).tail(1)[["split", "size", "step_k", "n_anchors", "overall_spearman", "fine_spearman", "ece", "pred_err_l1"]].round(3).to_string())
