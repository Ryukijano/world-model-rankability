"""Rankability metrics from data/scores/*.npz (login-node safe: numpy/scipy only).
Per anchor: Spearman(score, -dist); AUROC(score vs success=dist<tau); top-1 regret = dist[argmax score]-min(dist);
Hit@1/@3 (argmax score within the top-1/top-3 true candidates); pairwise accuracy. Mean +- s.e. over anchors.
Predeclared tau: 0.05 m (primary) and 0.5*d0. Usage: python3 analyze.py data/scores/anchors_pilot_v0__*.npz
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr, rankdata
from scipy.optimize import minimize
from scipy.special import expit


def ece_heldout(scores, dists, tau=0.05, bins=10):
    """Calibration: map within-anchor z-scored score -> P(dist<tau) by 1-D logistic fit on even anchors, ECE on odd anchors."""
    z = [(s - s.mean()) / (s.std() + 1e-9) for s in scores]; y = [(d < tau).astype(float) for d in dists]
    fit_x, fit_y = np.concatenate(z[0::2]), np.concatenate(y[0::2]); ev_x, ev_y = np.concatenate(z[1::2]), np.concatenate(y[1::2])
    if fit_y.sum() == 0 or ev_y.sum() == 0:
        return np.nan, np.nan
    nll = lambda w: -np.mean(fit_y * np.log(expit(w[0] * fit_x + w[1]) + 1e-9) + (1 - fit_y) * np.log(1 - expit(w[0] * fit_x + w[1]) + 1e-9))
    w = minimize(nll, np.zeros(2), method="Nelder-Mead").x
    p = expit(w[0] * ev_x + w[1]); edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi + (hi == 1))
        if m.any():
            ece += m.mean() * abs(p[m].mean() - ev_y[m].mean())
    return ece, float(w[0])


def auroc(score, pos):
    if pos.all() or (~pos).all():
        return np.nan
    r = rankdata(score); n1 = pos.sum(); n0 = len(pos) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pairwise_acc(score, dist):
    s = score[:, None] - score[None, :]; d = dist[None, :] - dist[:, None]  # better = smaller dist
    m = d != 0  # ties in score count 0.5 (tau-b style); ties in outcome are excluded
    return (((np.sign(s) == np.sign(d)) & m).sum() + 0.5 * ((s == 0) & m).sum()) / m.sum()


def metrics(score, dist, d0, src=None, sel=None):
    if sel is not None:
        score, dist = score[sel], dist[sel]
    order = np.argsort(dist); top = np.argmax(score)
    m = dict(spearman=spearmanr(score, -dist).correlation, auroc_5cm=auroc(score, dist < 0.05),
             auroc_halfd0=auroc(score, dist < 0.5 * d0), regret=dist[top] - dist.min(), hit1=float(top == order[0]),
             hit3=float(top in order[:3]), pairwise=pairwise_acc(score, dist), top1_dist=dist[top], best_dist=dist.min(),
             frac_pos_5cm=(dist < 0.05).mean())
    return m


def main(paths):
    rows = []
    for p in sorted(paths):
        z = np.load(p, allow_pickle=True); meta = json.loads(str(z["meta"]))
        src = z["cand_source"] if "cand_source" in z else None
        stuck = (z["mv_ratio_med"] < 0.5) if "mv_ratio_med" in z.files else np.zeros(len(z["anchor"]), bool)  # arm barely moves for most random cands
        keep = ~stuck if os.environ.get("ANALYZE_DROP_STUCK") == "1" else np.ones(len(z["anchor"]), bool)
        subsets = [("all", None)] + ([("rnd", 0), ("goal", 12)] if src is not None and (src > 0).any() else [])
        for key in ["l1", "cos"]:
            for sname, sv in subsets:
                per = []
                for i in range(len(z["anchor"])):
                    if not keep[i]:
                        continue
                    sel = None if sv is None else ((src[i] == 0) if sv == 0 else (src[i] > 0))
                    if sel is not None and sel.sum() < 4:
                        continue
                    per.append(metrics(z[key][i], z["dist"][i], z["d0"][i], sel=sel))
                agg = {k: (np.nanmean([q[k] for q in per]), np.nanstd([q[k] for q in per]) / np.sqrt(len(per))) for k in per[0]}
                if sv is None and keep.sum() >= 10:
                    e, slope = ece_heldout([z[key][i] for i in range(len(keep)) if keep[i]], [z["dist"][i] for i in range(len(keep)) if keep[i]]); agg["ece"] = (e, slope)
                rows.append((p.split("/")[-1].replace(".npz", "") + ("" if sname == "all" else f" [{sname}]"), key, len(per), meta.get("n_params", 0) / 1e6, agg))
    print(f"(stuck anchors are kept unless ANALYZE_DROP_STUCK=1)")
    print(f"{'scores file':58s} {'obj':3s} {'n':>4s} {'Mpar':>6s} {'spearman':>14s} {'auroc5cm':>14s} {'regret_m':>14s} {'hit1':>6s} {'hit3':>6s} {'pairw':>6s} {'pos5cm':>6s} {'ece':>6s}")
    for name, key, n, mp, a in rows:
        f = lambda k: f"{a[k][0]:+.3f}±{a[k][1]:.3f}"
        print(f"{name:58s} {key:3s} {n:4d} {mp:6.0f} {f('spearman'):>14s} {f('auroc_5cm'):>14s} {f('regret'):>14s} "
              f"{a['hit1'][0]:6.2f} {a['hit3'][0]:6.2f} {a['pairwise'][0]:6.3f} {a['frac_pos_5cm'][0]:6.2f} {a.get('ece', (np.nan,))[0]:6.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or glob.glob("data/scores/*.npz"))
