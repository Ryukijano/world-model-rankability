"""Rankability metrics (identical to the benchmark's probe/analyze.py). Score: higher = predicted better.
Per anchor: Spearman(score, -final_dist), AUROC(score vs final_dist < 5 cm), top-1 regret, Hit@1/3, pairwise accuracy
(score ties count 0.5); "fine" = the same within the 32 goal-directed candidates. Held-out ECE: within-anchor z-scored
score -> P(dist < 5 cm) by a 1-D logistic fit on even anchors, 10-bin ECE on odd anchors."""
import numpy as np
from scipy.stats import spearmanr, rankdata
from scipy.optimize import minimize
from scipy.special import expit


def auroc(score, pos):
    if pos.all() or (~pos).all():
        return np.nan
    r = rankdata(score); n1 = pos.sum(); n0 = len(pos) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pairwise_acc(score, dist):
    s = score[:, None] - score[None, :]; d = dist[None, :] - dist[:, None]
    m = d != 0
    return (((np.sign(s) == np.sign(d)) & m).sum() + 0.5 * ((s == 0) & m).sum()) / m.sum()


def anchor_metrics(score, dist):
    order = np.argsort(dist); top = np.argmax(score)
    return dict(spearman=spearmanr(score, -dist).correlation, auroc_5cm=auroc(score, dist < 0.05), regret_cm=100 * (dist[top] - dist.min()),
                hit1=float(top == order[0]), hit3=float(top in order[:3]), pairwise=pairwise_acc(score, dist))


def ece_heldout(scores, dists, tau=0.05, bins=10):
    z = [(s - s.mean()) / (s.std() + 1e-9) for s in scores]; y = [(d < tau).astype(float) for d in dists]
    fx, fy, ex, ey = np.concatenate(z[0::2]), np.concatenate(y[0::2]), np.concatenate(z[1::2]), np.concatenate(y[1::2])
    nll = lambda w: -np.mean(fy * np.log(expit(w[0] * fx + w[1]) + 1e-9) + (1 - fy) * np.log(1 - expit(w[0] * fx + w[1]) + 1e-9))
    w = minimize(nll, np.zeros(2), method="Nelder-Mead").x; p = expit(w[0] * ex + w[1]); ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (p >= lo) & (p < hi + (hi == 1))
        if m.any():
            ece += m.mean() * abs(p[m].mean() - ey[m].mean())
    return ece


def evaluate(score, dist, source, drop=None):
    """score, dist: [N_anchors, 64]; source: [N_anchors, 64] str in {random, goal_coarse, goal_fine}."""
    keep = np.ones(len(score), bool) if drop is None else ~drop
    S, D, G = score[keep], dist[keep], source[keep] != "random"
    allm = [anchor_metrics(s, d) for s, d in zip(S, D)]; fine = [anchor_metrics(s[g], d[g]) for s, d, g in zip(S, D, G)]
    rnd = [anchor_metrics(s[~g], d[~g]) for s, d, g in zip(S, D, G)]
    mean = lambda L, k: float(np.nanmean([q[k] for q in L])); se = lambda L, k: float(np.nanstd([q[k] for q in L]) / np.sqrt(len(L)))
    return {"n_anchors": int(keep.sum()), "overall_spearman": mean(allm, "spearman"), "overall_spearman_se": se(allm, "spearman"),
            "fine_spearman": mean(fine, "spearman"), "random_spearman": mean(rnd, "spearman"), "auroc_5cm": mean(allm, "auroc_5cm"),
            "regret_cm": mean(allm, "regret_cm"), "hit1": mean(allm, "hit1"), "hit3": mean(allm, "hit3"),
            "fine_pairwise": mean(fine, "pairwise"), "ece": float(ece_heldout(list(S), list(D)))}
