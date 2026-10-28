"""Why is pairwise accuracy among goal-directed candidates below chance? For each anchor (goal-directed subset), correlate the
score with: final distance (-), commanded translation norm, overshoot = |cmd| - |to_goal_base|, commanded z component,
and the pixel change the action causes (proxy: |cmd|). Usage: bias_check.py <scores.npz> <anchors_dir>"""
import os, sys
import numpy as np
from scipy.stats import spearmanr, kendalltau
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from analyze import pairwise_acc

z = np.load(sys.argv[1], allow_pickle=True); ad = sys.argv[2]
rows = []
for i, name in enumerate(z["anchor"]):
    a = np.load(os.path.join(ad, str(name))); src = z["cand_source"][i]; g = src > 0
    if g.sum() < 8:
        continue
    s = z["l1"][i][g]; d = z["dist"][i][g]; c = a["cands"][g, :3]; tg = a["to_goal_base"] if "to_goal_base" in a else a["goal_pos"] - a["eef0"]
    n = np.linalg.norm(c, axis=1); over = n - np.linalg.norm(tg); mv = np.linalg.norm(a["eef_final"][g] - a["eef0"][None], axis=1)
    proj = c @ (tg / (np.linalg.norm(tg) + 1e-9))  # along-goal component
    perp = np.linalg.norm(c - proj[:, None] * (tg / (np.linalg.norm(tg) + 1e-9))[None], axis=1)
    ties = (np.abs(s[:, None] - s[None, :]) < 1e-7).sum() - len(s)
    rows.append([kendalltau(s, -d).correlation, pairwise_acc(s, d), ties / (len(s) * (len(s) - 1)), spearmanr(s, -d).correlation, spearmanr(s, n).correlation, spearmanr(s, over).correlation, spearmanr(s, c[:, 2]).correlation,
                 spearmanr(s, mv).correlation, spearmanr(s, proj).correlation, spearmanr(s, -perp).correlation, spearmanr(-d, n).correlation])
r = np.array(rows); names = ["kendall(s,-d)", "pairwise_acc", "tie_frac", "score~-dist", "score~|cmd|", "score~overshoot", "score~cmd_z", "score~moved", "score~along-goal", "score~-perp", "-dist~|cmd|"]
print(os.path.basename(sys.argv[1]), f"n={len(r)} anchors, goal-directed subset; mean spearman (s.e.):")
for j, nm in enumerate(names):
    print(f"  {nm:18s} {np.nanmean(r[:, j]):+.3f} ({np.nanstd(r[:, j])/np.sqrt(len(r)):.3f})")
