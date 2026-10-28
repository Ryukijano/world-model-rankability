"""What breaks: per-anchor failure analysis of frozen RoboJEPA scorers on the pretrain bench (latent-L1 score)."""
import json, numpy as np
from scipy.stats import spearmanr
D = "/home/ubuntu/rankability/hf_export/"
A = np.load(D + "anchors_pretrain.npz")
dist, src, cands, eef_f, goal, d0, R, scene, tgb = (A[k] for k in
    ["dist", "cand_source", "cands", "eef_final", "goal_pos", "d0", "R_wb", "scene", "to_goal_base"])
SIZES = ["22m__150k", "50m__120k", "100m__150k", "300m__300k", "1b__300k", "2b__450k", "4b__450k", "8b__300k"]
print("cand_source counts per anchor:", {int(v): int((src[0] == v).sum()) for v in np.unique(src[0])})
out = {}
for s in SIZES:
    S = np.load(D + f"scores_pretrain__{s}__clean.npz")
    sc, ok = S["l1"], S["scored"]
    idx = np.where(ok)[0]
    best = sc[idx].argmax(1)
    regret = dist[idx, best] - dist[idx].min(1)
    # pairwise accuracy vs true outcome gap (all candidate pairs)
    gaps, accs = [], []
    bins = [0, .005, .01, .02, .04, .08, 1]
    hit = np.zeros(len(bins) - 1); tot = np.zeros(len(bins) - 1)
    for i in idx[::3]:
        d, v = dist[i], sc[i]
        dd = d[:, None] - d[None, :]; vv = v[:, None] - v[None, :]
        m = np.triu(np.ones_like(dd, bool), 1) & (np.abs(dd) > 1e-9)
        g = np.abs(dd[m]); corr = (np.sign(-dd[m]) == np.sign(vv[m])).astype(float) + 0.5 * (vv[m] == 0)
        b = np.digitize(g, bins) - 1
        np.add.at(hit, b, corr); np.add.at(tot, b, 1)
    # per-axis residual of the chosen candidate's final eef vs goal, in robot-base frame
    res_w = eef_f[idx, best] - goal[idx]
    res_b = np.einsum("nij,ni->nj", R[idx].transpose(0, 2, 1), res_w)
    oracle_b = np.einsum("nij,ni->nj", R[idx].transpose(0, 2, 1),
                         eef_f[idx, dist[idx].argmin(1)] - goal[idx])
    # bias toward small motions among random candidates (source 0)
    mag = np.linalg.norm(cands[idx, :, :3], axis=-1)
    rho_mag_score = np.nanmean([spearmanr(sc[i][src[i] == 0], -mag[k][src[i] == 0])[0] for k, i in enumerate(idx[::5])])
    rho_mag_true = np.nanmean([spearmanr(-dist[i][src[i] == 0], -mag[k][src[i] == 0])[0] for k, i in enumerate(idx[::5])])
    out[s] = dict(
        n=int(len(idx)), mean_regret_cm=float(100 * regret.mean()), frac_regret_gt5cm=float((regret > .05).mean()),
        pairwise_acc_by_gap_cm={f"{100*bins[j]:.1f}-{100*bins[j+1]:.0f}": float(hit[j] / tot[j]) for j in range(len(tot))},
        chosen_abs_residual_base_cm=dict(zip("xyz", (100 * np.abs(res_b).mean(0)).round(2).tolist())),
        oracle_abs_residual_base_cm=dict(zip("xyz", (100 * np.abs(oracle_b).mean(0)).round(2).tolist())),
        random_cands_rho_score_vs_small_motion=float(rho_mag_score), random_cands_rho_truth_vs_small_motion=float(rho_mag_true),
        regret_by_d0_cm={f"d0<{q:.2f}": float(100 * regret[(d0[idx] < q) & (d0[idx] >= p)].mean())
                         for p, q in zip([0, .05, .10, .15], [.05, .10, .15, 9])},
        regret_vertical_goal_cm=float(100 * regret[np.abs(tgb[idx, 2]) > np.linalg.norm(tgb[idx, :2], axis=1)].mean()),
        regret_horizontal_goal_cm=float(100 * regret[np.abs(tgb[idx, 2]) <= np.linalg.norm(tgb[idx, :2], axis=1)].mean()),
        _regret=regret.tolist(), _idx=idx.tolist())
# shared failures across sizes
reg = {s: dict(zip(out[s]["_idx"], out[s]["_regret"])) for s in SIZES}
common = sorted(set.intersection(*[set(r) for r in reg.values()]))
M = np.array([[reg[s][i] for i in common] for s in SIZES])
hard = (M > .05)
shared = {"n_common": len(common), "frac_hard_in_all_8": float(hard.all(0).mean()),
          "frac_hard_in_any": float(hard.any(0).mean()),
          "frac_hard_1b_also_hard_22m": float((hard[4] & hard[0]).sum() / max(hard[4].sum(), 1)),
          "spearman_regret_22m_vs_8b": float(spearmanr(M[0], M[7])[0])}
sc_by_scene = {int(k): float(100 * M[4][scene[common] == k].mean()) for k in np.unique(scene[common])}
for s in SIZES: out[s].pop("_regret"); out[s].pop("_idx")
res = dict(per_size=out, shared_failures=shared, regret_1b_by_scene_cm=sc_by_scene)
json.dump(res, open("/home/ubuntu/rankability/breakage/breakage.json", "w"), indent=1)
print(json.dumps(res, indent=1))
