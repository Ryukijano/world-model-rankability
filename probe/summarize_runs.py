import json, sys
for n in sys.argv[1:]:
    try: m = json.load(open(f"/projects/u6xn/rankability/runs/{n}/metrics.json"))
    except Exception as e: print(n, "no metrics", e); continue
    print(n, "step", m["step"], "params", m["params"])
    for r in m["evals"]:
        for sp in ("val", "test"):
            v = r[sp]; tag = r["tag"]
            extra = " ".join(f"{k}={v[k]:.3f}" for k in sorted(v) if k.startswith(("spearman_", "fine_spearman_", "hit1_")))
            print(f"  {tag:8s} s={r['samples']} k={r['flow_steps']} {sp:4s} n={v['n_anchors']} rho={v['overall_spearman']:.3f} fine={v['fine_spearman']:.3f} pair={v['fine_pairwise']:.3f} auroc={v['auroc_5cm']:.3f} regret={v['regret_cm']:.1f} hit1={v['hit1']:.3f} ece={v['ece']:.3f} perr={v['pred_err_l1']:.3f} | {extra}")
