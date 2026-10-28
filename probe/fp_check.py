"""Precision/determinism check: compare fp16 repeat runs and fp32 scores of the same anchors (files from 15_fp32_check)."""
import sys
import numpy as np
from scipy.stats import spearmanr
a, b, c = (np.load(p) for p in sys.argv[1:4])
print("fp16 repeat max|dl1| %.2e | fp16 vs fp32 max|dl1| %.2e | l1 scale %.3f" % (np.abs(a["l1"] - b["l1"]).max(), np.abs(a["l1"] - c["l1"]).max(), np.abs(a["l1"]).mean()))
src = a["cand_source"]; g = src > 0
print("l1 spread: overall std %.5f | within-goal std %.5f | goal-minus-rnd mean gap %.5f" % (
    a["l1"].std(), np.mean([a["l1"][i][g[i]].std() for i in range(len(src))]), np.mean([a["l1"][i][g[i]].mean() - a["l1"][i][~g[i]].mean() for i in range(len(src))])))
for name, z in [("fp16", a), ("fp32", c)]:
    print(name, "within-goal spearman per anchor", [round(float(spearmanr(z["l1"][i][g[i]], -z["dist"][i][g[i]]).correlation), 2) for i in range(len(src))])
print("fp16 vs fp32 rank agreement (spearman of scores) per anchor", [round(float(spearmanr(a["l1"][i], c["l1"][i]).correlation), 3) for i in range(len(src))])
