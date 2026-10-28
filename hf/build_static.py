"""Static-Space assets: leaderboard.json, cands_{split}.json (dist/source/final-ckpt scores per anchor), previews/{anchor}.jpg."""
import json, os, sys
import numpy as np, pandas as pd
from PIL import Image
SRC, DS, OUT = sys.argv[1:4]


def write_trained_leaderboard():
    trained = pd.read_csv(f"{DS}/leaderboard_trained.csv").round(5)
    records = json.loads(trained.to_json(orient="records"))
    json.dump(records, open(f"{OUT}/leaderboard_trained.json", "w"))


if sys.argv[4:] == ["--trained-only"]:
    write_trained_leaderboard()
    print("leaderboard_trained.json", os.path.getsize(f"{OUT}/leaderboard_trained.json"), "bytes")
    sys.exit(0)

lb = pd.read_csv(f"{DS}/leaderboard.csv"); json.dump(lb.round(5).to_dict("records"), open(f"{OUT}/leaderboard.json", "w"))
write_trained_leaderboard()
SIZES = ["22m", "50m", "100m", "300m", "1b", "2b", "4b", "8b"]
for split in ("pretrain", "target"):
    z = dict(np.load(f"{SRC}/anchors_{split}.npz")); ids = [a.replace(".npz", "") for a in z["anchor"]]
    sc = pd.read_parquet(f"{DS}/data/scores-{split}.parquet"); fin = lb[(lb.split == split) & (lb.cond == "clean")].sort_values("step_k").groupby("size").tail(1)
    scores = {}
    for _, r in fin.iterrows():
        s = sc[(sc["size"] == r["size"]) & (sc.step_k == r.step_k) & (sc.cond == "clean")].sort_values(["anchor_id", "cand_idx"])
        assert list(s.anchor_id.unique()) == ids
        scores[r["size"]] = np.round(s.score_neg_l1.values.reshape(len(ids), 64), 5).tolist()
    d = dict(ids=ids, scene=z["scene"].astype(int).tolist(), d0=np.round(z["d0"], 4).tolist(), dist=z["dist"].astype(np.float64).tolist(),
             src=z["cand_source"].astype(int).tolist(), sizes=[s for s in SIZES if s in scores], scores=scores)
    json.dump(d, open(f"{OUT}/cands_{split}.json", "w"), separators=(",", ":"))
    for i, a in enumerate(ids):
        g = np.concatenate([np.concatenate([z["init_imgs"][i, 0], z["goal_imgs"][i, 0]], 1), np.concatenate([z["init_imgs"][i, 1], z["goal_imgs"][i, 1]], 1)], 0)
        Image.fromarray(g).resize((480, 360), Image.LANCZOS).save(f"{OUT}/previews/{split}_{a}.jpg", quality=78)
    print(split, len(ids), os.path.getsize(f"{OUT}/cands_{split}.json") // 1024, "KB", flush=True)
