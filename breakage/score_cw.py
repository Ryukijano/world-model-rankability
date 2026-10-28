"""Change-weighted rescoring: same frozen RoboJEPA rollout as score_anchors.py (clean, 1-frame history), but the
latent-L1-to-goal score is also computed on only the patches that change most between the current frame and the goal
(top q per view, q in 10/25/50%), and on the patches the predictor itself moves most (top 25% of mean |pred-obs|)."""
import argparse, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/projects/u6xn/rankability/probe")
from score_anchors import encode_views, WorldModel, WorldModelState  # noqa: E402


def masked_score(g, pred, mask):  # g [V,P,C], pred [K,V,P,C], mask [V,P] bool -> [K]
    e = (pred - g[None]).abs().mean(-1)  # [K,V,P]
    m = mask[None].float()
    return -(e * m).sum((1, 2)) / m.sum()


def topq(x, q):  # x [V,P] -> bool mask of top q fraction per view
    k = max(1, int(round(q * x.shape[1])))
    thr = x.topk(k, dim=1).values[:, -1:]
    return x >= thr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True); p.add_argument("--weights", required=True)
    p.add_argument("--encoder", default="/projects/u6xn/rankability/ckpt/encoder/vjepa2_1_vitG_384.pt")
    p.add_argument("--anchors", required=True); p.add_argument("--out", required=True)
    p.add_argument("--embodiment_id", type=int, default=2); p.add_argument("--camera_id", type=int, nargs="+", default=[0, 2])
    p.add_argument("--image_size", type=int, nargs=2, default=[240, 320]); p.add_argument("--batch", type=int, default=100)
    p.add_argument("--shard", type=int, nargs=2, required=True)
    a = p.parse_args(); device = "cuda"; t0 = time.time()
    files = sorted(glob.glob(f"{a.anchors}/anchor_*.npz"))[a.shard[0]::a.shard[1]]
    a.out = a.out.replace(".npz", f"__shard{a.shard[0]}-{a.shard[1]}.npz")
    wm = WorldModel.load_pretrained(a.model_dir, load_decoder=False, load_encoder=True, load_compiled=False,
                                    encoder_path=a.encoder, device=device, weights_filename=a.weights)
    tm = wm.transition_model
    view_map = torch.tensor([a.camera_id]) if tm.view_map is not None else None
    emb = torch.tensor([[a.embodiment_id]])
    keys = ["l1", "cw_goal10", "cw_goal25", "cw_goal50", "cw_pred25"]
    out = {k: [] for k in ["anchor", "dist", "cand_source"] + keys}
    for f in files:
        d = np.load(f)
        goal_f = encode_views(wm, d["goal_imgs"], a.image_size, device)  # [1,1,V,h,w,C]
        obs_f = encode_views(wm, d["init_imgs"], a.image_size, device)
        V, C = goal_f.shape[2], goal_f.shape[-1]
        poses = torch.from_numpy(np.asarray(d["init_pose"])).float()[None, None]
        K = len(d["cands"]); act = torch.from_numpy(d["cands"]).float()[:, None]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            state = WorldModelState.from_features(features=obs_f, states=poses if tm.state_tokens > 0 else None,
                                                  view_map=view_map, embodiment_ids=emb).to(device)
            pred, _ = wm.step(act, state.repeat(K), use_cache=False, no_grad=True, batch=a.batch, device=device)
        pred = pred.float().reshape(K, V, -1, C).to(device)
        g = goal_f.to(device).reshape(V, -1, C); o = obs_f.to(device).reshape(V, -1, C)
        full = -(pred.reshape(K, -1) - g.reshape(1, -1)).abs().mean(-1)
        gchange = (g - o).abs().mean(-1)  # [V,P]
        pchange = (pred - o[None]).abs().mean(-1).mean(0)
        sc = {"l1": full, "cw_goal10": masked_score(g, pred, topq(gchange, .10)), "cw_goal25": masked_score(g, pred, topq(gchange, .25)),
              "cw_goal50": masked_score(g, pred, topq(gchange, .50)), "cw_pred25": masked_score(g, pred, topq(pchange, .25))}
        out["anchor"].append(os.path.basename(f)); out["dist"].append(d["dist"]); out["cand_source"].append(d["cand_source"])
        for k in keys:
            out[k].append(sc[k].cpu().numpy())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()}, meta=json.dumps(dict(vars(a), n=len(files), sec=time.time() - t0)))
    print(f"wrote {a.out} anchors={len(files)} total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
