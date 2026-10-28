"""Rankability probe, stage 2: score a fixed candidate bank with a frozen RoboJEPA world model.

For every anchor .npz from collect_anchors.py: encode the init frames (per view, exactly as policy_client.py does),
build the WorldModelState (1-frame history by default), repeat it K times, roll each candidate one WM step and score
the imagined features against the encoded goal frames with mpc_utils.l1 (RoboJEPA's planning objective; cos logged
too). Writes one .npz with scores and the copied simulator labels. Nothing is trained; candidates are never changed.
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
from PIL import Image, ImageFilter

R = "/projects/u6xn/rankability"
sys.path[:0] = [f"{R}/robo_jepa", f"{R}/robo_jepa/scripts/deployment"]
from app.robo_jepa.world_model import WorldModel, WorldModelState, poses_to_diffs  # noqa: E402
from app.robo_jepa.mpc import mpc_utils  # noqa: E402


def perturb(img, cond, rng):
    """Observation-side shift axes (applied to the init frames only; goal frames stay clean).
    noise<s>: gaussian pixel noise, sigma s (0-255 scale); blur<r>: gaussian blur radius r px;
    shift<p>: crop p px from one side and resize back (camera shift); dark<f>: brightness x f/100."""
    if cond == "clean":
        return img
    kind, val = cond.rstrip("0123456789"), float(cond[len(cond.rstrip("0123456789")):] or 0)
    if kind == "noise":
        return np.clip(img.astype(np.float32) + rng.normal(scale=val, size=img.shape), 0, 255).astype(np.uint8)
    if kind == "blur":
        return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(val)))
    if kind == "shift":
        H, W = img.shape[:2]; p_ = int(val)
        return np.array(Image.fromarray(img[p_:, p_:]).resize((W, H)))
    if kind == "dark":
        return np.clip(img.astype(np.float32) * val / 100.0, 0, 255).astype(np.uint8)
    raise ValueError(cond)


def encode_views(wm, imgs, size, device, cond="clean", rng=None, views=None):
    feats = []
    for vi, img in enumerate(imgs):  # [H,W,3] uint8 per view
        if views is not None and vi not in views:
            continue
        img = perturb(img, cond, rng)
        img = np.array(Image.fromarray(img).resize((size[1], size[0])))
        feats.append(wm.encode(img[None], size=tuple(size), device=device, dtype=torch.float16)[:, None, None])
    return torch.cat(feats, dim=2).to(torch.float32)  # [1, T=1, V, h, w, C]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True); p.add_argument("--weights", required=True)
    p.add_argument("--encoder", default=f"{R}/ckpt/encoder/vjepa2_1_vitG_384.pt")
    p.add_argument("--anchors", required=True); p.add_argument("--out", required=True)
    p.add_argument("--embodiment_id", type=int, default=2); p.add_argument("--camera_id", type=int, nargs="+", default=[0, 2])
    p.add_argument("--image_size", type=int, nargs=2, default=[240, 320]); p.add_argument("--max_anchors", type=int, default=10**9)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--cond", default="clean", help="clean | noise<sigma> | blur<r> | shift<px> | dark<pct>; init frames only")
    p.add_argument("--views", type=int, nargs="+", default=None, help="subset of view indices to keep (e.g. 0 = agentview only)")
    p.add_argument("--state_key", default="init_pose", help="init_pose (7-D 'pose', what policy_client feeds embodiment 2) | init_proprio (14-D, state_delta variant)")
    p.add_argument("--shard", type=int, nargs=2, default=None, metavar=("I", "N"), help="score anchors[I::N] only (out gets __shardI-N suffix)")
    p.add_argument("--fp32", action="store_true", help="disable fp16 autocast for the predictor step (precision check)")
    p.add_argument("--history", default="last", help="last: 1-frame context | full: all hist frames, KV cache warmed with poses_to_diffs like policy_client")
    a = p.parse_args(); device = "cuda"; t0 = time.time()
    files = sorted(glob.glob(f"{a.anchors}/anchor_*.npz"))[: a.max_anchors]
    if a.shard:
        files = files[a.shard[0]::a.shard[1]]; a.out = a.out.replace(".npz", f"__shard{a.shard[0]}-{a.shard[1]}.npz")
    assert files, "no anchors"
    wm = WorldModel.load_pretrained(a.model_dir, load_decoder=False, load_encoder=True, load_compiled=False,
                                    encoder_path=a.encoder, device=device, weights_filename=a.weights)
    tm = wm.transition_model
    n_params = sum(p.numel() for p in tm.parameters())
    print(f"loaded {a.model_dir}/{a.weights}: predictor params={n_params/1e6:.1f}M state_tokens={tm.state_tokens} "
          f"view_map={'yes' if tm.view_map is not None else 'no'} load={time.time()-t0:.0f}s", flush=True)
    cam = a.camera_id if a.views is None else [a.camera_id[v] for v in a.views]
    view_map = torch.tensor([cam]) if tm.view_map is not None else None
    emb = torch.tensor([[a.embodiment_id]])
    out = {k: [] for k in ["anchor", "l1", "cos", "l1_view", "dist", "reward", "cand_source", "d0", "l1_init_goal", "mv_ratio_med", "pred_err"]}
    for f in files:
        d = np.load(f); t = time.time()
        rng = np.random.default_rng([int(d["scene"]), int(d["task_seed"]), 7])
        goal_f = encode_views(wm, d["goal_imgs"], a.image_size, device, "clean", None, a.views)
        sk = a.state_key if a.state_key in d else {"init_pose": "init_proprio"}.get(a.state_key, a.state_key)
        if a.history == "full" and "hist_imgs" in d and len(d["hist_imgs"]) > 1:
            frames = d["hist_imgs"]; poses_np = d["hist_pose"] if sk == "init_pose" else d["hist_proprio"]
        else:
            frames = d["init_imgs"][None]; poses_np = d[sk][None]
        feats = torch.cat([encode_views(wm, fr, a.image_size, device, a.cond, rng, a.views) for fr in frames], dim=1)  # [1,T,V,h,w,C]
        obs_f = feats[:, -1:]
        poses = torch.from_numpy(np.asarray(poses_np)).float()[None]  # [1,T,D]
        use_state = tm.state_tokens > 0
        K = len(d["cands"]); act = torch.from_numpy(d["cands"]).float()[:, None]  # [K, 1, 7]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=not a.fp32):
            if feats.shape[1] == 1:
                state = WorldModelState.from_features(features=obs_f, states=poses if use_state else None, view_map=view_map, embodiment_ids=emb).to(device)
                pred, _ = wm.step(act, state.repeat(K), use_cache=False, no_grad=True, batch=a.batch, device=device)
            else:  # policy_client warm-up: step through pose diffs with the cache, overwrite features/states with the real next frame
                ctx = poses_to_diffs(poses[0])[None] if sk == "init_pose" else (poses[:, 1:] - poses[:, :-1])
                state = WorldModelState.from_features(features=feats[:, :1], states=poses[:, :1] if use_state else None, view_map=view_map, embodiment_ids=emb).to(device)
                for k in range(ctx.shape[1]):
                    _, state = wm.step(ctx[:, k:k + 1], state, no_grad=True, device=device, use_cache=True, batch=a.batch)
                    state.features = feats[:, k + 1:k + 2].to(device)
                    if use_state:
                        state.states = poses[:, k + 1:k + 2].to(device)
                pred, _ = wm.step(act, state.repeat(K), use_cache=True, no_grad=True, batch=a.batch, device=device)
        pred = pred.float().reshape(K, -1).to(device); g = goal_f.reshape(1, -1).to(device)
        pe = np.full(K, np.nan, np.float32)  # imagination error: L1 between predicted and actually-observed post-action features
        if "post_imgs" in d and d["post_imgs"].ndim == 5:
            for j in range(len(d["post_imgs"])):
                tf = encode_views(wm, d["post_imgs"][j], a.image_size, device, "clean", None, a.views).reshape(1, -1).to(device)
                pe[j] = -float(mpc_utils.l1(tf, pred[j:j + 1]))
        l1 = mpc_utils.l1(g, pred).cpu().numpy(); cs = mpc_utils.cos(g, pred).cpu().numpy()
        V = goal_f.shape[2]; pv = pred.reshape(K, V, -1); gv = goal_f.to(device).reshape(1, V, -1)
        l1v = mpc_utils.l1(gv, pv).cpu().numpy()  # [K, V]
        l1_ig = float(mpc_utils.l1(g, obs_f.reshape(1, -1).to(device)))
        for k, v in [("anchor", os.path.basename(f)), ("l1", l1), ("cos", cs), ("l1_view", l1v), ("dist", d["dist"]),
                     ("reward", d["reward"]), ("cand_source", d.get("cand_source", np.zeros(K, np.int8))),
                     ("d0", float(d["d0"]) if "d0" in d else np.nan), ("l1_init_goal", l1_ig),
                     ("pred_err", pe), ("mv_ratio_med", float(np.median(np.linalg.norm(d["eef_final"] - d["eef0"][None], axis=1) / np.maximum(np.linalg.norm(d["cands"][:, :3], axis=1), 1e-6))))]:
            out[k].append(v)
        from scipy.stats import spearmanr
        rho = spearmanr(l1, -d["dist"]).correlation
        print(f"{os.path.basename(f)} K={K} spearman(l1,-dist)={rho:+.3f} l1[min/max]={l1.min():.4f}/{l1.max():.4f} "
              f"l1(init,goal)={l1_ig:.4f} top1_dist={d['dist'][np.argmax(l1)]:.3f} best_dist={d['dist'].min():.3f} {time.time()-t:.1f}s", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()},
                        meta=json.dumps(dict(vars(a), n_params=n_params, n_anchors=len(files), sec=time.time() - t0, state_key_used=sk, n_hist=int(frames.shape[0]),
                                             git_sha=os.popen(f"git -C {R}/robo_jepa rev-parse HEAD").read().strip())))
    print(f"wrote {a.out} anchors={len(files)} total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
