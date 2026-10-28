"""Real-video action understanding on the DROID holdout (REAL lab, setup room), frozen RoboJEPA checkpoints.

For each holdout clip, the world model predicts every next frame teacher-forced (true history frames, true current
pose) under a candidate action sequence; a candidate's error is the mean latent L1 between its predictions and the
real next-frame features over the clip's valid frames. The true actions should give the lowest error.
  * identification: true actions vs 15 action sequences from other holdout episodes (top-1, MRR; chance 1/16).
  * edits of the true actions, scored pairwise (P[true error < edited error]): translation x0.5 / x2 / sign-flipped,
    rotation zeroed, gripper delta sign-flipped (only clips where the gripper moves), actions delayed one step,
    all-zero actions.
Clips are the first --n-clips of eval_droid_holdout.py's fixed-seed order, so they are a subset of its 1,024."""
import argparse, json, os, shutil, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/projects/u6xn/rankability/robo_jepa/scripts/evals")
from eval_droid_holdout import (DTYPE, encode_clips, ensure_metadata, load_dir, embodiment_id, seed_all)  # noqa: E402
from app.robo_jepa.world_model import WorldModel  # noqa: E402

EDITS = ["trans_x0.5", "trans_x2", "trans_flip", "rot_zero", "grip_flip", "delay1", "zero",
         "trans_x0.75", "trans_x0.9", "trans_x1.1", "trans_x1.25", "noise0.25", "noise0.5", "noise1"]


def edit(a, kind):
    a = a.clone()
    if kind == "trans_x0.5": a[..., :3] *= 0.5
    elif kind == "trans_x2": a[..., :3] *= 2.0
    elif kind == "trans_flip": a[..., :3] *= -1
    elif kind == "rot_zero": a[..., 3:6] = 0
    elif kind == "grip_flip": a[..., 6] *= -1
    elif kind == "delay1": a = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], 1)
    elif kind == "zero": a = torch.zeros_like(a)
    elif kind.startswith("trans_x"): a[..., :3] *= float(kind[7:])
    elif kind.startswith("noise"):  # per-step Gaussian on translation, std = factor x the clip's mean per-step |translation|
        g = torch.Generator().manual_seed(0)
        scale = a[..., :3].norm(dim=-1).mean() / 3 ** 0.5
        a[..., :3] += float(kind[5:]) * scale * torch.randn(a[..., :3].shape, generator=g)
    return a


def errors(wm, clip, acts, device):
    """acts [N,T,7] -> per-candidate teacher-forced next-frame latent L1 [N]."""
    N = acts.shape[0]
    to = lambda x: x.to(device, dtype=DTYPE).repeat(N, *([1] * (x.dim() - 1)))  # noqa: E731
    state, ext, dt, pm = to(clip["state"]), to(clip["extrinsics"]), to(clip["dt"]), clip["padding_mask"].to(device)
    vl = clip["view_list"].to(device).repeat(N, 1)
    vf = clip["features"].to(device)
    if wm.encoder_layernorm:
        vf = F.layer_norm(vf, (vf.size(-1),))
    if vf.shape[1] == acts.shape[1] + 1:
        vf, pm = vf[:, :-1], pm[:, :-1]
    emb = torch.tensor([embodiment_id(wm, clip["embodiment"][0])] * N, device=device) if wm.model.is_multiembodiment else None
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
        pred, _, _, _ = wm.forward(vf.repeat(N, *([1] * (vf.dim() - 1))), acts.to(device, dtype=DTYPE), state, ext,
                                   view_map=vl, dt=dt, embodiment_ids=emb)
    assert pred.shape[1:] == vf.shape[1:], (pred.shape, vf.shape)
    e = (pred[:, :-1].float() - vf[:, 1:].float()).abs().flatten(2).mean(-1)  # [N, T-1]
    m = (pm[:, :-1] * pm[:, 1:]).float()  # [1, T-1]
    return ((e * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True, help="name=models_dir:size:checkpoint_path")
    p.add_argument("--data", required=True); p.add_argument("--encoder", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--n-clips", type=int, default=256); p.add_argument("--seed", type=int, default=2026); p.add_argument("--n-neg", type=int, default=15)
    a = p.parse_args(); device = "cuda"; torch.cuda.set_device(0); os.makedirs(a.out_dir, exist_ok=True)
    ensure_metadata(a.data)
    encoder, clips = None, None
    for spec in a.ckpts:
        name, rest = spec.split("=", 1); models_dir, size, ckpt = rest.split(":")
        out = os.path.join(a.out_dir, f"{name}.json")
        if os.path.exists(out):
            print(f"skip {name}"); continue
        t0 = time.time(); tmp = load_dir(models_dir, size, ckpt, a.encoder)
        try:
            wm = WorldModel.load_pretrained(tmp, load_decoder=False, load_encoder=encoder is None, device=device)
        finally:
            shutil.rmtree(tmp)
        if encoder is None:
            encoder = wm.encoder
            cache = os.path.join(a.out_dir, f"clips_n{a.n_clips}_s{a.seed}.pt")
            if os.path.exists(cache):
                clips = torch.load(cache, weights_only=False)
            else:
                clips, _ = encode_clips(encoder, a.data, a.n_clips, a.seed, device)
                torch.save(clips, cache)
            valid = np.array([int(c["padding_mask"].sum()) for c in clips]); epi = [c["id"].split("@")[0] for c in clips]
            A = torch.cat([c["action"].float().flatten(0, 1) for c in clips])
            nz = (A.abs() > 0).float().mean(0)
            print("action dims", A.shape[-1], "nonzero frac per dim", [round(float(x), 3) for x in nz[:10]],
                  "max|a| dims>=7", float(A[:, 7:].abs().max()), flush=True)
            assert float(A[:, 7:].abs().max()) == 0.0, "action layout is not [6 pose, 1 gripper, zero padding]"
        else:
            wm.encoder = encoder
        wm.eval(); wm.predictor_window_steps = 1
        rng = np.random.default_rng(a.seed)
        ranks, pair, grip_moves = [], {k: [] for k in EDITS}, []
        for i, c in enumerate(clips):
            pool = [j for j in range(len(clips)) if epi[j] != epi[i] and valid[j] >= valid[i]]
            negs = rng.choice(pool, size=min(a.n_neg, len(pool)), replace=False)
            true = c["action"].float()
            acts = torch.cat([true] + [clips[j]["action"].float() for j in negs] + [edit(true, k) for k in EDITS])
            err = errors(wm, c, acts, device)
            e_true, e_neg, e_edit = err[0], err[1:1 + len(negs)], err[1 + len(negs):]
            ranks.append(int(1 + (e_neg < e_true).sum()))
            moves = float(true[0, :valid[i] - 1, 6].abs().sum()) > 0.05
            grip_moves.append(moves)
            for k, v in zip(EDITS, e_edit):
                if k == "grip_flip" and not moves:
                    continue
                pair[k].append(float(e_true < v) + 0.5 * float(e_true == v))
        step_cm = float(np.mean([100 * c["action"][0, :valid[i] - 1, :3].float().norm(dim=-1).mean() for i, c in enumerate(clips)]))
        ranks = np.array(ranks)
        res = dict(name=name, checkpoint=ckpt, n_clips=len(clips), n_candidates=a.n_neg + 1,
                   top1=float((ranks == 1).mean()), mrr=float((1 / ranks).mean()), mean_rank=float(ranks.mean()),
                   chance_top1=1 / (a.n_neg + 1),
                   pairwise_true_beats_edit={k: dict(acc=float(np.mean(v)), n=len(v)) for k, v in pair.items()},
                   n_clips_gripper_moves=int(np.sum(grip_moves)), mean_step_translation_cm=step_cm, seconds=time.time() - t0)
        json.dump(res, open(out, "w"), indent=1); print(json.dumps(res), flush=True)
        del wm; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
