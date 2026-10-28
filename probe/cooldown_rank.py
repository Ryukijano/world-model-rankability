"""Action-discriminative cooldown pilot: fine-tune a RoboJEPA predictor on frozen cached encoder features with
  loss = w_rank * CE(softmax(score/T), softmax(-dist/tau_d)) + w_pred * L1(pred, real post-action features)
where score_k = -mean|pred_k - goal| (RoboJEPA's planning objective) over each anchor's 64 replayed candidates.
w_rank=0 is the compute-matched control (prediction loss only, same data/steps). Evaluates the benchmark metrics on
held-out pretrain kitchens (val) and the 10 unseen target kitchens (test) before training and every eval_every steps."""
import argparse, glob, hashlib, json, os, platform, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from score_anchors import R, WorldModel, WorldModelState  # noqa: E402
from rankability_metrics import evaluate  # noqa: E402

SRCNAME = np.array(["random", "goal_coarse", "goal_fine"])


def load_set(adir, fdir, scenes=None, per_scene=None):
    out, missing, count = [], [], {}
    for f in sorted(glob.glob(f"{adir}/anchor_*.npz")):
        d = np.load(f); s = int(d["scene"])
        if (scenes is not None and s not in scenes) or (per_scene and count.get(s, 0) >= per_scene):
            continue
        if not os.path.exists(f"{fdir}/{os.path.basename(f)}"):
            missing.append(f); continue
        c = np.load(f"{fdir}/{os.path.basename(f)}")
        out.append(dict(id=os.path.basename(f)[:-4], scene=s, init=torch.from_numpy(c["init"]), goal=torch.from_numpy(c["goal"]),
                        post=torch.from_numpy(c["post"]) if c["post"].ndim > 1 else None, pose=torch.from_numpy(d["init_pose"]).float(),
                        cands=torch.from_numpy(d["cands"]).float(), dist=d["dist"].astype(np.float64), src=SRCNAME[d["cand_source"]]))
        count[s] = count.get(s, 0) + 1
    if missing:
        print(f"WARNING {len(missing)} anchors in {adir} have no cached features and were skipped", flush=True)
    return out


def predict(wm, x, cfg, grad):
    vm = torch.tensor([cfg["camera_id"]]); emb = torch.tensor([[cfg["embodiment_id"]]])
    st = WorldModelState.from_features(features=x["init"][None, None].float(), states=x["pose"][None, None], view_map=vm, embodiment_ids=emb).to("cuda")
    K = len(x["cands"])
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred, _ = wm.step(x["cands"][:, None], st.repeat(K), use_cache=False, no_grad=not grad, batch=K, device="cuda")
    return pred.float().reshape(K, -1)


@torch.no_grad()
def run_eval(wm, data, cfg):
    S, D, SRC, pe = [], [], [], []
    for x in data:
        pred = predict(wm, x, cfg, grad=False).cuda(); g = x["goal"].float().reshape(1, -1).cuda()
        S.append((-(pred - g).abs().mean(-1)).cpu().numpy()); D.append(x["dist"]); SRC.append(x["src"])
        if x["post"] is not None:
            P = len(x["post"]); pe.append((pred[:P] - x["post"].float().reshape(P, -1).cuda()).abs().mean(-1).mean().item())
    m = evaluate(np.array(S), np.array(D), np.array(SRC)); m["pred_err_l1"] = float(np.mean(pe)) if pe else float("nan")
    return {k: float(v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(); cfg = json.load(open(a.config)); t0 = time.time()
    if a.dry_run:
        cfg.update(steps=2, eval_every=2, train_per_scene=1, val_per_scene=1, test_per_scene=1)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    out = cfg["out_dir"] + ("_dryrun" if a.dry_run else ""); os.makedirs(out, exist_ok=True)
    prov = dict(config_path=os.path.abspath(a.config), config=cfg, seed=cfg["seed"], slurm_job=os.environ.get("SLURM_JOB_ID"), host=platform.node(),
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), python=sys.version.split()[0],
                robo_jepa_sha=os.popen(f"git -C {R}/robo_jepa rev-parse HEAD").read().strip(),
                script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                config_sha256=hashlib.sha256(open(a.config, "rb").read()).hexdigest(), dry_run=a.dry_run)
    print("PROVENANCE " + json.dumps(prov), flush=True)
    tr = load_set(cfg["train_anchors"], cfg["train_feats"], set(cfg["train_scenes"]), cfg.get("train_per_scene"))
    va = load_set(cfg["train_anchors"], cfg["train_feats"], set(cfg["val_scenes"]), cfg.get("val_per_scene"))
    te = load_set(cfg["test_anchors"], cfg["test_feats"], None, cfg.get("test_per_scene"))
    assert tr and va and te, "empty split"
    print(f"data: train={len(tr)} val={len(va)} test={len(te)} anchors ({time.time()-t0:.0f}s)", flush=True)
    wm = WorldModel.load_pretrained(cfg["model_dir"], load_decoder=False, load_encoder=False, load_compiled=False, device="cuda", weights_filename=cfg["weights"])
    tm = wm.transition_model; params = [p for p in tm.parameters() if p.requires_grad or True]
    for p in params:
        p.requires_grad_(True)
    log_T = torch.nn.Parameter(torch.tensor(float(np.log(cfg["score_temp_init"]))), requires_grad=True)
    opt = torch.optim.AdamW([{"params": params, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}, {"params": [log_T], "lr": 1e-2, "weight_decay": 0.0}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, cfg["warmup"])) * max(0.0, 1 - s / cfg["steps"]))
    hist = []

    def do_eval(step):
        tm.eval(); r = dict(step=step, val=run_eval(wm, va, cfg), test=run_eval(wm, te, cfg), sec=round(time.time() - t0)); tm.train()
        hist.append(r); print("EVAL " + json.dumps(r), flush=True)
    do_eval(0)
    rng = np.random.default_rng(cfg["seed"]); tm.train()
    for step in range(1, cfg["steps"] + 1):
        idx = rng.choice(len(tr), cfg["anchors_per_step"], replace=False); lr_, lp_ = 0.0, 0.0
        opt.zero_grad(set_to_none=True)
        for i in idx:
            x = tr[i]; pred = predict(wm, x, cfg, grad=True).cuda(); g = x["goal"].float().reshape(1, -1).cuda()
            score = -(pred - g).abs().mean(-1)
            tgt = torch.softmax(-torch.from_numpy(x["dist"]).float().cuda() / cfg["tau_dist"], 0)
            l_rank = -(tgt * torch.log_softmax(score / log_T.exp(), 0)).sum()
            P = len(x["post"]); l_pred = (pred[:P] - x["post"].float().reshape(P, -1).cuda()).abs().mean()
            loss = (cfg["w_rank"] * l_rank + cfg["w_pred"] * l_pred) / len(idx); loss.backward()
            lr_ += l_rank.item() / len(idx); lp_ += l_pred.item() / len(idx)
        gn = torch.nn.utils.clip_grad_norm_(params, cfg["clip"]); opt.step(); sched.step()
        if step % cfg["log_every"] == 0 or a.dry_run:
            print(f"step {step} l_rank={lr_:.4f} l_pred={lp_:.4f} T={log_T.exp().item():.5f} gnorm={gn:.3f} lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if step % cfg["eval_every"] == 0:
            do_eval(step)
    ck = f"{out}/predictor_final.pt"; torch.save({"transition_model": tm.state_dict(), "log_T": log_T.detach(), "provenance": prov}, ck)
    json.dump(dict(provenance=prov, evals=hist, checkpoint=ck), open(f"{out}/metrics.json", "w"), indent=1)
    print(f"CHECKPOINT {ck}\nMETRICS {out}/metrics.json total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
