"""Flow-matching vs deterministic predictor in frozen V-JEPA 2.1 latent space (same inputs as RoboJEPA: two-view ViT-G
feature maps [V=2,15,20,1664] + one canonical PandaOmron action chunk [4,12]); trained on cached RoboCasa365 pretrain-atomic
transitions (probe/cache_rc365.py), evaluated with the public rankability metrics on the benchmark anchors (val = held-back
pretrain kitchens, test = 10 unseen target kitchens) with RoboJEPA's planning score  score_k = -mean|z_hat_k - z_goal|.
  mode=flow : rectified-flow / conditional flow matching on the normalised residual d = (z1 - z0)/s;  x_tau = (1-tau) eps + tau d,
              v_theta(x_tau, tau | z0, a) ~ d - eps;  sampling = Euler with `sample_steps` steps, `samples` draws, score averaged.
              param=x : network predicts d (x-prediction) instead of d - eps; v = (d_hat - x_tau)/(1 - tau) at sampling time.
  mode=det  : identical network, tau fixed, L1 regression of d (the RoboJEPA objective at matched architecture/data/compute).
Resumable (checkpoint every ckpt_every steps + time budget) so 30-min chained jobs work; provenance printed like cooldown_rank.py."""
import argparse, glob, hashlib, json, math, os, platform, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from score_anchors import R  # noqa: E402  (sets up the robo_jepa import path)
from app.robo_jepa.embodiment_preprocessing import to_canonical12_from_droid7  # noqa: E402
from cooldown_rank import load_set  # noqa: E402
from rankability_metrics import evaluate  # noqa: E402

V, H, W, C = 2, 15, 20, 1664
NTOK = V * H * W


class RC365(torch.utils.data.Dataset):
    """Transitions (z0, a[4,12], z1) from cached episodes; feats are mmapped .npy so RAM stays small."""
    def __init__(self, d, episodes):
        self.items = []
        for e in episodes:
            m = np.load(f"{d}/ep{e:06d}.npz"); n = len(m["acts"])
            self.items += [(e, i) for i in range(n)]
        self.d = d; self.feats = {}; self.acts = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        e, i = self.items[k]
        if e not in self.feats:
            self.feats[e] = np.load(f"{self.d}/ep{e:06d}.feats.npy", mmap_mode="r"); self.acts[e] = np.load(f"{self.d}/ep{e:06d}.npz")["acts"]
        f = self.feats[e]
        return torch.from_numpy(np.ascontiguousarray(f[i])), torch.from_numpy(self.acts[e][i].reshape(-1)), torch.from_numpy(np.ascontiguousarray(f[i + 1]))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False); self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d)); nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, -1)
        h = modulate(self.n1(x), s1, sc1); x = x + g1[:, None] * self.attn(h, h, h, need_weights=False)[0]
        return x + g2[:, None] * self.mlp(modulate(self.n2(x), s2, sc2))


class Predictor(nn.Module):
    """DiT-style transformer over the 600 tokens of both views; conditioned on action (48-D) and flow time via adaLN-Zero.
    Input per token = [z0 (context), x_tau (noisy residual)] -> d; output = 1664-D velocity (flow) or residual (det)."""
    def __init__(self, d=384, depth=6, heads=6, act_dim=48):
        super().__init__()
        self.inp = nn.Linear(2 * C, d); self.pos = nn.Parameter(torch.zeros(1, NTOK, d)); nn.init.normal_(self.pos, std=0.02)
        self.act = nn.Sequential(nn.Linear(act_dim, d), nn.SiLU(), nn.Linear(d, d)); self.tim = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.out_n = nn.LayerNorm(d, elementwise_affine=False); self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d)); self.out = nn.Linear(d, C)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias); nn.init.zeros_(self.out_ada[1].weight); nn.init.zeros_(self.out_ada[1].bias)

    @staticmethod
    def t_embed(t, dim=256):
        half = dim // 2; fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * 1000 * fr[None]; return torch.cat([a.cos(), a.sin()], -1)

    def forward(self, z0, x, a, t):  # z0,x: [B,NTOK,C] (normalised), a: [B,48], t: [B]
        c = self.act(a) + self.tim(self.t_embed(t))
        h = self.inp(torch.cat([z0, x], -1)) + self.pos
        for b in self.blocks:
            h = b(h, c)
        s, sc = self.out_ada(c).chunk(2, -1)
        return self.out(modulate(self.out_n(h), s, sc))


class Model(nn.Module):
    def __init__(self, cfg, stats):
        super().__init__(); self.cfg = cfg; self.net = Predictor(cfg["d"], cfg["depth"], cfg["heads"])
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def norm(self, z):  # raw feats [B,V,H,W,C] -> [B,NTOK,C] standardised
        return ((z.float().reshape(z.shape[0], NTOK, C) - self.mu) / self.sd)

    def loss(self, z0, a, z1):
        z0n = self.norm(z0); d = (self.norm(z1) - z0n) / self.rs; B = z0.shape[0]
        if self.cfg["mode"] == "det":
            return F.l1_loss(self.net(z0n, torch.zeros_like(d), a, torch.ones(B, device=a.device)), d)
        tau = torch.rand(B, device=a.device) if self.cfg.get("tau_dist", "uniform") == "uniform" else torch.sigmoid(torch.randn(B, device=a.device))
        eps = torch.randn_like(d); x = (1 - tau)[:, None, None] * eps + tau[:, None, None] * d
        out = self.net(z0n, x, a, tau)
        return F.mse_loss(out, d) if self.cfg.get("param", "v") == "x" else F.mse_loss(out, d - eps)

    @torch.no_grad()
    def predict(self, z0, a, samples=1, steps=8, gen=None, zero_init=False):
        """Returns z1_hat [S,B,NTOK,C] in raw feature units. zero_init: start the ODE at x=0 instead of noise (one deterministic 'mean-path' sample)."""
        z0n = self.norm(z0); B = z0.shape[0]
        if self.cfg["mode"] == "det":
            d = self.net(z0n, torch.zeros_like(z0n), a, torch.ones(B, device=a.device))[None]
        else:
            out = []
            for _ in range(samples):
                x = torch.zeros_like(z0n) if zero_init else torch.randn(z0n.shape, device=a.device, generator=gen, dtype=z0n.dtype)
                for i in range(steps):
                    t = torch.full((B,), i / steps, device=a.device); o = self.net(z0n, x, a, t)
                    v = (o - x) / (1 - i / steps) if self.cfg.get("param", "v") == "x" else o  # x-prediction -> velocity
                    x = x + v / steps
                out.append(x)
            d = torch.stack(out)
        return (z0n[None] + d * self.rs) * self.sd + self.mu


def bench_actions(c7):  # [K,7] metric planner actions -> canonical [K,48] exactly like RoboJEPA's franka_omron preprocessing
    a12 = to_canonical12_from_droid7(c7.float()); return a12[:, None, :].expand(-1, 4, -1).reshape(len(c7), 48)


@torch.no_grad()
def run_eval(model, data, cfg, samples, steps, max_anchors=None, seed=0):
    model.eval(); S, D, SRC, pe, S1, SM, SZ = [], [], [], [], [], [], []
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    for x in data[:max_anchors]:
        K = len(x["cands"]); z0 = x["init"][None].cuda().expand(K, -1, -1, -1, -1); a = bench_actions(x["cands"]).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            zh = model.predict(z0, a, samples, steps, gen).float()  # [S,K,NTOK,C]
        g = x["goal"].float().reshape(1, 1, NTOK, C).cuda()
        sc = -(zh - g).abs().mean((-1, -2))  # [S,K]  expected distance (mean over samples of L1)
        scm = -(zh.mean(0) - g[0]).abs().mean((-1, -2))  # [K]  distance of the sample mean (posterior-mean estimate)
        if cfg["mode"] == "flow":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                zz = model.predict(z0, a, 1, steps, gen, zero_init=True).float()
            scz = -(zz[0] - g[0]).abs().mean((-1, -2))
        else:
            scz = sc[0]
        S.append(sc.mean(0).cpu().numpy()); S1.append(sc[0].cpu().numpy()); SM.append(scm.cpu().numpy()); SZ.append(scz.cpu().numpy()); D.append(x["dist"]); SRC.append(x["src"])
        if x["post"] is not None:
            P = len(x["post"]); pe.append((zh[:, :P].mean(0) - x["post"].float().reshape(P, NTOK, C).cuda()).abs().mean().item())
    m = evaluate(np.array(S), np.array(D), np.array(SRC)); m = {k: float(v) for k, v in m.items()}
    for nm, SS in (("1sample", S1), ("meanpred", SM), ("zeroinit", SZ)):
        mm = evaluate(np.array(SS), np.array(D), np.array(SRC)); m[f"spearman_{nm}"] = float(mm["overall_spearman"]); m[f"fine_spearman_{nm}"] = float(mm["fine_spearman"]); m[f"hit1_{nm}"] = float(mm["hit1"])
    m["pred_err_l1"] = float(np.mean(pe)) if pe else float("nan"); m["n_anchors"] = len(S); model.train(); return m


def feat_stats(ds, n=64, seed=0):
    idx = np.random.default_rng(seed).choice(len(ds), min(n, len(ds)), replace=False)
    z0 = torch.stack([ds[i][0] for i in idx]).float(); z1 = torch.stack([ds[i][2] for i in idx]).float()
    mu = z0.mean((0, 1, 2, 3)); sd = z0.std((0, 1, 2, 3)).clamp_min(1e-3); rs = (((z1 - z0) / sd).std()).clamp_min(1e-3)
    return dict(mu=mu, sd=sd, rs=rs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--dry_run", action="store_true"); ap.add_argument("--eval_only", action="store_true")
    a = ap.parse_args(); cfg = json.load(open(a.config)); t0 = time.time()
    sh = cfg.get("feat_shape")  # optional [V,H,W,C] override for a different frozen encoder's feature grid
    if sh:
        globals().update(V=sh[0], H=sh[1], W=sh[2], C=sh[3], NTOK=sh[0] * sh[1] * sh[2])
    if a.dry_run:
        cfg.update(steps=3, eval_every=3, ckpt_every=3, val_max=2, test_max=2, train_episodes=2, eval_samples=1, eval_steps=2, final_samples=1, final_steps=2)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    out = cfg["out_dir"] + ("_dryrun" if a.dry_run else ""); os.makedirs(out, exist_ok=True)
    prov = dict(config_path=os.path.abspath(a.config), config=cfg, seed=cfg["seed"], slurm_job=os.environ.get("SLURM_JOB_ID"), host=platform.node(),
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), python=sys.version.split()[0],
                robo_jepa_sha=os.popen(f"git -C {R}/robo_jepa rev-parse HEAD").read().strip(), script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                config_sha256=hashlib.sha256(open(a.config, "rb").read()).hexdigest(), dry_run=a.dry_run, start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("PROVENANCE " + json.dumps(prov), flush=True)
    eps = sorted(int(os.path.basename(f)[2:8]) for f in glob.glob(f"{cfg['train_feats']}/ep*.npz"))
    rng = np.random.default_rng(cfg["seed"]); rng.shuffle(eps); eps = eps[: cfg.get("train_episodes") or len(eps)]
    ds = RC365(cfg["train_feats"], eps); assert len(ds) > 0, "no training transitions"
    va = load_set(cfg["val_anchors"], cfg["val_feats"], set(cfg["val_scenes"]), cfg.get("val_per_scene"))
    te = load_set(cfg["test_anchors"], cfg["test_feats"], None, cfg.get("test_per_scene"))
    assert va and te, "empty eval split"
    print(f"data: train episodes={len(eps)} transitions={len(ds)} val={len(va)} test={len(te)} anchors ({time.time()-t0:.0f}s)", flush=True)
    ck_path = f"{out}/predictor_last.pt"; ck = torch.load(ck_path, map_location="cuda", weights_only=False) if os.path.exists(ck_path) else None
    stats = ck["stats"] if ck else feat_stats(ds, seed=cfg["seed"])
    model = Model(cfg, stats).cuda(); n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, cfg["warmup"])) * max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg["steps"])))))
    step, hist = 0, []
    if ck:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step = ck["step"]; hist = ck["hist"]
        print(f"RESUMED from {ck_path} at step {step}", flush=True)
    print(f"model: mode={cfg['mode']} d={cfg['d']} depth={cfg['depth']} params={n_par/1e6:.1f}M rs={float(model.rs):.4f}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch"], shuffle=True, num_workers=cfg.get("workers", 6), drop_last=True, pin_memory=True, persistent_workers=True)

    def save(final=False):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, hist=hist, stats={k: getattr(model, k).cpu() for k in ("mu", "sd", "rs")}, provenance=prov, cfg=cfg), ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)
        if final:
            torch.save(torch.load(ck_path, map_location="cpu", weights_only=False), f"{out}/predictor_final.pt")
        json.dump(dict(provenance=prov, evals=hist, step=step, checkpoint=ck_path, params=n_par), open(f"{out}/metrics.json", "w"), indent=1)

    def do_eval(tag, samples, steps, vmax, tmax):
        r = dict(step=step, tag=tag, samples=samples, flow_steps=steps, val=run_eval(model, va, cfg, samples, steps, vmax), test=run_eval(model, te, cfg, samples, steps, tmax), sec=round(time.time() - t0))
        hist.append(r); print("EVAL " + json.dumps(r), flush=True)

    if a.eval_only or step >= cfg["steps"]:
        if not a.eval_only and os.path.exists(f"{out}/predictor_final.pt"):
            print(f"FINAL_EXISTS {out}/predictor_final.pt (chain link after completion; nothing to do)", flush=True); return
        do_eval("final", cfg["final_samples"], cfg["final_steps"], None, None); save(final=True); return
    if step == 0 and cfg.get("eval_at_init", True):
        do_eval("init", cfg["eval_samples"], cfg["eval_steps"], cfg.get("val_max"), cfg.get("test_max"))
    model.train(); it = iter(dl); run_loss, nl = 0.0, 0
    while step < cfg["steps"]:
        if time.time() - t0 > cfg.get("time_budget", 1500):
            save(); print(f"TIME_BUDGET at step {step}; resubmit to continue", flush=True); return
        try:
            z0, act, z1 = next(it)
        except StopIteration:
            it = iter(dl); z0, act, z1 = next(it)
        z0, act, z1 = z0.cuda(non_blocking=True), act.cuda(non_blocking=True), z1.cuda(non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(z0, act, z1)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"]); opt.step(); sched.step(); step += 1
        run_loss += loss.item(); nl += 1
        if step % cfg["log_every"] == 0 or a.dry_run:
            print(f"step {step} loss={run_loss/nl:.4f} gnorm={gn:.3f} lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True); run_loss, nl = 0.0, 0
        if step % cfg["eval_every"] == 0 and step < cfg["steps"]:
            do_eval("periodic", cfg["eval_samples"], cfg["eval_steps"], cfg.get("val_max"), cfg.get("test_max"))
        if step % cfg["ckpt_every"] == 0:
            save()
    save(); do_eval("final", cfg["final_samples"], cfg["final_steps"], None, None); save(final=True)
    print(f"CHECKPOINT {out}/predictor_final.pt\nMETRICS {out}/metrics.json total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
