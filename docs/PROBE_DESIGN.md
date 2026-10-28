# Rankability-vs-compute probe: design note (v0, 24 Oct 2026)

Question. RoboJEPA (2610.10515) shows latent imagination error and closed-loop planning success both
improve predictably with training compute. ARC-Bench (2609.05461) shows frozen JEPA-WM scorers misrank
fixed action candidates and that closed-loop replanning hides it. Both can be true. The open measurement:
**does per-state action rankability (and its calibration, and its robustness to shift) scale with
predictor compute, and with the same exponent as prediction error?**

## Prior-work table (what each measures)

| Work | Scorer | Scale axis | Ranking metrics | Calibration | Shift axes | Closed loop |
|---|---|---|---|---|---|---|
| RoboJEPA 2610.10515 | RoboJEPA 22M-8B | 8 sizes x compute | none (planning success only) | no | no | CEM, RoboCasa + Franka |
| ARC-Bench 2609.05461 | DINO-WM/JEPA-WM ckpts; encoder swap DINOv2 / V-JEPA1-L / V-JEPA2-L / V-JEPA2-G | encoder only | top-1 regret, Hit@1/3, MRR, pairwise acc, per-anchor Spearman, wrong-anchor rate, Mirage rate | no | no | replanning-frequency ablation |
| AD-WM 2609.30264 | own JEPA + action-recovery reg. | no | whole-bank ranking vs CEM-elite regret | no | no | OGBench-Cube |
| Aim Short to Reach Far 2609.30036 | frozen LeWM | no | action ranking under anchored planning | no | no | yes |
| This probe | RoboJEPA ladder (8 sizes x ~6 ckpts each, every-step release) | **predictor size x training steps** | ARC-Bench set + AUROC(success) | **ECE / reliability of P(success) from score** | **history truncation, obs noise, camera shift, action noise, embodiment (DROID vs RoboCasa)** | later, only to confirm masking |

Differentiators: scale axis on the predictor (not encoder), calibration, controlled shift, embodiment.
Not claimed: first rankability audit (ARC-Bench), first action-discriminative objective (AD-WM).

## Protocol (frozen, no training)

- **Anchors**: N = 3,000 states from RoboJEPA's 4 RoboCasa tasks (Reach, ObjectReach, ObstacleReach,
  ObjectPush; ~750 each), sampled from the official env server at varying episode progress; each anchor
  stores the full observation history (so history truncation is a post-hoc perturbation), sim state, goal image.
- **Candidates**: K = 64 action chunks per anchor (horizon = RoboJEPA planning horizon, 8 or 16 steps):
  16 from CEM iteration-0 Gaussian (RoboJEPA's own proposal), 16 from a diffusion proposal
  (OpenWAM-alpha or LeRobot diffusion policy if an aarch64 install is clean; else skip and say so),
  16 expert-perturbed (scripted controller + Gaussian noise at 3 levels), 16 uniform in the maxnorm ball.
  Candidates are fixed once and SHA-hashed (ARC-Bench no-leak rule).
- **Ground truth**: execute each candidate open-loop in the simulator from the anchor's sim state;
  record true cost (task distance after the chunk) and binary success (task predicate within tolerance).
  3,000 x 64 = 192k short rollouts, CPU-only, MuJoCo; estimated 10-20 CPU-h total on Grace cores.
- **Score**: RoboJEPA objective exactly as in `mpc_utils.cem`: negative mean L1 between imagined last-step
  features and goal features (also log cosine). One forward per candidate: 192k x (sizes x ckpts) imagined rollouts.
- **Metrics per anchor, then aggregated with 95% bootstrap CIs**: Spearman(score, -true cost); AUROC(score -> success);
  top-1 regret; Hit@1/3; wrong-anchor rate; pairwise accuracy. **Calibration**: fit a one-parameter Platt map
  score -> P(success) on a disjoint anchor split, report ECE (15 bins) and reliability curves on the held-out split.
  **Baseline scorers**: random, true-cost oracle, encoder-only (goal distance of the *current* frame, no predictor),
  and the 22M checkpoint as the "small" control.
- **Shift axes** (applied to observations before encoding; candidates and labels unchanged):
  history truncation (full / 4 / 1 frame), Gaussian + blur sensor noise (2 levels), camera shift (render from
  a second RoboCasa camera / crop-shift), action noise added to executed chunk (labels recomputed),
  embodiment: repeat the protocol on DROID held-out clips using the released DROID L1 eval as proxy where sim is absent.
- **Scaling fit**: for each metric M, fit M(C) = a - b C^-alpha vs training compute C (FLOPs from the released
  configs: params x tokens), compare alpha_rank to alpha_pred (L1 on the same anchors). Report whether the
  ranking curve saturates while L1 keeps falling.

## Kill / go rules (pre-registered)
- KILL (claim dead): AUROC and Spearman flat across the 22M -> 8B ladder (CI-overlapping) under clean conditions
  AND perturbations do not change the ordering. Write it up as a negative result (ARC-Bench extension to scale).
- GO (method stage): any metric moves monotonically with compute, or shift sensitivity changes with scale.
  Then: action-discriminative cooldown (AD-WM-style) on 3 sizes using RoboJEPA's released cooldown recipe to test
  whether it changes alpha_rank. That is the compute-necessary part (est. 1-3k GH200-h, to be measured first).

## Compute accounting (probe)
- Scoring: 192k candidates x 8 sizes x 6 ckpts = 9.2M imagined rollouts. At the day-1 measured throughput
  (fill in from job 7676876) the 8B model dominates; budget target < 50 GH200-h; 4B/8B may use 2 ckpts each.
- Encoding: V-JEPA 2.1 ViT-G/384 features cached once per anchor x perturbation (3,000 x ~8 variants).
- Jobs stay <= 30 min, 1 node; the ladder runs as a job array over (size, ckpt).

## Licence
RoboJEPA code/weights CC BY-NC-SA 4.0: Space and released probe data non-commercial, share-alike. Probe code itself: Apache-2.0.

## Open items
- aarch64 install of mujoco 3.3.1 / robosuite 1.5.2 / robocasa 1.0.1 (job 7676876 checks wheels; EGL untested).
- RoboCasa kitchen assets ~23 GB download (login/data-mover, not inside a job).
- Diffusion proposal source on aarch64 (optional for v0).
- User's earlier V-JEPA 2-AC flat-energy result: locate for the motivation figure.
