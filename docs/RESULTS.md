# Rankability probe: consolidated results (frozen RoboJEPA, RoboCasa Reach)

Status: evidence complete for the frozen-scorer stage. All numbers below come from `results/*.csv` on Isambard
(`/projects/u6xn/rankability/results/`), produced by `probe/scaling_plot.py` from the per-checkpoint score files; job IDs are in STATUS.md.
Protocol: PROBE_DESIGN.md. Setup: PandaOmron, cameras agentview_left + eye_in_hand (320x240), chunk 4 (one world-model step),
Reach with embodiment 2 and the 7-D pose token, 1 frame of context, Meta's planning objective (negative mean latent L1 to the goal frame),
K=64 fixed candidates per anchor (32 random, 16 coarse goal-directed, 16 fine goal-directed), ground truth by replaying every candidate
from a MuJoCo state snapshot (replay determinism <= 0.2 mm on the main bench).

## Sets

| set | anchors | kitchens | bank | purpose |
|---|---|---|---|---|
| v3 / v3pe pilot | 100 | 1 (pretrain scene 0) | 0.30 m reach cap (35% of goal-directed candidates clipped to identical actions) | first graded-bank pilot; v3pe adds post-action frames for imagination error |
| v4 pilot | 100 | 1 (same) | 0.18 m cap (clipping-free) | isolates the bank fix from kitchen diversity |
| target bench | 499 | 10 RoboCasa target kitchens (held out from RoboCasa pretraining) | 0.18 m | unseen-kitchen axis, full compute ladders |
| pretrain bench | 2,996 of 3,000 | 30 pretrain kitchens x 100 seeds | 0.18 m | main result, final checkpoints of all 8 sizes |

## Final-checkpoint table

See `results/final_ckpt_table.md` (generated). Headline rows, pretrain bench (s.e. of overall Spearman .003):

| size | overall ρ | fine ρ (within goal-directed) | random ρ | AUROC@5cm | regret | Hit@1 | fine pairwise | ECE | imagination err |
|---|---|---|---|---|---|---|---|---|---|
| 22M | .747 | .359 | .363 | .888 | 5.7 cm | 2.2% | .626 | .029 | .273 |
| 50M | .718 | .282 | .358 | .866 | 6.6 cm | 1.3% | .598 | .046 | .263 |
| 100M | .761 | .392 | .409 | .893 | 5.6 cm | 3.7% | .639 | .027 | .253 |
| 300M | .784 | .411 | .494 | .900 | 5.1 cm | 4.4% | .647 | .020 | .240 |
| 1B | .789 | .418 | .513 | .902 | 5.0 cm | 4.7% | .650 | .020 | .223 |
| 2B | .790 | .419 | .525 | .902 | 4.9 cm | 5.2% | .650 | .019 | .221 |
| 4B | .791 | .413 | .537 | .901 | 4.9 cm | 5.3% | .648 | .021 | .218 |
| 8B | .789 | .409 | .536 | .900 | 5.0 cm | 5.5% | .647 | .022 | .218 |

σ=25 pixel noise on both cameras: 22M .722 / fine .311 / ECE .038; 1B .762 / .390 / .026.
The target bench (499 anchors, unseen kitchens) gives the same ordering within s.e.: .758 / .730 / .773 / .792 / .799 / .797 / .800 / .799.

## Findings

1. **Scale helps, then saturates.** Overall ranking rises 22M → 300M (+.04) and is flat from 300M/1B to 8B (±.002 on n=2,996).
   Fine ranking among already-good candidates caps at ρ ≈ .42 / pairwise .65 / Hit@1 5% for every size ≥ 300M. Regret and AUROC saturate at 300M.
2. **Imagination error keeps improving past the ranking plateau.** Latent L1 on real post-action frames falls monotonically through 8B
   (.273 → .218, another −2.4% from 1B to 8B) while every ranking/calibration metric is flat. The same decoupling appears within each
   size's compute ladder once ranking has converged (v3pe: 48 checkpoints; target bench: 1B flat from 50k steps while its error keeps falling).
3. **Within-size compute does matter at small sizes on a well-posed bank.** On the 0.18 m bank, 22M and 100M improve to their last checkpoint
   (22M: .47 at 10k → .74 at 50k → .755 at 150k; fine .17 → .28 → .35), on both the single kitchen (v4) and the unseen kitchens. The apparent
   saturation at 50k in the v3 pilot was a bank artefact (see 5). 1B+ are flat from 50k steps on every set.
4. **Calibration improves with training and size up to 1B, then flattens.** Held-out ECE .07 (22M, 10k) → .029 (22M final) → .020 (300M–2B); 8B .022.
   Undertrained scorers are uninformative rather than over-confident.
5. **The candidate bank matters more than any model.** Replacing the 0.30 m cap (which clipped 35% of goal-directed candidates to identical
   actions, so the scorer saw duplicates and the pairwise metric was penalised for ties) with the 0.18 m cap raised fine ρ from .19–.22 to .35
   at every size on the same kitchen. Pairwise accuracy counts score ties as 0.5 (τ-b style) in all reported numbers.
6. **Robustness.** Pixel noise is the dominant axis (σ=25 costs .03–.05 on the benches, .15–.22 on the clipped pilot); dropping the wrist camera
   costs ~.10; blur, 50% darkening and a 16-px crop cost .01–.03 (v3 grid). On the benches 1B degrades slightly *less* than 22M under σ=25
   (−.027 vs −.025 on pretrain; −.056 vs −.044 on target), reversing the pilot's picture; the 1B > 22M ordering holds under noise.
7. **50M is an outlier**: below 22M on every metric at its final checkpoint and non-monotone along its own compute curve (dip at 40–60k).
   Reported as-is.
8. **History.** Two frames of context (Meta's KV-cache warm-up) never beat one frame (|Δ| ≤ .02, pilots).

## Verdict against the pre-registered rule (PROBE_DESIGN.md)

Not "flat" (the 22M → 1B gain is real: +.04 overall, +.06 fine, s.e. .003, replicated on unseen kitchens), so the pure negative result is out.
But the ranking ladder ends ~1.5 orders of magnitude of compute before the released loss ladder does, and the ceiling of the latent-L1 scorer
on this task is fine ρ ≈ .42 / Hit@1 ≈ 5% regardless of predictor size. This is the figure-1 shape: ranking quality saturates while imagination
loss continues to improve. The live question for the method stage is whether an action-discriminative cooldown objective lifts the ceiling
(or changes the exponent) rather than whether more predictor compute does.

## Caveats and what is not covered

- One task family (Reach, horizon 1, 0.18 m), one simulator, one scorer objective (L1; cosine logged, not analysed). 4 of 3,000 pretrain anchors
  were still collecting at scoring time (n = 2,996). 107 anchors have a barely-moving arm for most random candidates; sensitivity with those
  dropped is in `results/scaling_anchors_bench_v2_nostuck.csv`: n = 2,892, overall ρ rises by a uniform +.027–.029 at every size
  (22M .774, 300M .812, 1B .818, 2B .820, 4B .820, 8B .818; σ=25: 22M .749, 1B .790), fine ρ, AUROC, regret, Hit@1 and ECE move by ≤ .003.
  Ordering and the 300M/1B plateau are unchanged.
- Action-noise and embodiment-shift conditions are not run. Compute ladders on the pretrain bench are final checkpoints only (full ladders on the
  target bench and v4).
- Not claimed: that fixed-candidate ranking is new (ARC-Bench, AD-WM). New here: predictor-scale and compute axes on the released RoboJEPA
  ladder, held-out ECE, unseen-kitchen generalisation, the imagination-error vs ranking comparison on identical inputs.
- RoboJEPA weights are CC BY-NC-SA 4.0: a public demo/leaderboard Space is fine, derived checkpoints inherit NC-SA.
