# Flow predictor in V-JEPA 2.1 space — working plan (2026-10-25, user-approved direction)

Hypothesis: RoboJEPA's deterministic L1 regressor (conditional mean) is the fine-ranking ceiling (fine rho~.42, Hit@1~5% for all sizes >=1B).
Test: same frozen encoder + same inputs (init feats [V,h,w,C], 7-D pose, 12-D action chunk), stochastic predictor via conditional flow matching
(Flow-JEPA 2608.29029 / "Reconstruction or Semantics?" NeurIPS26 semantic-latent DiT). Score candidate k by E_samples[-mean|z_hat - z_goal|] (and
 goal log-likelihood variant). Compare against: RoboJEPA 22M frozen, RoboJEPA 22M cooldown control (pred-only) and rank-cooldown, at matched data.

## Data (public, MIT): ember-lab-berkeley/robocasa365-pretrain-atomic (LeRobot v3, 7,356 eps, 1.495M frames @20fps, 4.24 GB total:
 mp4 4.07 GB in 22 files (3 cams x chunk files of 200 MB), parquet 0.17 GB). Cameras: robot0_agentview_left / _right / _eye_in_hand 256x256.
 action float64[12] = base_motion(4)+control_mode(1)+eef_delta_pos(3)+eef_delta_aa(3)+gripper(1); state float64[16] = base_pos(3)+base_quat(4)+eef_pos_rel(3)+eef_quat_rel(4)+gripper_qpos(2).
 target-atomic (9,126 eps, 6.23 GB) is in the 10 held-out kitchens = our test kitchens -> do NOT train on it.
 Bench uses robot0_agentview_left + robot0_eye_in_hand at 320x240; RoboCasa365 is 256x256 -> resize to bench preprocessing in encode_views.
 Pose token for bench = 7-D pose (eef pos+quat?); RoboCasa365 state has eef_pos_rel(3)+eef_quat_rel(4) -> check against score_anchors pose definition.
 Action mapping: bench candidate = 4-step chunk of eef delta (arm frame, base frame transform) -> RoboCasa365 eef_delta_pos/aa per 20fps frame;
 need to verify bench chunk semantics (control freq, chunk=4) vs demo frames; RoboJEPA embodiment 2/franka_omron action preprocessing handles 12-D?

## Cluster state: venv at /projects/u6xn/rankability/venv has lerobot 0.3.3, opencv-headless, imageio_ffmpeg, pandas (inside pytorch_26.09 SIF).
 Caches: /scratch/u6xn/ryukijano.u6xn/rank_feats/{pretrain 3000,target 499} 40 GB fp16 (60-day purge). Ckpts /projects/u6xn/rankability/ckpt/wm-*.
 Download pattern used before: transfer-only 30-min sbatch with wget -c (06_fetch_ladder.sbatch). huggingface_hub not on login python3.11; use venv in container.

## Steps
1. 23_fetch_rc365.sbatch: hf download pretrain-atomic (meta + parquet + agentview_left/eye_in_hand mp4 only, ~2.7 GB) into $PROJECTDIR/rankability/data/rc365. Resumable.
2. probe/cache_rc365.py: sample transitions (t, t+chunk) from episodes, encode init/goal frames with frozen ViT-G via score_anchors.encode_views -> fp16 npz shards on scratch. Pilot 500 transitions, then ~50k.
3. probe/flow_predictor.py: small transformer/DiT over V-JEPA tokens conditioned on (z_t, pose, action, t_flow); CFM loss; also deterministic-L1 twin at same size for control.
   Eval = run_eval from cooldown_rank.py with sample-averaged scores on val (5 pretrain kitchens) + 499 target anchors; log provenance like cooldown_rank.
4. Dry run -> <=500-sample pilot -> 30-min chained runs. Config JSON per run; git SHA in log.
Child session planning AIRR portfolio: 412f88cc0ddd46b78af5422a825701cc (notify on settle).

## Run log (2026-10-25)
- 7711081 fetch RoboCasa365 pretrain-atomic (meta, data, 2 cameras) -> /projects/u6xn/rankability/data/rc365/pretrain-atomic (2.8 GB, 16 mp4)
- 7711100 cache smoke: 10 episodes, --max_kept 16 -> rc365_feats/smoke10 (ALL_CACHED, 49 s)
- 7711105-08 cache sub500: 500 seeded episodes, 4 shards -> rc365_feats/sub500 (ALL_CACHED x4, ~1,100 s each, 45 GB, 23,324 transitions)
- hand-check (probe/handcheck_rc365.py on sub500): 0 bad files; feats [T,2,15,20,1664], acts [T-1,4,12], frames stride 4; all action dims in [-1,1]; torso always 0; base motion non-zero in 7% of sub-steps; ctrl_mode mean -0.83; grip +-1; 192 task strings. Latent |z1-z0| mean .14 (one chunk), |z3-z0| .27.
- caveat: RC365 chunks have 4 distinct 20 Hz actions; bench candidates are one 12D action repeated 4x (RoboJEPA's own convention), so the eval action distribution is narrower than training.
- 7711102/03, 7711281/82, 7711445/46: dry runs failed on config path (relative path / realpath resolving /projects to /lus, which is not bound in the container); fixed sbatch to use $R/<relative>.
- 7711511/12: dry run reached final eval; KeyError on metric key ('spearman' -> 'overall_spearman'); fixed.
- 7711758/59: dry run complete through final eval; torch.load weights_only failure when writing predictor_final.pt; fixed (weights_only=False on our own checkpoints).
- 7711769: flow dry run PASSED end to end incl. RESUMED from step 3 and predictor_final.pt. Untrained scores ~0 Spearman on val(250)/test(499), as expected.
- 7711852 (flow_pilot10) / 7711853 (det_pilot10): pilot on smoke10 (150 transitions, 300 steps), eval 50/50 anchors during, full val/test at end.
- 7711852/53 pilot10 PASSED (pipeline check): 300 steps on 150 transitions, flow loss 2.09->1.55, det 0.66->0.55; final eval on 250 val / 499 test ~0 Spearman (expected at this data size). Checkpoints predictor_final.pt + metrics.json written.
- sub500 runs (23,324 transitions, 6000 steps, batch 64, lr 3e-4, d=384 depth=6 18.8M params): flow chain 7711930/32/33/34, det chain 7711935/36/37/38 (afterany; each job resumes from predictor_last.pt, final eval 4 samples x 8 Euler steps on 250 val + 499 test).
- RESULT sub500 (6000 steps, ~4-6 min training): det twin test(499) rho .713 / fine .308 / pair .608 / AUROC .866 / regret 6.7 cm / ECE .037 / pred_err .238 (frozen 22M RoboJEPA: .758/.357/pred_err .267). flow with sample-averaged L1 scorer (4 samples x 8 Euler): rho .124 / fine ~0 / pred_err(sample mean) .31. Flow loss 2.0 -> 1.37.
  Interpretation: with the naive expected-distance scorer the flow predictor is far worse; sample noise swamps the candidate differences. Added posterior-mean (meanpred) and zero-init ODE (zeroinit) scorers; re-evaluating flow_sub500 with 16 samples x 16 steps, and launching 30k-step long runs for both twins.
- 7711970 cancelled (16x16 samples would exceed 30 min). 7711980 + first long chains 7711971-78 crashed in eval: new meanpred/zeroinit scorers broadcast goal [1,1,N,C] against [K,N,C] -> [1,K]; fixed to g[0]. Resubmitted: eval8 of flow_sub500 = 7712015; flow_sub500_long 7712016-19; det_sub500_long 7712020-23 (30k steps, final 8 samples x 8 steps).
- 7712015 eval8 of flow_sub500 (8 samples x 8 steps, test 499): expected-distance rho .126, 1-sample .113, posterior-mean .159, zero-init ODE path .283 (fine .064); pred_err of 8-sample mean .284 (det twin .238, init .381). So sample noise is part of the gap but the learned vector field itself is still far from the det twin at 6k steps (flow MSE 1.33 vs implied floor ~0.5-0.6). Waiting for the 30k-step twins before judging.
- RESULT long (30k steps, ~20 min training): det test rho .736 / fine .298 / pred_err .244 (slightly worse pred_err than 6k .238 -> ~80 epochs over 500 eps, mild overfit). flow periodic (2x4) climbing steadily .23/.43/.51/.61/.63 overall, fine .02->.16 at 5k..30k; loss 1.32 still falling. zero-init ODE path is NOT a good mean estimate (.06-.25); sample mean ~ expected distance. Final 8x8 eval in chain2 7712017.
- Added param=x (x-prediction) variant; submitted flowx_sub500_long 7712518-20 (30k steps, matched to v-pred run) and flow_sub500_90k 7712521-26 (v-pred, 90k steps) to test whether flow simply needs more compute. Chain links after completion now exit with FINAL_EXISTS instead of re-running the final eval.
- RESULT flow_sub500_long final (30k steps, 8 samples x 8 Euler, test 499): expected-distance rho .606 / fine .129 / AUROC .763 / regret 7.5 / ECE .078; posterior-mean scorer .627 / fine .181; 1-sample .541; pred_err(8-sample mean) .278. Matched det twin: .736 / .298 / .869 / 6.4 / .032 / pred_err .244. Flow is still improving with steps; v-pred 90k (7712543-48) and x-pred 30k (7712540-42) running (first submissions 7712518-26 crashed on a variable-shadowing bug in predict, fixed).
- progress: flowx (x-pred) 30k periodic test(100) rho .666 / fine .168, pred_err .244 (matches det twin's pred_err; det ranking at same n=100 periodic was .707/.296). flow v-pred 90k at 45k: test(100) .696 / fine .186 (still climbing), pred_err .336.
- RESULT flowx_sub500_long final (x-pred, 30k, 8x8, test 499): expected-distance rho .622 / fine .151 / AUROC .781 / regret 6.7 / ECE .055 / pred_err(8-sample mean) .242 (= det twin .244); posterior-mean scorer .700 / fine .241; 1-sample .582/.114; zero-init .477/.131. Val(250): .658/.199, meanpred .700/.281. x-pred beats v-pred at 30k (.606/.129, meanpred .627/.181) and matches det pred_err, but still -.04 overall / -.06 fine vs det (.736/.298) even with the best scorer.
- 2026-10-06 ~07:50Z: cluster hiccup (home dir "No space left", slurmctld unreachable) while 90k chain (7712545-48) pending; 90k at ~54k: test(100) .684-.704 / fine .196.
- RESULT flow_sub500_90k final (v-pred, 90k steps = 3x det compute, 8x8, test 499, job 7712545, metrics runs/flow_sub500_90k/metrics.json): expected-distance rho .723 / fine .254 / pair .591 / AUROC .851 / regret 5.3 / hit1 .026 / ECE .045 / pred_err .277; meanpred .700/.251; 1-sample .676/.188; zero-init .249/.041. Val(250) .734/.290. Still below det twin at 30k (.736/.298/.607/.869/6.4/.016/.032/.244) and still climbing; x-pred 30k best scorer .700/.241. VERDICT at 18.8M/sub500: flow does not beat matched L1; gate (beat by >=.03 fine) fails at 1x and 3x compute. Chain links 7712546-48 cancelled (nothing to do).
