# Vision, model predictive control, and world-action models

*Gyanateet Dutta, University of Leeds. Scoring ran on [Isambard-AI](https://www.bristol.ac.uk/research/centres/bristol-supercomputing/articles/2026/inside-isambard-ai.html) at the University of Bristol, part of the UK [AI Research Resource (AIRR)](https://www.gov.uk/government/publications/ai-research-resource).*

*Dataset: [Ryukijano/robojepa-rankability](https://huggingface.co/datasets/Ryukijano/robojepa-rankability).
Leaderboard: [Ryukijano/world-model-rankability](https://huggingface.co/spaces/Ryukijano/world-model-rankability).
Code: [Ryukijano/world-model-rankability](https://github.com/Ryukijano/world-model-rankability).
The numbers in the measurement sections are from simulation and from recorded robot video. No physical robot was used.*

On 2,996 RoboCasa reaching states, Meta's released RoboJEPA models, 22 million to 8 billion parameters, rank the same 64 candidate motions by how close each imagined future is to the goal. Overall Spearman rank correlation is .747 at 22M, drops to .718 at 50M, reaches .784 at 300M, and then stays between .789 and .791 through 8B. The standard error is .003. Imagination error, the latent distance between the imagined frame and the real next frame, falls from .273 to .218 over the same ladder. Among candidates already aimed at the goal, the correlation tops out at .418.

I ran the ladder because an end-to-end vision-language-action (VLA) policy hides this comparison. The policy maps a camera image and an instruction to a motor command. A new kitchen has meant more demonstrations and another training run. RoboJEPA writes the future down, so the ordering can be scored.

![Overall Spearman, fine Spearman, and imagination error for eight RoboJEPA sizes. Imagination error uses the right-hand axis.](https://raw.githubusercontent.com/Ryukijano/world-model-rankability/main/figures/imagination_vs_ranking.svg)

*On 2,996 pretraining-kitchen states, one fixed bank of 64 actions, open-loop Reach, one world-model step. The shaded band from 300M to 8B marks where overall Spearman stays between .784 and .791. It is not a confidence interval. The standard error of the overall correlation is .003.*

## Four computations

Refining an action, predicting what it will do, planning with that prediction, and observing the outcome are four different computations.

A diffusion or flow policy edits the motor command. A world model predicts the future of that command, in pixels or in a latent. Planning holds the predictor fixed and changes the action, either by gradient steps or by scoring a bank of candidates. The MuJoCo replay, or the recorded video, is the outcome. A VLA runs the edit, the prediction, and the plan inside one forward pass. This benchmark imagines each candidate and compares that order with the replay.

The loop I want, and have not trained, is observe, represent, propose, predict, refine, act. Memory is part of the representation. A picture of the candidate is part of the prediction. Language is part of the proposal. A diffusion head and a reinforcement-learning update are the refinement. The numbers below stop at predict, then choose, on models Meta trained.

## Vision learned the feature, then dropped the label

Computer vision spent a long time on features a person designed: edges, corners, histograms. AlexNet (Krizhevsky, Sutskever, and Hinton, 2012) showed that a deep network trained on ImageNet labels could learn the features instead. ResNet (He, Zhang, Ren, and Sun, 2015) made those networks deep enough to train. The Vision Transformer (Dosovitskiy et al., 2020) replaced the convolutional stack with the same block used for language.

Labels do not scale to video. Self-supervised learning trains the representation from the video itself. SimCLR (Chen et al., 2020) pulled two crops of one image together and pushed other images away. DINO (Caron et al., 2021) did without the negative samples. A masked autoencoder (He et al., 2022) hid patches and reconstructed pixels. A joint-embedding predictive architecture (JEPA) hides the target and predicts its representation, not its pixels: I-JEPA for images (Assran et al., 2023), V-JEPA for video (Bardes et al., 2024), then V-JEPA 2. RoboJEPA, the model in the ladder, is a predictor on a frozen V-JEPA 2.1 encoder.

A classifier's features answer names. A JEPA's features are trained to be predictable a step ahead. Distances in that space are a control signal when they order real outcomes. The benchmark tests that condition.

## Model predictive control already planned with a model

Model predictive control (MPC) does the third computation with a model of the plant. At each time it proposes a short action sequence, rolls the model forward, scores the rollout, applies the first move, and replans when the next observation arrives. The plant model is held fixed while the action is updated. Process industries used this pattern for decades. On robots, online trajectory optimization of the kind in Tassa, Erez, and Todorov (2012) ran the same loop on contact-rich simulated bodies: a local model, a quadratic cost, an action update, then a new observation.

Those robot controllers needed a state. Mobile robots took it from lidar. In the DARPA Grand Challenge and Urban Challenge years, 2004 to 2007, vehicles such as Stanford's Stanley carried lidar so the planner had a pose and a geometric map of obstacles (Thrun et al., 2006). A point cloud is enough to keep a vehicle off a wall. It does not carry the cup, the contact, or the grasp. Manipulation went back to cameras, and the state had to be learned.

## Monte Carlo replaced the linear solve

A linear MPC step assumes the local model and the cost are smooth enough to differentiate. Contact, discrete choices, and a learned model break that. Sampling is the replacement. Draw action sequences, roll the model, keep the ones with a better score, fit a new proposal to those, and repeat.

The cross-entropy method (Rubinstein, 1999) is that loop. Model-predictive path integral control (Williams, Aldrich, and Theodorou, 2017) is the same idea with a different weighting of the samples. PETS (Chua, Calandra, McAllister, and Levine, 2018) put a probabilistic neural dynamics model under a cross-entropy planner. PlaNet and Dreamer (Hafner et al., 2019, 2020) learned the dynamics in a latent and trained the policy on imagined rollouts. Monte Carlo tree search is the other sampling tradition, built for games. On robot arms the version that mattered was the action-sequence version: a cloud of futures, an elite set, a new cloud.

LeWM, the latent world model that later JEPA planners compare against, plans with the cross-entropy method. If latent distance to a goal image orders the wrong end positions, the elite set is the wrong set of actions.

## Reinforcement learning specified the task as a reward

Reinforcement learning (RL) arrives when writing the dynamics is harder than scoring the outcome. The agent tries actions, receives a reward, and updates a policy. DQN (Mnih et al., 2015) did this for pixels in Atari. PPO (Schulman et al., 2017) and SAC (Haarnoja et al., 2018) made continuous actions practical. End-to-end visuomotor policies (Levine, Finn, Darrell, and Abbeel, 2016) and QT-Opt (Kalashnikov et al., 2018) ran the loop from camera images to gripper motion.

Simulators made the trial cheap. On a real robot the same loop pays for resets, breakage, and a reward a person has to write. A demonstration is easier to collect than that reward, so imitation became the training run people actually scaled. RL remained as a later fine-tune, on a policy that already imitates.

## One network from image and sentence to torque

A VLA is that imitation policy with language in the input. RT-1 (Brohan et al., 2022) trained a transformer on robot demonstrations to map images and instructions to actions. RT-2 (Brohan et al., 2023) co-fine-tuned a vision-language model so that robot actions were tokens in the same network as web text and images. OpenVLA (Kim et al., 2024) did this in the open. π0 (Black et al., Physical Intelligence) keeps a vision-language backbone and attaches a flow-matching action expert, so the motor command is generated by the same iterative family as a diffusion policy.

These models are trained, or fine-tuned, on demonstrations for the arms they run. A new scene has meant a new training run. The deployed forward pass does not show whether two nearby motions were ordered by their consequences, so the next kitchen still needs new weights. I measured a world model that writes the prediction down. I did not run this bank on RT-2, OpenVLA, or π0.

## Two world models, two uses

A world model answers a different question from a VLA. The VLA asks which action to emit. The world model asks what happens if this action is emitted.

Latent world models, including V-JEPA 2 and RoboJEPA, predict a representation. They are cheap to roll, and the plan is a comparison of latent distances. Generative world models predict pixels, usually with a video diffusion backbone. They are expensive, and a person can watch the rollout.

A pixel rollout can be watched before the arm moves, and kept or discarded as training data. UniSim (Yang et al., 2023) trained a generative model of real interactions and used it as a policy simulator. Ctrl-World (Guo, Shi, Chen, and Finn, arXiv:2510.10125) starts from a video diffusion model, conditions on actions at frame level, and rolls a generalist policy in imagination on DROID. The authors report that the model ranks policies without a real-robot rollout, and that supervised fine-tuning on imagined successes raises policy success by 44.7%. I have not reproduced that paper.

A latent distance has no picture attached. It ranks actions when a smaller distance is a better action. The ladder below tests that on Meta's RoboJEPA models, with their planning score: negative mean latent L1 to the goal frame.

## A world-action model trains both questions

[LeWAM](https://le-wam.github.io/) (Fu, Siebert, Halicki, and Balestriero, arXiv:2610.12818) trains the action and the consequence in one network of about 17 million parameters. The default encoder is a ResNet-18 with spatial softmax and a two-layer projection to a 192-dimensional state, trained with the predictor. They also evaluate a frozen V-JEPA 2.1 encoder as a variant. The predictor is a Mixture-of-Transformers. One branch predicts the next latent from clean actions. One branch predicts an action-flow velocity from noisy actions. The joint objective in their equation (5) is

\[
\mathcal{L} = \mathcal{L}_{\mathrm{dyn}} + \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{sig}} \mathcal{L}_{\mathrm{sig}}.
\]

\(\mathcal{L}_{\mathrm{dyn}}\) is latent dynamics, \(\mathcal{L}_{\mathrm{act}}\) is action flow matching, and the third term is SIGReg, an isotropic-Gaussian penalty on the latents. One coefficient, \(\lambda_{\mathrm{sig}}\), multiplies that penalty.

Two other papers use the name LeWAM for different systems. Hegde, Popov, Aljalbout, and Smolyanskiy (arXiv:2610.12407) describe a JEPA world-action model with diffusion-steering MPC. Fang, Duan, Wu, Wang, and Qi (arXiv:2609.27455) describe a latent evolving world-action model. The mask and the table in this section are Fu et al. only.

The dynamics branch has to see the demonstrated action, because that action is the cause. The action branch must not see it, or training copies the demonstration. Fu et al. state the rule in the Figure 3 caption and in section 3.1: the state query reads history and clean actions, and noisy action tokens read state history and their own causal prefix. History latents attend only to themselves, which blocks a clean action from being written into a token the policy later reads. Goal information conditions the noisy action tokens. Appendix A writes the same rule as sets. Grouped, a history query reads only the history latents. The state query reads the history latents, itself, and the clean action chunk. A clean-action query reads the history and the clean actions up to its own index. A noisy-action query reads the history and the noisy actions up to its own index, plus the goal token when a goal is supplied. It does not read the clean actions. If a history token could read a clean future action, the block on the action branch would be bypassed through that history token.

| Query | History | State query | Goal | Clean actions | Noisy actions |
|---|---|---|---|---|---|
| History | all | no | no | no | no |
| State query | all | itself | no | the clean chunk | no |
| Clean action at index j | all | no | no | clean actions up to j | no |
| Noisy action at index j | all | no | the goal token, if a goal is supplied | no | noisy actions up to j |

Their planner then freezes those weights. They sample action chunks from the learned flow, roll the latent dynamics with the first K actions (the planning setup uses a frame skip of K = 5), and take gradient steps on the actions through the frozen dynamics. The cost is mean squared error between the predicted latent and the goal latent, plus a small penalty for leaving the original proposal. The robot executes the first K actions and replans. The optimizer changes the action. It does not change the network.

The authors report, on their benchmarks, success from 28.6% to 89.7% on contact-rich tasks and from 10.9% to 46.2% on a 100-step goal offset across four tasks, and 32.3 times as many plans per second as the cross-entropy baseline they compare with. Across ten simulated environments they report 93.5% mean success against 50.9% for LeWM with CEM. On Push-T, the planning gain over direct control is 14.7 percentage points with SIGReg and 2.7 without it. I have not reproduced those runs.

On a physical robot they also execute the policy directly, first half of each action chunk, 30 trials each, on Pick and Place T, Push-T, and Tape Stack. They describe the result as competitive. Diffusion Policy has the higher average.

| Method | Pick and Place T | Push-T | Tape Stack | Average | GFLOPs per chunk |
|---|---:|---:|---:|---:|---:|
| Diffusion Policy | 19/30 | 16/30 | 0/30 | 38.9% | 270.0 |
| π0.5 | 14/30 | 3/30 | 9/30 | 28.9% | 3603.6 |
| LeWAM, frozen V-JEPA 2.1 | 15/30 | 12/30 | 5/30 | 35.6% | 270.2 |
| LeWAM, end to end | 16/30 | 8/30 | 6/30 | 33.3% | 270.2 |

One LeWAM chunk is 270.2 GFLOPs against 3603.6 for π0.5. They report that ratio as 13.3 times as many action chunks per TFLOP. The 35.6% row uses a frozen V-JEPA 2.1 encoder. The default model is the end-to-end row at 33.3%. Both LeWAM rows sit below Diffusion Policy's 38.9% average on this table.

## Iteration on the action is a different job

Diffusion policies became a default because a squared-error regression can average two demonstrated ways past an obstacle into a path that hits it. [Pan, Anantharaman, Huang, Jin, Pfrommer, Yuan, Permenter, Qu, Boffi, Shi, and Simchowitz](https://arxiv.org/abs/2512.01809), in *Much Ado About Noising* (arXiv:2512.01809), report a different account on the behaviour-cloning benchmarks they study. Generative control policies, in their results, do not owe the gains to multiple modes or to a more complex map from observation to action. The piece they isolate is iterative computation whose intermediate steps are supervised, paired with noise during training.

They separate three ingredients: fitting the full action distribution, injecting noise, and supervising the intermediate computation. A minimal iterative policy (MIP) keeps the second and the third, in two network calls, and drops distribution matching. Their section title is "minimal iterative policy".

The interpolant in the paper is \(I_t = t a + (1-t) z\), so \(t = 1\) is the clean action and \(t = 0\) is noise. They fix \(t_\star = 0.9\). Two-step denoising trains the network so that

\[
\pi_\theta(o, z, 0) \rightarrow t_\star^{-1} I_{t_\star}, \qquad \pi_\theta(o, I_{t_\star}, t_\star) \rightarrow a.
\]

Sampling draws \(\hat{a}_0 = \pi_\theta(o, z, 0)\), then \(\hat{a} = \pi_\theta(o,\ t_\star \hat{a}_0 + (1-t_\star) z,\ t_\star)\). MIP replaces the first target by the clean action, sets the first input to zero, and leaves noise only in the second training term. Inference uses \(z = 0\):

\[
\hat{a}_0 = \pi_\theta(o, 0, 0), \qquad \hat{a} = \pi_\theta(o,\ t_\star \hat{a}_0,\ t_\star).
\]

The first call proposes. The second call refines, from a point that is \(t_\star\) times the proposal. MIP edits an action. The scene that follows is a different model.

They use the name **manifold adherence** for a related measurement: under perturbed observations, the size of the component of the predicted action that lies off the span of nearby demonstrated actions. Flow policies and MIP show a smaller off-manifold component than regression, at similar validation loss. They treat the control consequence as a conjecture, and they say the mechanism is open. An error along a demonstration can still be the wrong speed.

## Memory has to be trained for

[Long-WAM](https://nvlabs.github.io/LongLive/Long-WAM/) (arXiv:2610.10528) asks how much visual history a world-action model can use. The abstract reports that on RoboCasa with a GR-1 robot, stretching context from 0 to 19.2 seconds raises success from 63.3% to 78.7% only when the video backbone was pretrained autoregressively, and that a bidirectional initialization shows no net gain on that comparison. The same abstract reports 19 of 20 trials on a dynamic cup-stacking task, against 0 of 20 for the π0.5 and Fast-WAM baselines they name. I have not reproduced those runs. The project page lists further headlines. The figures in this paragraph are from the abstract.

On the RoboJEPA reaching probe, a second frame of context never beat one frame. The absolute change in Spearman was at most .02 on the pilots. Their experiment adds up to 19.2 seconds of history on a different model and task. On this probe, the extra frame changed the correlation by at most that .02.

## What I measured

Each RoboJEPA size sits on one frozen V-JEPA 2.1 ViT-G encoder. The task is RoboCasa Reach with a PandaOmron, an external camera and a wrist camera, frames at 320 by 240, one frame of context, and a 4-step action chunk, which is one world-model step. Embodiment is the 7-D pose token used with these checkpoints.

There are 3,499 states: 3,000 from 30 kitchens in the pretraining distribution, of which 2,996 were scored, and 499 from 10 kitchens held out. Each state has the same 64 candidates. Thirty-two are random. Sixteen are coarse and sixteen are fine, both aimed at the goal, inside a 0.18 m reach so that goal-directed candidates are not clipped into duplicates. An earlier 0.30 m bank clipped 35% of the goal-directed candidates and dropped fine Spearman to the range .19 to .22. The bank fix moved the fine correlation more than model size did. I replayed every candidate in MuJoCo from a saved state. Repeating a replay agrees to within 0.2 mm.

Spearman is the correlation between the model's ordering of the 64 actions and the true ordering. Fine Spearman uses only the goal-directed half. Regret is how much further from the goal the chosen action ends, compared with the best candidate in the set. Hit@1 is how often the top-scored action is the true best candidate. Imagination error is the latent L1 between the imagined next frame and the real next frame. The score ranks this fixed bank in open loop against the MuJoCo end position. Gradient steps through the dynamics, and closed-loop task success, are separate experiments.

| Size | Overall Spearman | Fine Spearman | Regret | Hit@1 | Imagination error |
|---|---:|---:|---:|---:|---:|
| 22M | .747 | .359 | 5.7 cm | 2.2% | .273 |
| 50M | .718 | .282 | 6.6 cm | 1.3% | .263 |
| 100M | .761 | .392 | 5.6 cm | 3.7% | .253 |
| 300M | .784 | .411 | 5.1 cm | 4.4% | .240 |
| 1B | .789 | .418 | 5.0 cm | 4.7% | .223 |
| 2B | .790 | .419 | 4.9 cm | 5.2% | .221 |
| 4B | .791 | .413 | 4.9 cm | 5.3% | .218 |
| 8B | .789 | .409 | 5.0 cm | 5.5% | .218 |

The 50M checkpoint is below 22M on every column. From 1B to 8B, overall Spearman stays inside .789 to .791, while imagination error is still lower at 4B and 8B than at 300M. On the 499 unseen kitchens the shape repeats: .758 at 22M, .730 at 50M, .799 at 1B, .799 at 8B. At the small end, training steps still move the 22M model on the 0.18 m bank: overall Spearman .47 at 10k steps, .74 at 50k, and .755 at 150k.

![Pairwise accuracy against the true difference in end position, for every RoboJEPA size](https://raw.githubusercontent.com/Ryukijano/world-model-rankability/main/figures/resolution_floor.png)

Pairs whose true outcomes differ by less than 5 mm are ordered correctly 51% to 53% of the time, at every size. At 1 to 2 cm the rate is 59% to 61%. Above 8 cm it is 88% to 93%. The floor does not move when the predictor grows.

Of the states where the 1B pick ends more than 5 cm worse than the best candidate, 78% are also failures for 22M, and 19% fail at all eight sizes. The 1B pick ends 3.0 cm, 2.9 cm, and 3.5 cm off in x, y, and height. The best candidate in the set was 0.6 to 0.7 cm off on each axis.

Scoring only the patches that change removes rank signal. For the 1B model, fine Spearman falls from .418 to .361, .306, and .260 when the score keeps the top 50%, 25%, and 10% of patches by latent change. The same drop appears at 22M and at 8B. The action signal in these latents is spread across the frame.

Fixed-candidate ranking is already in ARC-Bench (arXiv:2609.05461) and AD-WM (arXiv:2609.30264). This ladder adds the released RoboJEPA scale axis, imagination error on the same inputs as the ranking, held-out kitchens, and the millimetre floor.

### Recorded video, at a coarser grain

On 256 held-out DROID clips, each model scores the true action sequence against 15 sequences from other episodes, and against edits of the true action. Scores sum about 15 teacher-forced steps, so this test is easier than the single-step simulator ranking. Mean true translation is 3.9 cm per world-model step. The 8B model is not in this table.

Picking the true sequence out of 16 is a top-1 of .996 to 1.0. Chance is .0625. Sign-flipped translation, zeroed rotation, a flipped gripper, a one-step delay, and an all-zero action are detected at .91 to 1.0 depending on the checkpoint. Magnitude is the failure. Pairwise accuracy of the true action against a rescaled translation:

| Checkpoint | Translation times 0.9 | Translation times 1.1 |
|---|---:|---:|
| Meta 22M, 10k steps | .477 | .648 |
| Meta 22M, 150k steps | .562 | .801 |
| Meta 1B, 300k steps | .793 | .934 |

A 10% slower motion is near chance for the small early checkpoint and is still missed about one time in five by the 1B model.

### A rank loss and an imagination loss move apart

Fine-tuning RoboJEPA-22M with a ranking loss on replayed outcomes, 1,000 steps, 3 seeds, 20 states from each of 25 pretraining kitchens, reached .823 overall Spearman and .492 fine Spearman on held-out kitchens. A compute-matched control fine-tune, without the ranking loss, reached .794 and .407. The frozen 8B model is at .799 overall on the same unseen kitchens. Imagination error rose from .242 to .268. The ranking fine-tune inherits the CC BY-NC-SA 4.0 terms of the RoboJEPA weights.

The frozen encoder bounds how far that fine-tune can go. If V-JEPA 2.1 never encoded a few millimetres of end-effector motion, no predictor on top of it can rank those motions by anything but noise.

## What this says about end-to-end VLAs

The ladder is evidence about one scorer: latent L1 on a frozen V-JEPA 2.1 representation, open loop, Reach, one step. It is evidence that, inside this family, predictor scale past about 300 million parameters keeps improving the imagined frame and leaves the ordering of actions in place. A training signal that improves the ordering makes the imagined frame worse.

An end-to-end VLA can hold a world model, a language model, and an action head in one set of weights. When the setting changes, the usual fix is another training run, because that ordering was never pulled out and scored. I do not have this bank for RT-2, OpenVLA, or π0. I have it for a latent world model that chooses by imagination, and that choice saturates while the prediction error is still falling. A policy that never writes the prediction down can carry the same ceiling, with the next fine-tune as the visible symptom.

## A direction I have not trained

No gate below has been run.

I want a future you can look at: imagine the candidate in pixels, so a short reach is visible, and accept or reject it before the arm moves. The latent has to order those futures, including the DROID near misses, where the true action is scaled by about 10% and by about 25%. Language sits on the proposal, so a model can state the goal and later state which imagined future missed it. The action update is a fine-tune, first by imitation and then by reinforcement learning, with a diffusion or flow head. Memory belongs in that stack only when training reads it. Long-WAM's autoregressive pretraining and the flat two-frame pilots here are the same lesson on different tasks.

The nearer experiment, also untrained, is smaller than that stack. It is specified in `breakage/MODEL_PLAN.md` as draft v1. A pixel world-action model in the LeWM style, on the order of 15 million to 100 million parameters, trained end to end with no frozen V-JEPA encoder. An action-contrastive term so the latent has to carry which action happened. An optional teacher anchor. A preference head over K candidates. A separate action head whose loss is blocked from the shared encoder, as an ablation. The gates, before any scale-up, are: beat Meta's 1B on the DROID rescales at 10% and at 25%, beat fine Spearman .418 on this reaching bench, separate success from failure on held-out DROID episodes, and run closed-loop RoboCasa. A LeWM baseline on the converted DROID set, 4,999 training episodes and 102 holdout, takes 0.7 seconds per step and 13.8 GB peak on one GH200. That is a measured training cost, not a ranking result.

## Where the numbers are

The [leaderboard](https://huggingface.co/spaces/Ryukijano/world-model-rankability) has the eight sizes. The [dataset](https://huggingface.co/datasets/Ryukijano/robojepa-rankability) accepts a score file through `evaluate.py`. A live Space runs the checkpoints on ZeroGPU. On 30 states, live scores and stored scores agree at a median Spearman of .99995.

Papers and code used above:

- Pan et al., [Much Ado About Noising](https://arxiv.org/abs/2512.01809). The MIP construction and the claim about iterative computation are theirs.
- Fu, Siebert, Halicki, and Balestriero, [LeWAM](https://arxiv.org/abs/2610.12818). Author-reported success rates are cited as reported.
- [Long-WAM](https://arxiv.org/abs/2610.10528). The 63.3% to 78.7% figure is the abstract.
- Guo, Shi, Chen, and Finn, [Ctrl-World](https://arxiv.org/abs/2510.10125). The 44.7% figure is theirs.
- RoboJEPA and V-JEPA 2.1, Meta. RoboCasa. DROID.
- ARC-Bench, arXiv:2609.05461. AD-WM, arXiv:2609.30264.

Scores derived from RoboJEPA weights are CC BY-NC-SA 4.0, and a ranking fine-tune of those weights inherits that licence. RoboCasa is MIT. The benchmark is for non-commercial research. No physical robot was used. A different score, a different task, or a representation trained on small motions could still plan. This benchmark does not test those.

Figure files, as direct links for a Hugging Face post:

- Ranking against imagination error: https://raw.githubusercontent.com/Ryukijano/world-model-rankability/main/figures/imagination_vs_ranking.svg
- Resolution floor: https://raw.githubusercontent.com/Ryukijano/world-model-rankability/main/figures/resolution_floor.png
- Leaderboard screenshot already in the repo: https://raw.githubusercontent.com/Ryukijano/world-model-rankability/main/figures/static_space.png
