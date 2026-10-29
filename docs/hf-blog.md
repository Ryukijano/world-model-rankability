# From Diffusion Policies to World-Action Models: Teaching Robots to Predict, Plan, and Act

![Inside a latent world-action model: grouped attention, and planning by gradients through a frozen dynamics model.](https://cdn-uploads.huggingface.co/production/uploads/623c636949b6a399ee11152e/27HV3OIxuuO_2K7LZOdaT.png)

*Notes on generative robot policies, joint-embedding predictive architectures, latent-space planning, and a measurement of when a closer imagined future is still the same choice.*

*Gyanateet Dutta, University of Leeds. The scoring jobs ran on [Isambard-AI](https://www.bristol.ac.uk/research/centres/bristol-supercomputing/articles/2026/inside-isambard-ai.html), part of the UK [AI Research Resource (AIRR)](https://www.gov.uk/government/publications/ai-research-resource). Dataset: [Ryukijano/robojepa-rankability](https://huggingface.co/datasets/Ryukijano/robojepa-rankability). Leaderboard: [Ryukijano/world-model-rankability](https://huggingface.co/spaces/Ryukijano/world-model-rankability). Code: [Ryukijano/world-model-rankability](https://github.com/Ryukijano/world-model-rankability). The numbers I report from my own runs are simulation and recorded video. No physical robot was used.*

Robotics sits at an intersection of perception, prediction, and decision-making.

A robot observes the world, proposes an action, and then has to deal with what actually happens. The next camera frame may differ from everything in the training set. Objects move, contacts are imprecise, and a small error can accumulate over a long sequence of actions.

So robot learning is a mapping from observations to actions, and it is also a question about how the world responds. I have been reading several lines of work that split that question differently:

- **Diffusion and flow-matching policies** generate a structured action sequence.
- **A minimal iterative policy (MIP)** refines an action in two supervised steps.
- **Joint-embedding predictive architectures (JEPAs)** predict in a learned representation, rather than reconstructing pixels.
- **World-action models** train that prediction and the action together.
- **Differentiable model-predictive control** then changes the candidate action using gradients through a learned model, with the model held fixed.

The interesting part is how these fit together, and where a measurement says they come apart.

## 1. The problem with simply imitating demonstrations

A conventional imitation policy learns

\[
a_t = \pi_\theta(o_t).
\]

In practice many robot policies predict an action chunk, a short sequence of future controls,

\[
A_t = (a_t, a_{t+1}, \ldots, a_{t+H-1}).
\]

Chunking keeps the motion consistent and means the network is called less often.

Behavioral cloning still has the familiar weakness. The policy is trained on demonstrations and deployed on trajectories that its own mistakes helped create. A slightly wrong action moves the robot into a state that was rare in the data. The next prediction then has to work outside the distribution the policy learned to imitate. Ross, Gordon, and Bagnell called the remedy of collecting data on the learner's own states DAgger. Generative policies were a different response: learn a distribution over plausible sequences, so two good ways past an obstacle are not averaged into a path that hits it.

## 2. Why generative action policies work, and what Pan et al. actually isolated

Diffusion policies turn noise into an action by a sequence of denoising steps. Flow matching learns a vector field that carries a sample toward the action distribution. Both can represent more than one plausible action.

The open question is whether the gain comes from fitting that distribution, or from the iteration itself.

[Pan, Anantharaman, Huang, Jin, Pfrommer, Yuan, Permenter, Qu, Boffi, Shi, and Simchowitz](https://arxiv.org/abs/2512.01809), in *Much Ado About Noising*, compare generative and regression policies under matched architectures. Their claim is that the advantage is not a universal property of being generative. On the behaviour-cloning tasks they study, multiple modes and a more expressive map from observation to action do not explain the gap. What they isolate is iterative computation whose intermediate step is supervised, paired with noise during training.

That is a more useful question than "are generative policies better?": which property of learning and of inference actually changes the control.

## 3. MIP: two calls, not a diffusion loop

The method in that paper is a **minimal iterative policy**. The name in the abstract and in the section title is "minimal," not "multistep inverse problems."

Their interpolant is \(I_t = t a + (1-t) z\), so \(t = 1\) is the clean action and \(t = 0\) is noise. They fix \(t_\star = 0.9\). Two-step denoising trains

\[
\pi_\theta(o, z, 0) \rightarrow t_\star^{-1} I_{t_\star}, \qquad \pi_\theta(o, I_{t_\star}, t_\star) \rightarrow a.
\]

Sampling draws \(\hat a_0 = \pi_\theta(o, z, 0)\), then \(\hat a = \pi_\theta(o,\ t_\star \hat a_0 + (1-t_\star) z,\ t_\star)\). MIP replaces the first target with the clean action, feeds zeros into the first call, and leaves noise only in the second training term. At inference, \(z = 0\):

\[
\hat a_0 = \pi_\theta(o, 0, 0), \qquad \hat a = \pi_\theta(o,\ t_\star \hat a_0,\ t_\star).
\]

The first call proposes. The second refines, from a point that is \(t_\star\) times the proposal. A sketch that drops \(t_\star\) and writes \(\hat A_1 = f_\theta(o, \hat A_0)\) is the right picture and the wrong formula.

MIP edits an action. It does not say what that action does to the scene. They also measure something they call **manifold adherence**: under a perturbed observation, how much of the predicted action falls outside the span of nearby demonstrated actions. Flow policies and MIP show a smaller off-manifold piece than regression, at similar validation loss. They treat the consequence for control as a conjecture, and they say the mechanism is open. An error along a demonstration can still be the wrong speed.

## 4. From predicting actions to predicting the world

An action model can be right about which motor command usually accompanies an image, and silent about what that command will cause.

A world model learns a state and how the state moves. Encode the observation, \(z_t = E(o_t)\), and predict

\[
\hat z_{t+1} = F_\phi(z_t, a_t).
\]

Pixel prediction spends capacity on appearance. A JEPA predicts the representation of a hidden target, not the pixels: I-JEPA for images (Assran et al., 2023), V-JEPA for video (Bardes et al., 2024), then V-JEPA 2. RoboJEPA, the model I score below, is a predictor on a frozen V-JEPA 2.1 encoder. Distances in that space are a control signal when they order real outcomes. That condition is the benchmark.

Some world-action models stay in pixels, so you can watch the future before the arm moves. The figure below is that kind of system: a diffusion transformer denoises future images, masks, and actions together. It is not LeWAM. Fu et al. use a ResNet-18 and a mixture-of-transformers, not a VAE and a DiT.

![A diffusion world-action model trained to denoise future images, masks, and actions together.](https://cdn-uploads.huggingface.co/production/uploads/623c636949b6a399ee11152e/byPIY8Snfdju_dBNWAo8x.png)

A watched rollout is also a training signal. UniSim (Yang et al., 2023) used a generative model of real interactions as a policy simulator. Ctrl-World (Guo, Shi, Chen, and Finn, [arXiv:2510.10125](https://arxiv.org/abs/2510.10125)) starts from a video diffusion model and rolls a generalist policy in imagination on DROID. The authors report that it ranks policies without a real-robot rollout, and that supervised fine-tuning on imagined successes raises policy success by 44.7%. I have not reproduced that paper. The use is the one I care about: look at the future, then train on the futures you accept.

## 5. LeWAM: one network for the prediction and the action

[LeWAM](https://le-wam.github.io/) (Fu, Siebert, Halicki, and Balestriero, [arXiv:2610.12818](https://arxiv.org/abs/2610.12818)) trains both questions in one network of about 17 million parameters. The default encoder is a ResNet-18 with spatial softmax and a two-layer projection to a 192-dimensional state, trained with the predictor. A frozen V-JEPA 2.1 encoder is a variant they also evaluate. The predictor is a Mixture-of-Transformers. One branch predicts the next latent from clean actions. One branch predicts an action-flow velocity from noisy actions.

The joint objective in their equation (5) is one sum, with a single coefficient on the regularizer:

\[
\mathcal{L} = \mathcal{L}_{\mathrm{dyn}} + \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{sig}} \mathcal{L}_{\mathrm{sig}}.
\]

\(\mathcal{L}_{\mathrm{dyn}}\) is latent dynamics, \(\mathcal{L}_{\mathrm{act}}\) is action flow matching, and \(\mathcal{L}_{\mathrm{sig}}\) is SIGReg, an isotropic-Gaussian penalty on the latents. A schematic with three independent \(\lambda\) weights is not their objective.

Two other papers use the name LeWAM for different systems. Hegde, Popov, Aljalbout, and Smolyanskiy ([arXiv:2610.12407](https://arxiv.org/abs/2610.12407)) describe a JEPA world-action model with diffusion-steering MPC. Fang, Duan, Wu, Wang, and Qi ([arXiv:2609.27455](https://arxiv.org/abs/2609.27455)) describe a latent evolving world-action model. The numbers in this section are Fu et al. only. I have not reproduced their runs.

On their benchmarks they report success from 28.6% to 89.7% on contact-rich tasks and from 10.9% to 46.2% on a 100-step goal offset, and 32.3 times as many plans per second as the cross-entropy baseline they compare with. Across ten simulated environments they report 93.5% mean success against 50.9% for LeWM with CEM. On Push-T, the planning gain over direct control is 14.7 percentage points with SIGReg and 2.7 without it.

On a physical robot they also execute the policy directly, 30 trials each, on Pick and Place T, Push-T, and Tape Stack. They call the result competitive. Diffusion Policy's average is 38.9%. The default end-to-end LeWAM is 33.3%. The frozen V-JEPA 2.1 variant is 35.6%. One LeWAM action chunk is 270.2 GFLOPs, against 3603.6 for π0.5, which they report as 13.3 times as many chunks per TFLOP.

## 6. The attention mask is part of the learning problem

The dynamics branch has to see the demonstrated action, because that action is the cause. The action branch must not see it, or training copies the demonstration.

Fu et al. state the rule in the Figure 3 caption and in section 3.1. The state query reads history and clean actions. Noisy action tokens read state history and their own causal prefix. History latents attend only to themselves, so a clean action cannot be written into a token the policy later reads. Goal information conditions the noisy action tokens. Appendix A writes the same rule as sets. The poster at the top of this post is a sketch of that idea. The grid below follows the appendix.

| Query | History | State query | Goal | Clean actions | Noisy actions |
|---|---|---|---|---|---|
| History | all | no | no | no | no |
| State query | all | itself | no | the clean chunk | no |
| Clean action at index j | all | no | no | clean actions up to j | no |
| Noisy action at index j | all | no | the goal token, if a goal is supplied | no | noisy actions up to j |

If a history token could read a clean future action, the block on the action branch would be bypassed through that history token. The mask is not an efficiency trick. It decides which prediction is legal.

## 7. Differentiable latent MPC, and a score that can still pick the wrong action

Once the dynamics are differentiable, planning can optimize the action and leave the network alone. From a latent \(z_t\) and a candidate sequence \(A\), roll

\[
\hat z_{t+k+1} = F_\phi(\hat z_{t+k}, a_{t+k}),
\]

score the rollout with a running cost and a terminal cost, and take

\[
A^\star = \arg\min_A J(A).
\]

Gradients flow to the action variables. The weights stay frozen. LeWAM does this with mean squared error to a goal latent, a small penalty for leaving the original proposal, and a frame skip of \(K = 5\). The robot executes the first \(K\) actions and replans.

A low predicted cost is not a good real outcome. Errors accumulate, and the optimizer can exploit a mistake in the model. That is the reason I ran a simpler test on a family of latent models that already exist.

Meta's released RoboJEPA checkpoints, 22 million to 8 billion parameters, share one frozen V-JEPA 2.1 ViT-G encoder. On 2,996 RoboCasa reaching states I gave every size the same 64 candidate motions and asked which imagined future was closest to the goal, using their planning score: negative mean latent L1. Overall Spearman rank correlation is .747 at 22M, drops to .718 at 50M, reaches .784 at 300M, and then stays between .789 and .791 through 8B. The standard error is .003. Imagination error, the latent distance from the imagined frame to the real next frame, falls from .273 to .218. Among candidates already aimed at the goal, the correlation tops out at .418.

![Overall Spearman, fine Spearman, and imagination error across eight RoboJEPA sizes. Imagination error uses the right-hand axis. The shaded band is the flat region of overall Spearman, not a confidence interval.](https://raw.githubusercontent.com/Ryukijano/world-model-rankability/main/figures/imagination_vs_ranking.svg)

The imagined frame keeps getting closer. The choice does not. Section 9 has the protocol and the rest of the table. The score is open-loop ranking of a fixed bank against a MuJoCo replay. It is not the gradient loop above, and it is not a closed-loop success rate.

## 8. Why temporal context matters, when the training reads it

One frame often cannot tell you velocity, contact, or what the last action already did.

[Long-WAM](https://nvlabs.github.io/LongLive/Long-WAM/) ([arXiv:2610.10528](https://arxiv.org/abs/2610.10528)) studies how much visual history a world-action model can use. The abstract reports that on RoboCasa with a GR-1, stretching context from 0 to 19.2 seconds raises success from 63.3% to 78.7% only when the video backbone was pretrained autoregressively, and that a bidirectional initialization shows no net gain on that comparison. The same abstract reports 19 of 20 trials on a dynamic cup-stacking task, against 0 of 20 for the π0.5 and Fast-WAM baselines they name. I have not reproduced those runs. Further headlines are on the project page. These figures are from the abstract.

On my reaching probe, a second frame never beat one frame. The absolute change in Spearman was at most .02. The setups differ in the model, the task, and the amount of history. On this probe, extra context changed the correlation only when the training procedure had a reason to read it. Long-WAM's result and mine are the same lesson at two very different lengths.

## 9. What the ladder shows

Each state uses a PandaOmron, an external camera and a wrist camera, frames at 320 by 240, one frame of context, and a 4-step chunk, which is one world-model step. There are 3,499 states: 2,996 scored from 30 pretraining kitchens, and 499 from 10 kitchens held out of RoboCasa pretraining. Thirty-two candidates are random. Sixteen are coarse and sixteen are fine, both aimed at the goal, inside 0.18 m so they are not clipped into duplicates. An earlier 0.30 m bank clipped 35% of the goal-directed candidates and dropped fine Spearman into the range .19 to .22. Fixing the bank moved the fine correlation more than model size did. Replays from a saved MuJoCo state agree to within 0.2 mm.

Spearman compares the model's order with the true order. Fine Spearman uses the goal-directed half. Regret is how much further from the goal the chosen action ends, compared with the best candidate in the set. Hit@1 is how often the top score is the true best. Imagination error is latent L1 between the imagined next frame and the real next frame.

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

The 50M checkpoint is below 22M on every column. From 1B to 8B, overall Spearman stays inside .789 to .791. On the 499 unseen kitchens the shape repeats: .758 at 22M, .730 at 50M, .799 at 1B, .799 at 8B. At the small end, training steps still move the 22M model: overall Spearman .47 at 10k steps, .74 at 50k, and .755 at 150k, on the 0.18 m bank.

![Pairwise accuracy against the true difference in end position.](https://raw.githubusercontent.com/Ryukijano/world-model-rankability/main/figures/resolution_floor.png)

Pairs whose true outcomes differ by less than 5 mm are ordered correctly 51% to 53% of the time, at every size. At 1 to 2 cm the rate is 59% to 61%. Above 8 cm it is 88% to 93%. The floor does not move when the predictor grows.

Of the states where the 1B pick ends more than 5 cm worse than the best candidate, 78% are also failures for 22M, and 19% fail at all eight sizes. The 1B pick ends 3.0 cm, 2.9 cm, and 3.5 cm off in x, y, and height. The best candidate was 0.6 to 0.7 cm off on each axis.

Scoring only the patches that change removes rank signal. For the 1B model, fine Spearman falls from .418 to .361, .306, and .260 when the score keeps the top 50%, 25%, and 10% of patches by latent change. The same drop appears at 22M and at 8B.

On 256 held-out DROID clips, picking the true action sequence out of 16 is nearly perfect. Mean true translation is 3.9 cm per step. Magnitude is the failure. Against a translation scaled by 0.9, pairwise accuracy is .477 for Meta's 22M model at 10k steps, .562 at 150k steps, and .793 for the 1B model at 300k steps. A 10% slower motion is still missed about one time in five by the 1B model. These scores sum about 15 teacher-forced steps, so the test is easier than the single-step simulator ranking. The 8B model is not in that table.

A ranking fine-tune of RoboJEPA-22M, 1,000 steps, 3 seeds, 20 states from each of 25 pretraining kitchens, reached .823 overall Spearman and .492 fine Spearman on held-out kitchens. A compute-matched control, without the ranking loss, reached .794 and .407. The frozen 8B model is at .799 overall on those same kitchens. Imagination error rose from .242 to .268. Training the model to choose made the imagined frame worse. That fine-tune inherits the CC BY-NC-SA 4.0 terms of the RoboJEPA weights.

Fixed-candidate ranking is already in ARC-Bench ([arXiv:2609.05461](https://arxiv.org/abs/2609.05461)) and AD-WM ([arXiv:2609.30264](https://arxiv.org/abs/2609.30264)). This ladder adds the released RoboJEPA scale axis, imagination error on the same inputs, held-out kitchens, and the millimetre floor.

I built the benchmark because an end-to-end vision-language-action policy hides this comparison. RT-1, RT-2, OpenVLA, and π0 map an image and an instruction to a motor command, and a new scene has meant another training run. I did not run this bank on those policies. I ran it on a world model that writes the prediction down. A policy that never writes it down can carry the same ceiling, and the visible symptom is the next fine-tune.

## 10. The loop, and what I have not trained

The pieces above are one loop, and they are still four different computations. Refining an action, predicting its consequence, planning with that prediction, and observing the outcome do not become the same thing because they share a network.

\[
\text{Observe} \rightarrow \text{Represent} \rightarrow \text{Propose} \rightarrow \text{Predict} \rightarrow \text{Refine} \rightarrow \text{Act}
\]

A diffusion or flow policy proposes and refines the command. A world model predicts. Planning holds the predictor fixed. The replay, or the real video, is the observation that closes the loop. A vision-language-action policy runs several of these inside one forward pass. The pipeline is a way to combine the papers above. It is not a claim that any one of them implements every box.

What I want next, and have not trained:

A future you can look at, so a short reach is visible before the arm moves. A latent that orders the near misses, including a true DROID action scaled by about 10% and by about 25%. Language on the proposal, so a model can state the goal and later state which imagined future missed it. Then a fine-tune, by imitation and by reinforcement learning, with a diffusion or flow head. Memory only if training reads it.

The nearer experiment is smaller, and it is also untrained. The note in `breakage/MODEL_PLAN.md` is draft v1. A pixel world-action model in the LeWM style, on the order of 15 million to 100 million parameters, trained end to end, with no frozen V-JEPA encoder. An action-contrastive term. An optional teacher anchor. A preference head. A separate action head whose loss is blocked from the shared encoder, as an ablation. The gates, before any scale-up: beat Meta's 1B on the DROID rescales at 10% and at 25%, beat fine Spearman .418 on this reaching bench, separate success from failure on held-out DROID, and run closed-loop RoboCasa. A LeWM baseline on the converted DROID set takes 0.7 seconds per step and 13.8 GB peak on one GH200. That is a measured training cost, not a ranking result.

## Closing

What I find compelling is the shift from a single observation-to-action map toward a predictive decision. Generative policies model a distribution over actions. JEPAs learn a representation by predicting ahead. World-action models train that prediction and the action together. Differentiable planning uses the model to edit a candidate while the weights stay frozen.

None of this removes the need for feedback, or for a prediction that actually orders outcomes. On the RoboJEPA ladder, a closer imagined frame stopped buying a better choice at about 300 million parameters, and a loss that improved the choice made the imagined frame worse.

The direction I want is still the one in the loop above: learn representations that predict the consequences of actions, generate promising sequences, and use the model to decide which actions are worth taking. The benchmark is a measurement of that last step, on one score, one task, and one family of latent models.

### Papers and resources

- [Much Ado About Noising](https://arxiv.org/abs/2512.01809), Pan et al. The MIP construction is theirs.
- [LeWAM](https://arxiv.org/abs/2610.12818), Fu, Siebert, Halicki, and Balestriero. Author-reported success rates are cited as reported.
- [Long-WAM](https://arxiv.org/abs/2610.10528). The 63.3% to 78.7% figure is the abstract.
- [Ctrl-World](https://arxiv.org/abs/2510.10125), Guo, Shi, Chen, and Finn. The 44.7% figure is theirs.
- RoboJEPA and V-JEPA 2.1, Meta. RoboCasa. DROID.
- [Leaderboard](https://huggingface.co/spaces/Ryukijano/world-model-rankability) and [dataset](https://huggingface.co/datasets/Ryukijano/robojepa-rankability).

Scores derived from RoboJEPA weights are CC BY-NC-SA 4.0, and a ranking fine-tune inherits that licence. RoboCasa is MIT. The benchmark is for non-commercial research. A different score, a different task, or a representation trained on small motions could still plan. This post does not test those.
