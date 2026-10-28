# From Diffusion Policies to World-Action Models: can a robot's imagination tell it what to do?

*Notes on generative robot policies, joint-embedding predictive architectures and latent-space planning, and a benchmark
that tests whether the imagination actually steers the choice of action. Everything measured here was measured in
simulation or on recorded robot video. No physical robot was used.*

Robotics sits at an intersection of perception, prediction and decision-making. A robot observes the world, proposes
an action, and then has to deal with what actually happens. Objects move unexpectedly, contacts are imprecise, and
small errors accumulate over a long sequence of actions.

This makes robot learning more than mapping observations to actions. It is also a problem of learning how the world
responds to those actions.

Over the past few weeks I have been reading in one direction and measuring in another. The reading covers diffusion
and flow-matching policies, iterative refinement, JEPAs, world-action models and differentiable planning. The
measuring asks one narrow question of the world models that already exist: when they imagine the result of each
candidate action and pick the closest to the goal, do they pick the one that works?

The first half of this post is the reading. The second half is what the measurements say, and what I am building in
response.

## Part 1: the ideas

### 1. The problem with simply imitating demonstrations

A conventional imitation-learning policy learns a mapping from observations to actions:

\[
a_t = \pi_\theta(o_t)
\]

In practice most robot policies predict an action chunk rather than a single action:

\[
A_t = (a_t, a_{t+1}, \ldots, a_{t+H-1})
\]

Chunking improves temporal consistency and reduces how often the policy has to be queried. But behavioural cloning
has a familiar weakness: the policy is trained on demonstrations and deployed on trajectories partly determined by
its own mistakes. A slightly wrong action moves the robot into a state that was rare or absent in training, and the
next prediction has to operate outside the distribution it learned.

### 2. Why generative action policies work

Diffusion policies generate actions by iteratively transforming noise into structured predictions. Flow-matching
policies learn a vector field that transports samples toward the action distribution. Both can represent multimodal
behaviour: when several actions are plausible, the policy does not have to average them.

But is multimodality the reason they work? [Much Ado About Noising](https://arxiv.org/abs/2512.01809) (Pan et al.,
ICLR 2026) compares generative and regression policies across 28 behaviour-cloning tasks under matched architectures
and training. Their finding is that the advantage is tied to stochasticity during training and to supervised
iterative computation, not to distribution matching as such. Flow-based policies keep an edge on some high-precision
tasks.

So the question becomes: which properties of the learning and inference procedure actually improve control?

### 3. MIP: can a policy improve its own prediction?

The same paper proposes the Minimal Iterative Policy (MIP). A shared network is trained on clean and perturbed action
inputs and supervised to recover the demonstrated action. At inference it runs two stages:

\[
\hat A_0 = f_\theta(o, 0), \qquad \hat A_1 = f_\theta(o, \hat A_0)
\]

The first proposes, the second refines. (These equations are an illustration; the scaling and conditioning of the
noisy input are part of the method.) A two-step regression policy roughly matches flow-based control on their
benchmarks. Some of the value of diffusion-style control comes from how computation is organised, not only from the
distribution being modelled.

### 4. From predicting actions to predicting the world

Action generation alone does not tell us whether an action will produce a good outcome. World models learn how the
environment responds. Encode the observation,

\[
z_t = E(o_t),
\]

then predict how the representation changes under an action:

\[
\hat z_{t+1} = F_\phi(z_t, a_t).
\]

Joint-embedding predictive architectures (JEPAs) do this prediction in representation space instead of pixel space,
so the model does not spend capacity on appearance details that are irrelevant to control. For robotics the
opportunity is obvious: learn representations of the world, learn how actions change them, and use the model to make
better decisions.

### 5. LeWAM: world prediction and action generation in one model

[LeWAM](https://le-wam.github.io/) combines four things that are usually trained separately: a representation of
observations, prediction of future representations, flow-matching action generation, and SIGReg regularisation of
the representation space. Schematically,

\[
\mathcal L = \lambda_{\mathrm{pred}} \mathcal L_{\mathrm{pred}}
+ \lambda_{\mathrm{action}} \mathcal L_{\mathrm{flow}}
+ \lambda_{\mathrm{reg}} \mathcal L_{\mathrm{SIGReg}}.
\]

The joint problem matters because a representation trained only for visual similarity is not necessarily the best
one for control, and a policy trained only to imitate may miss the structure that determines whether its actions
work.

### 6. Attention masks are part of the model

In joint world-and-action training the dynamics branch must see the clean action to predict its consequence, and
the policy branch must not see it, or it learns to copy the answer. LeWAM handles this with a structured attention
mask across token groups (history, future-state queries, clean actions, goal, noisy actions). Blocking one edge is
not enough if the same information can enter an intermediate token and travel through later layers. The mask is an
information-flow contract, and it defines part of the learning problem.

### 7. Differentiable latent MPC

With a differentiable dynamics model, candidate actions can be optimised against predicted outcomes:

\[
J(A) = \tfrac{1}{2}\,\lVert F_\phi(z_t, A) - z_g \rVert_2^2 + \tfrac{\lambda}{2}\,\lVert A - A_0 \rVert_2^2,
\qquad A^* = \arg\min_A J(A).
\]

The model weights stay frozen; gradients flow into the action variables. Execute a prefix, observe, replan. The
catch is that gradients exploit model error. A low predicted cost does not guarantee a good real outcome.

That catch is the whole of Part 2.

## Part 2: does the imagination actually steer?

Every planner in Part 1 rests on the same assumption: if the model imagines the future well, its ranking of
candidate actions is good enough to act on. I built a benchmark to test this directly on the best open world models
available, Meta's RoboJEPA family (22M to 8B parameters, all running on a frozen V-JEPA 2.1 encoder), and then used
it to find where the assumption fails.

Data, evaluator and leaderboard: [Ryukijano/robojepa-rankability](https://huggingface.co/datasets/Ryukijano/robojepa-rankability)
and [Ryukijano/world-model-rankability](https://huggingface.co/spaces/Ryukijano/world-model-rankability).
Code: [github.com/Ryukijano/world-model-rankability](https://github.com/Ryukijano/world-model-rankability).

### 8. The benchmark

3,499 robot-arm states from RoboCasa kitchens: 3,000 from the 30 kitchens the models trained on and 499 from 10
held-out kitchens. For each state, 64 fixed candidate motions: 32 random and 32 aimed at the goal, half of them
coarse and half fine. The ground truth for each candidate is its real outcome, obtained by replaying it in MuJoCo from
a saved simulator state. Replays of the same action agree to within 0.2 mm.

Each model scores the same candidates with Meta's own planning cost: minus the distance between the imagined next
frame and the goal frame in feature space. We then ask how well that score orders the candidates by their true
outcomes.

| Model | Overall Spearman | Fine Spearman | Regret | Imagination error |
|---|---:|---:|---:|---:|
| 22M | .747 | .359 | 5.7 cm | .273 |
| 300M | .784 | .411 | 5.1 cm | .240 |
| 1B | .789 | .418 | 5.0 cm | .223 |
| 8B | .789 | .409 | 5.0 cm | .218 |

*3,000 pretraining-kitchen states. Fine Spearman: ordering among the goal-directed candidates only. Regret: how much
further from the goal the model's pick ends than the best available candidate. Imagination error: distance between
imagined and real next frame.*

Ranking improves from 22M to about 300M and then stays flat to 8B, within ±.002. The imagined future keeps getting
more accurate the whole way. Among candidates already aimed at the goal, the Spearman tops out at .42 and the single
best action is picked about 5% of the time at every size from 300M up.

Imagination quality and choice quality are different quantities, and past about 1B parameters they decouple.

### 9. Where the choices break

Split candidate pairs by how different their true outcomes are, and ask how often the model orders each pair
correctly:

![Resolution floor](../figures/resolution_floor.png)

- Outcomes under 5 mm apart: 51 to 53% correct, for every model size. A coin flip.
- 1 to 2 cm apart: 59 to 61%.
- Over 8 cm apart: 88 to 93%.

The floor does not move with scale. Three more things we measured:

- **Failures are shared.** Of the states where the 1B model's pick is more than 5 cm worse than the best candidate,
  78% are also failures for 22M, and 19% fail for all eight sizes. The cause is something the models share, most
  likely the objective or the representation, not capacity.
- **Height is the weakest direction.** The 1B pick ends 3.0, 2.9 and 3.5 cm off in x, y and height, while the best
  candidate on offer was 0.6 to 0.7 cm off. Most of the error is the model's.
- **It is not the scoring.** ThinkWorld's training reward compares features only on the image patches that change
  most. We tried the same for scoring: restrict the cost to the top 50%, 25% or 10% of changing patches. Fine
  Spearman for 1B fell from .418 to .361, .306 and .260. Ordering got worse the harder we restricted, at every size.
  The action-relevant signal is spread over the frame; the predictor itself does not resolve fine differences.

This is the section's claim: a reproducible floor, measured against simulator ground truth, that no released model
size escapes.

### 10. It is not just simulation

The same idea on real robot video: 256 held-out DROID clips from a lab the models never trained on. The model scores
the true action sequence against 15 from other episodes, and against edited copies of the true action.

| Test | Result |
|---|---|
| Pick the true action out of 16 | .996 to 1.0 top-1, every checkpoint (chance .0625) |
| True vs sign-flipped direction | .91 to 1.0 |
| True vs rotation zeroed | .94 to 1.0 |
| True vs delayed one step, or all-zero | 1.0 |
| **True vs translation ×0.9** | **22M: .48 to .62. 1B: .79** |
| **True vs translation ×1.1** | **22M: .50 to .65. 1B: .93** |

Coarse action identity is solved: wrong episode, wrong direction, missing rotation, wrong timing are all caught. What
fails is magnitude. A 10% change in arm speed, about 4 mm per world-model step, is near chance for the 22M models.
Scale helps here, to 79% for 1B accumulated over about 15 steps, but that is a coarser, summed test than the
single-step floor and still leaves a fifth of the edits undetected.

### 11. One thing that works, and its cost

Fine-tuning RoboJEPA-22M with a ranking loss on replayed outcomes, 20 states from each of 25 pretraining kitchens,
raised held-out overall Spearman to .823 and fine Spearman to .492, above the frozen 8B, against a compute-matched
control at .794 and .407. Its imagination error rose from .242 to .268. Teaching the model to choose made it imagine
worse. Section 5's joint objective is not free; the two pull on the same features.

## Part 3: what I am building

Put Parts 1 and 2 together and the loop from Section 7 has a specific broken link: Predict feeds Refine, and
Predict's score is flat exactly where Refine needs a gradient. Better imagination will not fix it (Section 8), and a
better cost function will not fix it (Section 9). The floor is in what the predictor represents, and the released
models cannot see past their frozen encoder.

So the model is a small world-action model trained from pixels on open real video (DROID, AgiBot, Open-AoE),
LeWM-style, with three changes aimed at the measurements above:

- **Hard counterfactual actions as negatives.** The true action scaled ×0.9 or ×1.1, or nudged by millimetres,
  generated on the fly from any clip. No labels and no simulator. Existing action-contrastive losses compare against
  actions from other episodes, which Section 10 shows is already solved; the signal has to come from the near misses.
- **Gradient insulation between choosing and imagining.** The action head's gradients are blocked from the shared
  encoder, so the trade-off in Section 11 cannot happen through the representation.
- **Planning evaluated on outcomes.** Gradient MPC and CEM through the trained model, scored by replayed outcome on
  the benchmark and by task success in RoboCasa closed loop, not by predicted cost.

The bar is fixed before training: beat Meta's 1B on the ±10% real-video test (.79 and .93) and its .418 fine
Spearman, at a fraction of the size. If it cannot, the plan changes before any scale-up.

### Open questions from Part 1, now with a place to answer them

- Generative inference versus iterative refinement: compare a flow head and a two-step MIP head as the proposer, on
  the same bench.
- Representation quality for control: the benchmark's fine Spearman is exactly the "useful for predicting action
  consequences, not just encoding visual structure" measure.
- Planning horizon versus model error: regret per step in the closed-loop RoboCasa runs.
- Evaluation beyond success rate: resolution floor, per-axis residual, shared failures, and prediction error reported
  together, so a gain in one cannot hide a loss in another.

## Try it

- [Leaderboard and state explorer](https://huggingface.co/spaces/Ryukijano/world-model-rankability): every number
  above, and 3,499 states to click through.
- [Dataset and evaluator](https://huggingface.co/datasets/Ryukijano/robojepa-rankability): score your own model's
  candidates and submit.
- Live demo: pick a state, scale the model's own pick by ×0.9 or ×1.1, and watch what the score does. All eight
  RoboJEPA sizes run on ZeroGPU.

### Papers and resources

- [Much Ado About Noising: Dispelling the Myths of Generative Robotic Control](https://arxiv.org/abs/2512.01809)
- [LeWAM: End-to-End World and Action Modeling with JEPAs](https://le-wam.github.io/)
- [Long-WAM](https://nvlabs.github.io/LongLive/Long-WAM/)
- [ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning](https://huggingface.co/papers/2610.37666)
- RoboJEPA and V-JEPA 2.1 (Meta); RoboCasa; DROID

*Part 1 is a synthesis of the cited work and my own questions. Part 2 is measured: all numbers come from the linked
dataset and code. Scores derived from RoboJEPA are CC BY-NC-SA 4.0; RoboCasa is MIT. Nothing here is a claim about
performance on a physical robot.*
