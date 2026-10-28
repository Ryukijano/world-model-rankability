# Notes on the blog draft

## Fix before publishing

1. **MIP is "minimum iterative policy", not "Multistep Inverse Problems".** From the abstract of arXiv 2512.01809: "a minimum iterative policy (MIP), a lightweight two-step regression-based policy, essentially matches the performance of flow GCPs". The same abstract gives a sharper version of your section 2 point: the gains of generative policies come "from iterative computation, as long as intermediate steps are supervised during training and this supervision is paired with a suitable level of stochasticity", not from modelling multiple modes.
2. **Section 6, the Mixture-of-Transformers attention mask.** I couldn't find this on le-wam.github.io. Check the LeWAM paper before keeping it, or cut it.
3. **Two papers called LeWAM.** The one you link (le-wam.github.io, UBC / Brown) trains latent prediction + action flow matching + SIGReg together. A different paper, arXiv 2609.27455 (Fang et al.), has the same name. Say which one you mean once.
4. **Add numbers where you have them.**
   - Long-WAM: going from 2.4 s to 19.2 s of context raises RoboCasa GR-1 success from 66.3% to 78.7%. Bidirectional initialisation shows no net gain.
   - LeWAM page: 93.5% mean success vs 50.9% for LeWM + CEM on its 10 tasks. SIGReg gives +14.7 pp planning gain over direct control, vs +2.7 without it.

## The bigger point

The draft is a literature survey, and every section ends with a disclaimer ("this is a schematic…", "should not be interpreted as…"). Readers will skim it. You have something most posts like this don't: measured results showing where world models break. I'd cut most of the disclaimers down to one line at the end, and add a section like the one below between your current sections 7 and 8. Every number in it comes from our own runs.

---

## What breaks when you ask a world model to choose an action

The pitch for world models in control is "imagine the consequences of each action, then pick the best one". So I tested exactly that.

**Setup.** I used Meta's released RoboJEPA world models at eight sizes (22M to 8B parameters) on 3,000 RoboCasa reaching states. For each state there are 64 candidate actions. I replayed each candidate in the simulator to get its true outcome, then asked each model to rank the candidates by how close its imagined future is to the goal.

**1. Scale stops helping long before prediction stops improving.** Ranking quality rises from 22M to 300M (Spearman ρ .747 → .784) and is flat from there to 8B (.789). Meanwhile imagination error keeps falling all the way to 8B (.273 → .218). Better at predicting the future does not mean better at choosing actions.

**2. There is a resolution floor of a few centimetres.** Take pairs of candidate actions whose true outcomes differ by less than 5 mm. Every model orders them correctly only 51–53% of the time, close to a coin flip. At 1–2 cm it's 59–61%, and only above 8 cm does it reach about 92%. The 300M, 1B, 2B, 4B and 8B curves sit on top of each other.

**3. The failures are systematic, not random.** 78% of the states where the 1B model picks badly (more than 5 cm worse than the best candidate) are also bad for the 22M model. 19% of states are bad for all eight sizes. That points at the training objective or the representation, not at capacity.

**4. Height is the weakest axis.** The action the 1B model picks ends on average 3.0 cm off target along x, 2.9 cm along y, and 3.5 cm in height. The best available candidate is 0.6–0.7 cm off on each axis, so most of that error comes from the model.

**5. The objective is what limits it.** I fine-tuned the 22M model for 1k steps with a loss that rewards ranking the better action higher (3 seeds, against a compute-matched control). On the unseen-kitchen states, ranking went from .794 to .823, and from .407 to .492 among the near-good candidates. The frozen 8B model scores .799 overall on the same states, so the fine-tuned 22M beats it with about 360× fewer parameters. But imagination error got *worse* (.242 → .268). Training for "imagine accurately" and training for "choose well" pull in different directions.

**6. On real robot video, I can reproduce the training trend but haven't tested ranking yet.** I trained RoboJEPA-22M from scratch on 5,000 raw DROID episodes on one GPU. Its 9-step imagination error on the held-out lab is 0.348, against 0.326 for Meta's released 10k-step checkpoint and 0.301 at 150k steps. Ranking on real video still needs "which action was better" labels, which real data almost never has. That is the open problem the next experiments target.

*Figure: resolution_floor.png. Caption: how often each model orders a pair of candidate actions correctly, against the true difference in their outcomes.*

---

Caveats to keep (one line each): one task family (reaching, one world-model step), one simulator, one scoring rule (latent L1 to the goal image).
