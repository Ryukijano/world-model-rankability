import spaces  # must come before torch

import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import rjlive  # noqa: E402

WEIGHTS_REPO = "Ryukijano/robojepa-22m-rankability-live"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
weights_dir = snapshot_download(WEIGHTS_REPO, token=os.environ.get("HF_TOKEN"))
WM, LOAD_INFO = rjlive.load_world_model(weights_dir, DEVICE)

with np.load(os.path.join(os.path.dirname(__file__), "data", "anchors300.npz")) as _z:
    D = {k: _z[k] for k in _z.files}  # in memory: a lazy NpzFile handle breaks across the ZeroGPU worker fork
N = len(D["idx"])
SOURCE = {0: "random", 1: "goal-directed, coarse", 2: "goal-directed, fine"}
SHORT = {0: "random", 1: "coarse", 2: "fine"}
COLOURS = {0: "#9aa3b2", 1: "#2f5bd3", 2: "#e08a1e"}


@spaces.GPU(duration=30)
def run_model(i: int, scale: float) -> np.ndarray:
    """Score the state's 64 candidate actions and a copy of each with its translation scaled, on the GPU."""
    cands = D["cands"][i]
    edited = cands.copy()
    edited[:, :3] *= scale
    return rjlive.score_candidates(WM, D["init_imgs"][i], D["goal_imgs"][i], D["init_pose"][i],
                                   np.concatenate([cands, edited], 0), DEVICE)


def plot(dist_cm, score, src, pick, best):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for s in (0, 1, 2):
        m = src == s
        ax.scatter(dist_cm[m], score[m], s=22, c=COLOURS[s], label=SOURCE[s], marker="os^"[s])
    ax.scatter([dist_cm[pick]], [score[pick]], s=170, facecolors="none", edgecolors="#d6336c", linewidths=2, label="model's pick")
    ax.scatter([dist_cm[best]], [score[best]], s=170, facecolors="none", edgecolors="#2b8a3e", linewidths=2, marker="s", label="best candidate")
    ax.set_xlabel("true distance to goal after the action (cm), lower is better")
    ax.set_ylabel("model score, higher is better")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for a in (ax.xaxis.label, ax.yaxis.label):
        a.set_color("#7a8496")
    ax.tick_params(colors="#7a8496")
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#7a8496")
    ax.margins(0.08)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, labelcolor="#7a8496")
    fig.tight_layout()
    return fig


def analyse(state: int, scale: float):
    """Run RoboJEPA-22M on one benchmark state and compare its ranking of the 64 candidate actions with the simulator's ground truth."""
    i = int(state)
    pre = D["l1_pre"][i]
    try:
        scores = run_model(i, float(scale))
        live, edited = scores[:64], scores[64:]
        agree = spearmanr(live, pre).correlation
        status = f"Computed live on a ZeroGPU GPU. Matches the stored benchmark scores (Spearman {agree:.4f})."
    except Exception as e:  # quota exhausted, queue timeout, ...
        live, edited = pre, None
        status = f"The GPU call failed ({type(e).__name__}), so these are the stored benchmark scores for this state."
    dist_cm = D["dist"][i] * 100
    src = D["cand_source"][i]
    rho = spearmanr(live, -dist_cm).correlation
    pick, best = int(live.argmax()), int(dist_cm.argmin())
    order = np.argsort(-live)
    true_rank = np.argsort(np.argsort(dist_cm)) + 1
    df = pd.DataFrame({
        "Rank": np.arange(1, 11),
        "Motion": order[:10],
        "Score": [f"{v:.4f}" for v in live[order[:10]]],
        "Ends (cm from goal)": [f"{v:.1f}" for v in dist_cm[order[:10]]],
        "True rank": true_rank[order[:10]],
        "Type": [SHORT[int(s)] for s in src[order[:10]]],
    })
    text = (f"The arm starts {D['d0'][i] * 100:.1f} cm from the goal (kitchen {int(D['scene'][i])}). "
            f"The model picks candidate {pick}, which ends {dist_cm[pick]:.1f} cm from the goal. "
            f"The best candidate ends {dist_cm[best]:.1f} cm away, so the regret is {dist_cm[pick] - dist_cm[best]:.1f} cm. "
            f"Spearman between the model's order and the true order: {rho:.2f}.\n\n{status}")
    if edited is None:
        edit = "The edit needs the live GPU call, so it was not scored."
    else:
        e = float(edited[pick])
        rank = int((live > e).sum()) + 1
        edit = (f"Candidate {pick} with its motion scaled ×{scale:.2f} scores {e:.4f}, against {live[pick]:.4f} for the "
                f"original (difference {e - live[pick]:+.5f}), so the model ranks it {rank} of 65. The scaled action was not simulated, so there is no "
                f"true outcome to compare against. Over all 64 candidates, the scaled copy scores "
                f"{'higher' if (edited > live).mean() > 0.5 else 'lower'} than the original "
                f"{max((edited > live).mean(), (edited <= live).mean()) * 100:.0f}% of the time.")
    views = [D["init_imgs"][i][0], D["init_imgs"][i][1], D["goal_imgs"][i][0], D["goal_imgs"][i][1]]
    return (*views, text, df, plot(dist_cm, live, src, pick, best), edit)


INTRO = """# Live RoboJEPA-22M: ranking arm motions toward a goal
Pick a robot state. The model sees the current camera views and the goal, imagines where each of 64 candidate
motions would leave the arm, and ranks them. We compare its ranking with what happened when each motion was
replayed in the MuJoCo simulator. A run takes about 5 to 15 s on a shared GPU."""

ABOUT = """**Model.** Meta's released [RoboJEPA](https://github.com/facebookresearch/robo_jepa)-22M (150k training steps)
with the frozen [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) ViT-G/384 image encoder. The score of a
motion is minus the mean absolute difference between the imagined next frame and the goal frame, in the encoder's
feature space. We did not train this model.

**Data.** 300 of the 3,000 states from the pretraining kitchens of the
[rankability benchmark](https://huggingface.co/datasets/Ryukijano/robojepa-rankability) (RoboCasa Reach, Franka arm
on a mobile base, agent and wrist cameras). Each state has 32 random motions and 32 aimed at the goal. Everything
here is simulated; no physical robot is involved.

**Results across model sizes** are on the [leaderboard](https://huggingface.co/spaces/Ryukijano/world-model-rankability).
On the full benchmark, motions whose true outcomes differ by under 5 mm are ordered correctly about half the time,
the same as a coin flip.

**Licence.** The weights and vendored code are CC BY-NC-SA 4.0 (non-commercial use only)."""

with gr.Blocks(title="Live RoboJEPA-22M action ranking") as demo:
    gr.Markdown(INTRO)
    with gr.Row(equal_height=True):
        state = gr.Slider(0, N - 1, value=0, step=1, label=f"Robot state (0 to {N - 1})")
        scale = gr.Slider(0.5, 2.0, value=1.1, step=0.05, label="Edit: scale the model's pick by",
                          info="Tests whether the score reacts to a slightly longer or shorter motion")
        btn = gr.Button("Rank the 64 motions", variant="primary", scale=0, min_width=200)
    with gr.Row():
        with gr.Column(min_width=250):
            gr.Markdown("**Now**")
            with gr.Row():
                v0 = gr.Image(min_width=120, label="agent camera", height=180, show_label=True, interactive=False, buttons=[])
                v1 = gr.Image(min_width=120, label="wrist camera", height=180, show_label=True, interactive=False, buttons=[])
        with gr.Column(min_width=250):
            gr.Markdown("**Goal**")
            with gr.Row():
                g0 = gr.Image(min_width=120, label="agent camera", height=180, show_label=True, interactive=False, buttons=[])
                g1 = gr.Image(min_width=120, label="wrist camera", height=180, show_label=True, interactive=False, buttons=[])
    summary = gr.Markdown()
    with gr.Row():
        with gr.Column():
            table = gr.Dataframe(label="The model's top 10 motions (type: random, or aimed at the goal coarsely or finely)", interactive=False, wrap=True)
        with gr.Column():
            fig = gr.Plot(label="All 64 motions: model score against true outcome")
    edit = gr.Markdown()
    with gr.Accordion("About this demo", open=False):
        gr.Markdown(ABOUT)
    btn.click(analyse, [state, scale], [v0, v1, g0, g1, summary, table, fig, edit])

if __name__ == "__main__":
    demo.launch(mcp_server=True, theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"))
