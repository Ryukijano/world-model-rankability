"""Rankability probe, stage 1: fixed anchors x fixed candidate action chunks with simulator ground truth.

For each (scene_seed, reach_task_seed) reach episode: reset to init + goal frames, snapshot the MuJoCo state, sample K
candidate 7-D actions (translation in the maxnorm ball, zero rotation/gripper, matching mpc_utils.cem's proposal
family), replay each from the snapshot through the blocking env and record final EE pose, reach reward, success and
distance-to-goal. Writes one .npz per anchor (images + candidates + outcomes). No world model here: scoring is stage 2.
Determinism check: candidate 0 is replayed twice; the two final EE positions must agree to < 1e-4 m.
"""
import argparse, json, os, sys, time
import numpy as np

R = "/projects/u6xn/rankability"
sys.path[:0] = [f"{R}/robo_jepa", f"{R}/robo_jepa/scripts/deployment"]
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from action_components import split_7d_action  # noqa: E402
from robots.robocasa.robot_env import RoboCasaEnv  # noqa: E402
from robots.robocasa.blocking_robot_env import ReachBlockingRoboCasaEnv  # noqa: E402


def _ball(t, maxnorm):
    n = np.linalg.norm(t, axis=1, keepdims=True)
    return np.where(n < maxnorm, t, t / n * maxnorm)


def sample_candidates(rng, k, maxnorm, std, to_goal):
    """Fixed bank of k translations (7-D action: [dx,dy,dz, rx,ry,rz, grip], rotation/gripper 0).
    source 0 (k/2): cem iteration-0 family: random direction * radius (r~U(0,1)*std), projected onto the maxnorm ball.
    source 1 (k/4): goal-directed: (goal-eef) * U(0.5,1.5) + N(0, 0.03) per axis   (expert-perturbed, coarse noise)
    source 2 (k/4): goal-directed: (goal-eef) * U(0.8,1.2) + N(0, 0.01) per axis   (expert-perturbed, fine noise)
    Goal position is used only to build the bank; labels come from simulator replay."""
    k0, k1 = k // 2, k // 4; k2 = k - k0 - k1
    d = rng.normal(size=(k0, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    t0 = d * rng.uniform(size=(k0, 1)) * std
    t1 = to_goal[None] * rng.uniform(0.5, 1.5, size=(k1, 1)) + rng.normal(scale=0.03, size=(k1, 3))
    t2 = to_goal[None] * rng.uniform(0.8, 1.2, size=(k2, 1)) + rng.normal(scale=0.01, size=(k2, 3))
    t = _ball(np.concatenate([t0, t1, t2]), maxnorm)
    src = np.array([0] * k0 + [1] * k1 + [2] * k2, np.int8)
    return np.concatenate([t, np.zeros((k, 4))], axis=1).astype(np.float32), src


def snapshot(env):
    sim = env._env.sim
    return sim.get_state(), (None if env._base_joint_qpos is None else np.array(env._base_joint_qpos, dtype=np.float64))


def restore(env, snap):
    st, base = snap
    sim = env._env.sim; sim.set_state(st); sim.forward()
    if base is not None:
        env._base_joint_qpos = base.copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--scene_seeds", type=int, nargs="+", default=[0])
    p.add_argument("--task_seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--k", type=int, default=64); p.add_argument("--maxnorm", type=float, default=0.3)
    p.add_argument("--std", type=float, default=0.3); p.add_argument("--rng_seed", type=int, default=0)
    p.add_argument("--cameras", nargs="+", default=["robot0_agentview_left", "robot0_eye_in_hand"])
    p.add_argument("--chunk_size", type=int, default=4); p.add_argument("--max_anchors", type=int, default=500)
    p.add_argument("--worker", type=int, default=0); p.add_argument("--nworkers", type=int, default=1)
    p.add_argument("--reach_max_distance", type=float, default=0.30)
    p.add_argument("--kitchen_split", default="pretrain", help="pretrain (~2,500-scene pool, kitchen drawn by scene seed) | target (10 held-out kitchens, seed%%10)")
    p.add_argument("--context_steps", type=int, default=0,
                   help="random context chunks executed after reset (std --ctx_std) before the anchor; frames/poses of every step are kept for the history axis")
    p.add_argument("--ctx_std", type=float, default=0.05)
    p.add_argument("--store_post_frames", type=int, default=0, help="store post-action frames for the first N candidates (prediction-error axis)")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    meta = dict(vars(a), git_sha=os.popen(f"git -C {R}/robo_jepa rev-parse HEAD").read().strip(), t0=time.time())
    json.dump(meta, open(f"{a.out}/meta_w{a.worker}.json", "w"), indent=1)
    n_done = 0
    for scene in a.scene_seeds:
        base = RoboCasaEnv(task_name="CloseStandMixerHead", camera_names=a.cameras, camera_width=320, camera_height=240,
                           seed=scene, robot="PandaOmron", chunk_size=a.chunk_size,
                           wm_embodiment="franka_omron_state_delta", reach_task=True, reach_task_seed=scene,
                           reach_max_distance=a.reach_max_distance, reach_clear_radius=1.0, kitchen_split=a.kitchen_split)
        env = ReachBlockingRoboCasaEnv(base, reach_task_seed=scene, reach_min_distance=0.0, reach_max_distance=a.reach_max_distance,
                                       reach_clear_radius=1.0, kitchen_split=a.kitchen_split, pos_tol=0.001, max_steps=200)
        for ts in a.task_seeds[a.worker :: a.nworkers]:
            if n_done >= a.max_anchors or os.path.exists(f"{a.out}/anchor_s{scene:03d}_t{ts:04d}.npz"):
                continue
            t = time.time()
            goal, _ = env.reset(mode="goal", seed=scene, reach_task_seed=ts)
            init, _ = env.reset(mode="init", seed=scene, reach_task_seed=ts)
            hist = [init]; ctx_actions = np.zeros((0, 7), np.float32)
            if a.context_steps > 0:
                crng = np.random.default_rng([a.rng_seed, scene, ts, 99])
                ctx_actions = np.zeros((a.context_steps, 7), np.float32)
                ctx_actions[:, :3] = _ball(crng.normal(scale=a.ctx_std, size=(a.context_steps, 3)), a.maxnorm)
                for ca in ctx_actions:
                    init, _, _, _ = env.step(split_7d_action(ca)); hist.append(init)
            snap = snapshot(base)
            eef0 = np.array(base._get_eef_pos()); goal_pos = np.array(getattr(base, "_reach_goal_world", np.full(3, np.nan)))
            # env eef_translation is in the mobile-base frame (target_world = eef + R_wb @ a[:3]); express goal direction there
            R_wb = np.asarray(base._get_base_world_R(), dtype=np.float64)
            to_goal_base = R_wb.T @ (goal_pos - eef0)
            cands, src = sample_candidates(np.random.default_rng([a.rng_seed, scene, ts]), a.k, a.maxnorm, a.std, to_goal_base)
            order = list(range(a.k)) + [0]  # replay candidate 0 again for determinism
            eef_final = np.zeros((len(order), 3)); rew = np.zeros(len(order)); succ = np.zeros(len(order), bool)
            dist = np.full(len(order), np.nan); post_imgs = []; post_pose = []
            for j, ci in enumerate(order):
                restore(base, snap)
                obs, r, done, info = env.step(split_7d_action(cands[ci]))
                if j < a.store_post_frames:
                    post_imgs.append(np.stack([obs[k] for k in sorted(obs) if k.startswith("img_")])); post_pose.append(obs.get("pose", np.zeros(7)))
                eef_final[j] = base._get_eef_pos(); rew[j] = r; succ[j] = bool(info.get("success", obs.get("success", False)))
                dist[j] = info.get("reach_distance_to_goal", np.nan)
            det_err = float(np.linalg.norm(eef_final[-1] - eef_final[0]))
            np.savez_compressed(f"{a.out}/anchor_s{scene:03d}_t{ts:04d}.npz", scene=scene, task_seed=ts, kitchen_split=a.kitchen_split,
                                goal_imgs=np.stack([goal[k] for k in sorted(goal) if k.startswith("img_")]),
                                init_imgs=np.stack([init[k] for k in sorted(init) if k.startswith("img_")]),
                                init_proprio=init.get("proprio_robocasa", np.zeros(0)), init_pose=init.get("pose", np.zeros(0)),
                                hist_imgs=np.stack([np.stack([h[k] for k in sorted(h) if k.startswith("img_")]) for h in hist]),
                                hist_pose=np.stack([h.get("pose", np.zeros(7)) for h in hist]),
                                hist_proprio=np.stack([h.get("proprio_robocasa", np.zeros(14)) for h in hist]),
                                ctx_actions=ctx_actions, eef0=eef0, goal_pos=goal_pos, R_wb=R_wb, to_goal_base=to_goal_base,
                                cands=cands, cand_source=src, d0=float(getattr(base, "_reach_d0", np.nan)), eef_final=eef_final[: a.k], reward=rew[: a.k], success=succ[: a.k],
                                dist=dist[: a.k], det_err=det_err,
                                post_imgs=np.stack(post_imgs) if post_imgs else np.zeros((0,), np.uint8), post_pose=np.stack(post_pose) if post_pose else np.zeros((0, 7)))
            n_done += 1
            print(f"anchor scene={scene} ts={ts} succ={succ[:a.k].mean():.2f} rew[min/med/max]="
                  f"{rew[:a.k].min():.3f}/{np.median(rew[:a.k]):.3f}/{rew[:a.k].max():.3f} det_err={det_err:.2e} "
                  f"{time.time()-t:.1f}s", flush=True)
        env.close() if hasattr(env, "close") else None
    print("done", n_done, "anchors", f"{(time.time()-meta['t0'])/60:.1f} min")


if __name__ == "__main__":
    main()
