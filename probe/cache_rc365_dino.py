"""Cache frozen DINOv3 ViT-L/16 patch features (fp16) for RoboCasa365 pretrain-atomic demos. Encoder-swap twin of
cache_rc365.py: identical episode subset (seeded permutation), identical kept frames and actions; only the encoder
(ViT-L 16px -> [V,15,20,1024] instead of ViT-G 16px -> [V,15,20,1664]) differs. Same output layout:
<out>/ep<idx>.feats.npy + ep<idx>.npz. No network access; dataset and weights pre-staged."""
import argparse, glob, json, os, sys, time
import av, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from score_anchors import R  # noqa: E402
from dino_encode import load_dino, encode_dino  # noqa: E402
from cache_rc365 import CAMS, decode_segment  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=f"{R}/data/rc365/pretrain-atomic"); p.add_argument("--out", required=True)
    p.add_argument("--n_episodes", type=int, required=True, help="size of the deterministic episode subset (seeded permutation)")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    p.add_argument("--stride", type=int, default=4); p.add_argument("--max_kept", type=int, default=64)
    p.add_argument("--dino_dir", default=f"{R}/ckpt/dinov3-vitl16"); p.add_argument("--image_size", type=int, nargs=2, default=[240, 320])
    p.add_argument("--time_budget", type=float, default=1500, help="seconds; stop cleanly before the 30-min walltime")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    info = json.load(open(f"{a.root}/meta/info.json")); fps = float(info["fps"])
    ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.root}/meta/episodes/*/*.parquet"))]).set_index("episode_index").sort_index()
    sel = np.random.default_rng(a.seed).permutation(len(ep))[: a.n_episodes]
    sel = sorted(int(ep.index[i]) for i in sel)[a.shard[0]::a.shard[1]]
    todo = [e for e in sel if not os.path.exists(f"{a.out}/ep{e:06d}.npz")]
    print(f"subset={a.n_episodes} shard={a.shard} episodes={len(sel)} todo={len(todo)} fps={fps} stride={a.stride} encoder=dinov3-vitl16", flush=True)
    if not todo:
        print("ALL_CACHED"); return
    model = load_dino(a.dino_dir)
    pq_cache = {}; n_done = 0
    for e in todo:
        if time.time() - t0 > a.time_budget:
            print(f"TIME_BUDGET reached after {n_done} episodes; resubmit to continue", flush=True); break
        r = ep.loc[e]; T = int(r["length"]); keep = np.arange(0, T, a.stride)[: a.max_kept]
        if len(keep) < 2:
            continue
        pqf = f"{a.root}/data/chunk-{int(r['data/chunk_index']):03d}/file-{int(r['data/file_index']):03d}.parquet"
        if pqf not in pq_cache:
            pq_cache = {pqf: pd.read_parquet(pqf, columns=["episode_index", "frame_index", "action", "observation.state", "task_index"])}
        d = pq_cache[pqf]; d = d[d["episode_index"] == e].sort_values("frame_index")
        assert len(d) == T and (d["frame_index"].values == np.arange(T)).all(), f"ep {e}: parquet rows {len(d)} != length {T}"
        act = np.stack(d["action"].values).astype(np.float32); st = np.stack(d["observation.state"].values).astype(np.float32)
        assert act.shape[1] == 12 and st.shape[1] == 16, (act.shape, st.shape)
        n_fr = int(keep[-1]) + 1
        frames = []
        for cam in CAMS:
            k = f"videos/observation.images.{cam}"
            vp = f"{a.root}/{k}/chunk-{int(r[k + '/chunk_index']):03d}/file-{int(r[k + '/file_index']):03d}.mp4"
            frames.append(decode_segment(vp, float(r[k + "/from_timestamp"]), float(r[k + "/to_timestamp"]), fps, n_fr)[keep])
        frames = np.stack(frames, 1)  # [N,V,H,W,3]
        feats = np.stack([encode_dino(model, fr, a.image_size) for fr in frames])  # [N,V,h,w,C]
        acts = np.stack([act[i:i + a.stride] for i in keep[:-1]])  # [N-1,stride,12]
        o = f"{a.out}/ep{e:06d}.npz"
        np.save(o[:-4] + ".feats.tmp.npy", feats); os.replace(o[:-4] + ".feats.tmp.npy", o[:-4] + ".feats.npy")
        np.savez(o + ".tmp.npz", acts=acts, state=st[keep], frame_idx=keep, episode_index=e, length=T,
                 task=str(r["tasks"][0]) if hasattr(r["tasks"], "__len__") else str(r["tasks"]), source_prefix=str(r["source_prefix"]))
        os.replace(o + ".tmp.npz", o); n_done += 1
        if n_done % 10 == 1:
            print(f"ep {e}: T={T} kept={len(keep)} feats={feats.shape} acts={acts.shape} {time.time()-t0:.0f}s", flush=True)
    print(f"cached {n_done}/{len(todo)} episodes in {time.time()-t0:.0f}s -> {a.out}", flush=True)
    if n_done == len(todo):
        print("ALL_CACHED")


if __name__ == "__main__":
    main()
