"""Numpy-only hand-check of a cached RoboCasa365 feature dir (runs on the login node). Prints shapes, action stats per dim, alignment checks."""
import sys, glob, json, numpy as np
d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
files = sorted(glob.glob(f"{d}/ep*.npz"))[:n]; bad = []; acts = []; T = 0; tasks = {}
for f in files:
    m = np.load(f, allow_pickle=True); F = np.load(f.replace(".npz", ".feats.npy"), mmap_mode="r")
    a = m["acts"]; ok = (F.shape[1:] == (2, 15, 20, 1664)) and (a.shape == (F.shape[0] - 1, 4, 12)) and np.isfinite(a).all()
    fr = m["frames"] if "frames" in m else None
    if fr is not None: ok &= bool(np.all(np.diff(fr) == 4)) and len(fr) == F.shape[0]
    if not ok: bad.append(f)
    acts.append(a.reshape(-1, 12)); T += len(a); tasks[str(m["task"])] = tasks.get(str(m["task"]), 0) + 1
A = np.concatenate(acts)
print(f"episodes={len(files)} transitions={T} bad={len(bad)} tasks={len(tasks)}"); print("bad:", bad[:5])
names = ["base_x","base_y","base_yaw","torso","ctrl_mode","dx","dy","dz","rx","ry","rz","grip"]
for i, nm in enumerate(names):
    v = A[:, i]; print(f"{nm:9s} mean={v.mean():+.3f} std={v.std():.3f} min={v.min():+.2f} max={v.max():+.2f} frac_nonzero={np.mean(np.abs(v)>1e-6):.2f}")
f0 = np.load(files[0].replace(".npz", ".feats.npy"), mmap_mode="r"); x = np.asarray(f0[:4], dtype=np.float32)
print(f"feat sample: mean={x.mean():.4f} std={x.std():.4f} |z1-z0| mean={np.abs(x[1]-x[0]).mean():.4f} |z3-z0| mean={np.abs(x[3]-x[0]).mean():.4f}")
print("top tasks:", sorted(tasks.items(), key=lambda kv: -kv[1])[:8])
