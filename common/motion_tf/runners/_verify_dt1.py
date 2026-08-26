"""Verify the dt=1 rebuilt caches: per-family shape, total sample count, and frame-to-frame relative
delta (small => smooth short-horizon => dt=1 consecutive; large ~0.4 => still strided/coarse-dt)."""
import os, glob
import numpy as np

PRE = os.environ.get("PREBUILT_DIR", "/code-vol/data/prose/prebuilt")
N, T = 100000, 20


def framedelta(a):                                   # a: (n,T,H,W,C) physical
    a = a.astype(np.float32)
    num = np.sqrt(((a[:, 1:] - a[:, :-1]) ** 2).sum(axis=(2, 3, 4)))
    den = np.sqrt((a[:, :-1] ** 2).sum(axis=(2, 3, 4))) + 1e-8
    return float((num / den).mean())


for fam in ["shallow_water", "diff_react"]:
    p = os.path.join(PRE, f"{fam}_n{N}_t{T}.npy")
    if not os.path.exists(p):
        print(f"{fam}: MISSING {p}", flush=True); continue
    arr = np.load(p, mmap_mode="r")
    print(f"{fam:14s} shape={arr.shape} framedelta={framedelta(np.asarray(arr[:8])):.4f}", flush=True)

for fam in ["com_ns", "pdearena_ns", "incom_ns"]:
    d = os.path.join(PRE, f"{fam}_n{N}_t{T}_shards")
    sh = sorted(glob.glob(os.path.join(d, "shard_*.npy")))
    if not sh:
        print(f"{fam}: NO SHARDS in {d}", flush=True); continue
    total = sum(int(np.load(s, mmap_mode="r").shape[0]) for s in sh)
    a0 = np.asarray(np.load(sh[0], mmap_mode="r")[:8])
    print(f"{fam:14s} nshards={len(sh)} total={total} sample_shape={a0.shape[1:]} "
          f"framedelta={framedelta(a0):.4f}", flush=True)
print("VERIFY DONE", flush=True)
