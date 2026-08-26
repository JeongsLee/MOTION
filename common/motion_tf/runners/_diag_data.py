"""Diagnose whether the dt=1 caches contain samples that blow up instance_norm: per-channel std over the
INPUT window (T_in frames) ~0 => (x-m)/(s+1e-6) explodes. Also NaN/Inf check. Reports per family."""
import os, glob
import numpy as np

PRE = os.environ.get("PREBUILT_DIR", "/code-vol/data/prose/prebuilt")
N, T = 100000, 20
T_IN = 10
FAM_C = {"shallow_water": 1, "diff_react": 2, "com_ns": 4, "pdearena_ns": 3, "incom_ns": 3}


def files(fam):
    p = os.path.join(PRE, f"{fam}_n{N}_t{T}.npy")
    if os.path.exists(p):
        return [p]
    return sorted(glob.glob(os.path.join(PRE, f"{fam}_n{N}_t{T}_shards", "shard_*.npy")))


for fam, C in FAM_C.items():
    fs = files(fam)
    if not fs:
        print(f"{fam}: no cache", flush=True); continue
    stds = []          # per-sample per-channel input-window std
    nan = inf = nsamp = 0
    for f in fs:
        arr = np.load(f, mmap_mode="r")
        # subsample up to 400 rows per family for speed
        idx = np.linspace(0, arr.shape[0] - 1, min(400, arr.shape[0])).astype(int)
        for i in idx:
            u = np.asarray(arr[i, :T_IN, :, :, :C], dtype=np.float32)   # (T_in,H,W,C)
            nan += int(np.isnan(u).any()); inf += int(np.isinf(u).any())
            s = u.std(axis=(0, 1, 2))                                   # (C,) per-channel std over input window
            stds.append(s); nsamp += 1
        if len(stds) >= 400:
            break
    S = np.stack(stds, 0)                                               # (n,C)
    smin = S.min(); n_tiny = int((S < 1e-4).sum()); n_zero = int((S == 0).sum())
    print(f"{fam:14s} n={nsamp} C={C} | std min={smin:.3e} median={np.median(S):.3e} "
          f"| #(std<1e-4)={n_tiny} #(std==0)={n_zero} | NaN_rows={nan} Inf_rows={inf}", flush=True)
print("DIAG DONE", flush=True)
