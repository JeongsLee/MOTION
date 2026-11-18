"""Verify /eu/data/poseidon_ft: (1) every task file opens, shapes, NaN, per-trajectory std > 0 on train/val/test rows;
(2) the subset logic on NS-PwC, where the full assembled file exists: velocity_9.nc (last in lexicographic order) must equal
assembled rows [20000 - n_last, 20000), and our val/test rows must match."""
import os, glob, json, numpy as np, netCDF4, h5py
from huggingface_hub import hf_hub_download
R = "/eu/data/poseidon_ft"
for f in sorted(glob.glob(f"{R}/*/*.nc")):
    with netCDF4.Dataset(f) as nc:
        var = max(nc.variables, key=lambda v: nc.variables[v].ndim); v = nc.variables[var]; n = v.shape[0]
        idx = [0, n // 2, n - 361, n - 241, n - 1]; print(f"{os.path.basename(f)}: n={n}", flush=True)
        x = np.stack([np.asarray(v[i]) for i in idx])
        print(f"{os.path.basename(f):22s} {var:9s} {tuple(v.shape)} {v.dtype} size {os.path.getsize(os.path.realpath(f))/1e9:.2f} GB "
              f"nan {int(np.isnan(x).sum())} std/traj {[round(float(a.std()),3) for a in x]}", flush=True)
with h5py.File(f"{R}/Helmholtz/Helmholtz.h5") as h:
    ks = list(h.keys()); print("Helmholtz.h5", len(ks), "samples; Sample_0:", {k: h[ks[0]][k].shape for k in h[ks[0]]}, flush=True)
p = hf_hub_download("camlab-ethz/NS-PwC", "velocity_9.nc", repo_type="dataset", local_dir="/tmp/nspwc")
with netCDF4.Dataset(p) as a, netCDF4.Dataset(f"{R}/NS-PwC/NS-PwC.nc") as b:
    nl = a.dimensions["sample"].size; N = b.dimensions["sample"].size; off = N - nl
    ok = all(np.array_equal(np.asarray(a["velocity"][j]), np.asarray(b["velocity"][off + j])) for j in (0, nl - 361, nl - 241, nl - 1))
    print(f"NS-PwC ordering check: assembled N={N}, last chunk n={nl}, rows [{off},{N}) == velocity_9.nc: {ok}", flush=True)
for t in ("NS-SVS", "NS-BB", "NS-SL", "FNS-KF", "CE-RPUI"):
    print(t, json.load(open(f"{R}/{t}/{t}_sub.json")), flush=True)
print("VERIFY_DONE", flush=True)
