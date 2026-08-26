import os, glob, numpy as np
from netCDF4 import Dataset
A = "/eu/data/poseidon/_assembled"; OUT = "/seoul/fig3_local"; os.makedirs(OUT, exist_ok=True)
NTAKE = 16
SPEC = {
 "NS-PwC": dict(var="velocity", ch2slot=[0,1,2]),
 "ACE": dict(var="solution", ch2slot=[2], no_channel=True),
 "Wave-Layer": dict(var="solution", ch2slot=[2,6]),
 "Poisson-Gauss": dict(steady=("source","solution"), ch2slot=[2,7]),
}
print("ASSEMBLED:", sorted(os.listdir(A)))
for ds, sp in SPEC.items():
    if "steady" in sp:
        nc = Dataset(f"{A}/{ds}.nc"); vsrc = nc.variables[sp["steady"][0]]; vsol = nc.variables[sp["steady"][1]]
        n = vsrc.shape[0]
        def rd(i):
            s = np.asarray(vsrc[i], np.float32); u = np.asarray(vsol[i], np.float32)
            return np.stack([s.squeeze(), u.squeeze()], 0)[None]   # (1,2,H,W): t-axis=1
        C = 2
    else:
        shards = os.path.join(A, f"{ds}_shards")
        if os.path.isdir(shards):
            fs = sorted(glob.glob(shards+"/*.npy")); arrs=[]; ns=[]
            for f in fs: ns.append(np.load(f, mmap_mode="r").shape[0])
            n = sum(ns)
            def rd(i, fs=fs, ns=ns):
                for f, m in zip(fs, ns):
                    if i < m:
                        a = np.asarray(np.load(f, mmap_mode="r")[i], np.float32); return a
                    i -= m
        else:
            nc = Dataset(f"{A}/{ds}.nc"); v = nc.variables[sp["var"]]; n = v.shape[0]
            noch = sp.get("no_channel", False)
            def rd(i, v=v, noch=noch):
                a = np.asarray(v[i], np.float32)
                return a[:, None] if noch else a                    # (T,C,H,W)
        C = rd(0).shape[1]
    # global stats: first min(64, n-240) samples (loader convention)
    ssum = np.zeros(C); ssq = np.zeros(C); cnt = 0
    for i in range(min(64, n-240)):
        a = rd(i).astype(np.float64)
        ssum += a.sum(axis=(0,2,3)); ssq += (a**2).sum(axis=(0,2,3)); cnt += a.shape[0]*a.shape[2]*a.shape[3]
    m = (ssum/cnt).astype(np.float32); s = (np.sqrt(np.maximum(ssq/cnt-(ssum/cnt)**2,0))+1e-6).astype(np.float32)
    idx = list(range(n-NTAKE, n))
    traj = np.stack([rd(i) for i in idx])                            # (16,T,C,H,W)
    np.savez_compressed(f"{OUT}/{ds}_test{NTAKE}.npz", traj=traj, mean=m, std=s,
                        idx=np.array(idx), n_total=n, ch2slot=np.array(sp["ch2slot"]))
    print(f"{ds}: n={n} traj{traj.shape} mean={m} std={s} -> saved")
print("EXTRACT_DONE")
