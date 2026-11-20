"""REVISION-R1 11-06: CFDBench row mirror (across the flow axis), one test trajectory, BCAT / PROSE-FD from the /eu
symmetry dumps (_{tag}symE_{id,frns}_cfdbench.npz; frns = input row-flipped without sign change, as in the equivalence test of Fig. 4d).
Inverse = row flip of the prediction. First 16 test trajectories (MOTION-NS dumps the same 16, see
run_cfd_onetraj.sh); for each: GT, identity prediction and inverse-transformed mirror prediction, last predicted
frame, the two flow channels. Writes a small npz for plotting.
  python cfd_onetraj.py /eu OUT.npz"""
import numpy as np, sys
ROOT, OUT = sys.argv[1], sys.argv[2]
N = 16
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).sum()) / (np.sqrt((b ** 2).sum()) + 1e-12))
out = {}
for m, tag in (("BCAT", "bcat"), ("PROSE-FD", "prose")):
    z = np.load(f"{ROOT}/_{tag}symE_id_cfdbench.npz"); P, Y = z["pred"], z["gt"]
    Pm = np.load(f"{ROOT}/_{tag}symE_frns_cfdbench.npz")["pred"]
    print(m, P.shape, Y.shape, Pm.shape, flush=True)
    f = P.shape[1] - 1
    out[f"{m}|gt"] = Y[:N, f, :, :, :2].astype(np.float32)
    out[f"{m}|id"] = P[:N, f, :, :, :2].astype(np.float32)
    out[f"{m}|fr_ns"] = np.flip(Pm[:N, f, :, :, :2], axis=1).astype(np.float32)   # inverse row mirror, no sign
    out[f"{m}|err_id"] = np.array([rel(out[f"{m}|id"][k], out[f"{m}|gt"][k]) for k in range(N)])
    out[f"{m}|err_fr_ns"] = np.array([rel(out[f"{m}|fr_ns"][k], out[f"{m}|gt"][k]) for k in range(N)])
    out[f"{m}|frame"] = f
    print(m, "err id", np.round(100 * out[f"{m}|err_id"], 3), "\n   mirror", np.round(100 * out[f"{m}|err_fr_ns"], 3), flush=True)
np.savez_compressed(OUT, **out); print("wrote", OUT)
