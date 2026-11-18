"""REVISION-R1 11-03 (R1 comment 4): inverse-transformed BCAT / PROSE-FD shallow-water predictions for one test
trajectory, from the /eu symmetry dumps (pred padded to C=4, h in channel 0). For each model: GT, identity prediction,
and the inverse-transformed predictions of the row reflection (fr), the transposition (d1) and the 8-cell shift (sh8),
with the corrected scalar inverse and, for fr and d1, the previous vector inverse. Trajectory = median identity error
of BCAT; frame = last predicted frame. Writes a small npz for plotting."""
import numpy as np, sys
ROOT, OUT = sys.argv[1], sys.argv[2]
G = {"fr": (0, 1, 0), "d1": (1, 0, 0)}
def inv_buggy(p, g):
    t, fr, fc = g; C = p.shape[-1]
    if fr:
        p = np.flip(p, axis=1).copy()
        if C >= 2: p[..., 0] = -p[..., 0]
    if fc:
        p = np.flip(p, axis=2).copy()
        if C >= 2: p[..., 1] = -p[..., 1]
    if t:
        p = np.swapaxes(p, 1, 2).copy()
        if C >= 2: sw = list(range(C)); sw[0], sw[1] = 1, 0; p = p[..., sw]
    return p
def inv_scalar(p, g):
    t, fr, fc = g
    if fr: p = np.flip(p, axis=1)
    if fc: p = np.flip(p, axis=2)
    if t: p = np.swapaxes(p, 1, 2)
    return np.ascontiguousarray(p)
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).sum()) / (np.sqrt((b ** 2).sum()) + 1e-12))
out = {}; k = None
for m, tag in (("BCAT", "bcat"), ("PROSE-FD", "prose")):
    z = np.load(f"{ROOT}/_{tag}symE_id_shallow_water.npz"); print(m, z.files, z["pred"].shape, flush=True)
    P, Y = z["pred"], z["gt"]
    if k is None:
        e = np.sqrt(((P[..., 0] - Y[..., 0]) ** 2).sum((1, 2, 3)) / (Y[..., 0] ** 2).sum((1, 2, 3)))
        k = int(np.argsort(e)[len(e) // 2]); f = P.shape[1] - 1; out["traj"] = k; out["frame"] = f
    gt = Y[k, f, :, :, 0].astype(np.float32); pid = P[k, f, :, :, 0].astype(np.float32)
    out["gt"] = gt; out[f"{m}|id"] = pid; out[f"{m}|err|id"] = rel(pid, gt)
    for g in ("fr", "d1"):
        p4 = np.load(f"{ROOT}/_{tag}symE_{g}_shallow_water.npz")["pred"][k, f].astype(np.float32)   # (H, W, C)
        a = inv_scalar(p4[None, ..., :1], G[g])[0, ..., 0]; b = inv_buggy(p4[None], G[g])[0, ..., 0]
        out[f"{m}|{g}"] = a; out[f"{m}|{g}|buggy"] = b
        out[f"{m}|err|{g}"] = rel(a, gt); out[f"{m}|err|{g}|buggy"] = rel(b, gt)
    p4 = np.load(f"{ROOT}/_{tag}symE_sh8_shallow_water.npz")["pred"][k, f, :, :, 0].astype(np.float32)
    c = [np.roll(p4, (s, s), axis=(0, 1)) for s in (8, -8)]; i = int(np.argmin([rel(x, gt) for x in c]))
    out[f"{m}|sh8"] = c[i]; out[f"{m}|err|sh8"] = rel(c[i], gt); out[f"{m}|sh8_sign"] = (8, -8)[i]
    print(m, {x: round(out[x], 4) for x in out if x.startswith(m + "|err")}, flush=True)
np.savez_compressed(OUT, **out); print("wrote", OUT, "traj", k)
