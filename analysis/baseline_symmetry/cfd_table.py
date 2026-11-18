"""REVISION-R1 (user 10-21): Fig. 4c CFDBench row on the two flow channels. Identity error and the error of the
symmetry-averaged prediction (identity + admitted mirror) for BCAT / PROSE-FD, on all real channels (as the submitted table)
and on the two flow channels (as MOTION). Several mirror implementations are evaluated so the submitted 3-channel value
(BCAT 0.377 / 0.561) can be matched before switching to 2 channels."""
import numpy as np, os, json, sys
ROOT, OUT = sys.argv[1], sys.argv[2]
def per_sf(a, b, ch):
    a, b = a[..., ch], b[..., ch]
    return np.sqrt(np.square(a - b).sum(axis=(2, 3, 4), dtype=np.float64)) / (np.sqrt(np.square(b).sum(axis=(2, 3, 4), dtype=np.float64)) + 1e-12)
def flip_sign(p, axis, sc):
    q = np.flip(p, axis=axis).copy()
    if sc is not None: q[..., sc] = -q[..., sc]
    return q
res = {}
for model, pref in (("BCAT", "_bcatsymE"), ("PROSE-FD", "_prosesymE")):
    f = lambda t: f"{ROOT}/{pref}_{t}_cfdbench.npz"
    with np.load(f("id")) as z: G = z["gt"].astype(np.float32); P0 = z["pred"].astype(np.float32)
    ch3 = [c for c in range(G.shape[-1]) if np.sqrt((G[..., c] ** 2).mean()) > 0]; ch2 = [c for c in ch3 if c < 2]
    row = {"real_channels": ch3, "id": {"3ch": 100 * float(per_sf(P0, G, ch3).mean()), "2ch": 100 * float(per_sf(P0, G, ch2).mean())}}
    vars_ = {}
    if os.path.exists(f("frns")):
        with np.load(f("frns")) as z: vars_["frns(row mirror, no sign)"] = np.flip(z["pred"].astype(np.float32), axis=2).copy()
    with np.load(f("fr")) as z: vars_["fr(row mirror, ch0 sign)"] = flip_sign(z["pred"].astype(np.float32), 2, 0)
    with np.load(f("fc")) as z: vars_["fc(col mirror, ch1 sign)"] = flip_sign(z["pred"].astype(np.float32), 3, 1)
    for k, Pm in vars_.items():
        A = 0.5 * (P0 + Pm)
        row[k] = {"replica_3ch": 100 * float(per_sf(Pm, G, ch3).mean()), "replica_2ch": 100 * float(per_sf(Pm, G, ch2).mean()),
                  "avg_3ch": 100 * float(per_sf(A, G, ch3).mean()), "avg_2ch": 100 * float(per_sf(A, G, ch2).mean()),
                  "defect_3ch": 100 * float(per_sf(Pm, P0, ch3).mean()), "defect_2ch": 100 * float(per_sf(Pm, P0, ch2).mean())}
    res[model] = row; print(model, json.dumps(row), flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("CFD_TABLE_OK", flush=True)
