import numpy as np, os
def eps(P0, Pg_inv):
    den = np.sqrt((P0.astype(np.float64) ** 2).sum(axis=(2, 3, 4)))
    num = np.sqrt(((Pg_inv.astype(np.float64) - P0.astype(np.float64)) ** 2).sum(axis=(2, 3, 4)))
    return float(np.median((num / (den + 1e-12)).ravel()) * 100)
for model, pref in (("BCAT", "_bcatsymE"), ("PROSE-FD", "_prosesymE")):
    f0, ff = f"/eu/{pref}_id_cfdbench.npz", f"/eu/{pref}_frns_cfdbench.npz"
    if not (os.path.exists(f0) and os.path.exists(ff)): print("missing", model); continue
    P0 = np.load(f0)["pred"]; Pf = np.flip(np.load(ff)["pred"], axis=2).copy()   # no sign flip
    print(f"[{model} cfdbench] EPS_EQUIV(row mirror, scalar) = {eps(P0, Pf):.3f}%", flush=True)
d = "/eu/_bylfa_cfd_condsym.npz"
if os.path.exists(d):
    z = np.load(d); P0 = z["pid"]; Pf = z["pfr"]
    print(f"[MOTION cfdbench] EPS_EQUIV(row mirror, scalar) = {eps(P0, Pf):.3f}%", flush=True)
else:
    print("MOTION cfd dump missing")
print("DONE", flush=True)
