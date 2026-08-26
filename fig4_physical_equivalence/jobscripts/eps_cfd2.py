"""cfd equivariance defect, masked to the two real (u-like) channels, with a
triangle-inequality sanity check against each pass's own error."""
import numpy as np, os
def stats(P0, Pg, GT, nch):
    P0, Pg = P0[..., :nch].astype(np.float64), Pg[..., :nch].astype(np.float64)
    den = np.sqrt((P0 ** 2).sum(axis=(2, 3, 4)))
    eps = np.median((np.sqrt(((Pg - P0) ** 2).sum(axis=(2, 3, 4))) / (den + 1e-12)).ravel()) * 100
    out = [f"eps={eps:.3f}%"]
    if GT is not None:
        G = GT[..., :nch].astype(np.float64)
        dg = np.sqrt((G ** 2).sum(axis=(2, 3, 4))) + 1e-12
        e0 = np.median((np.sqrt(((P0 - G) ** 2).sum(axis=(2, 3, 4))) / dg).ravel()) * 100
        eg = np.median((np.sqrt(((Pg - G) ** 2).sum(axis=(2, 3, 4))) / dg).ravel()) * 100
        out.append(f"err_id={e0:.3f}% err_g={eg:.3f}% (bound {e0+eg:.3f}%)")
    return " | ".join(out)

for model, pref in (("BCAT", "_bcatsymE"), ("PROSE-FD", "_prosesymE")):
    f0, ff = f"/eu/{pref}_id_cfdbench.npz", f"/eu/{pref}_frns_cfdbench.npz"
    if not (os.path.exists(f0) and os.path.exists(ff)): print("missing", model); continue
    z0 = np.load(f0)
    P0, GT = z0["pred"], z0["gt"]
    Pf = np.flip(np.load(ff)["pred"], axis=2).copy()
    print(f"[{model} cfdbench, 2ch]", stats(P0, Pf, GT, 2), flush=True)

d = "/eu/_bylfa_cfd_condsym.npz"
if os.path.exists(d):
    z = np.load(d)
    print("[MOTION cfdbench, 2ch]", stats(z["pid"], z["pfr"], None, 2), flush=True)
print("DONE", flush=True)
