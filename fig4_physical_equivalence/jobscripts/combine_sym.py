import numpy as np, glob, os, sys
PFX = sys.argv[1]  # _bcatsym or _prosesym
fams = sorted(set(os.path.basename(p)[len(PFX)+4:-4] for p in glob.glob(f"/eu/{PFX}_id_*.npz")))
print("families:", fams, flush=True)
def inv(p, tag):
    fr = tag in ("fr", "r180"); fc = tag in ("fc", "r180")
    if fr:
        p = np.flip(p, axis=2).copy()
        if p.shape[-1] >= 2: p[..., 0] = -p[..., 0]
    if fc:
        p = np.flip(p, axis=3).copy()
        if p.shape[-1] >= 2: p[..., 1] = -p[..., 1]
    return p
for fam in fams:
    gt = np.load(f"/eu/{PFX}_id_{fam}.npz")["gt"]
    preds = {}
    for tag in ("id", "fr", "fc", "r180"):
        f = f"/eu/{PFX}_{tag}_{fam}.npz"
        if os.path.exists(f):
            preds[tag] = inv(np.load(f)["pred"], tag)
    def rel(p):
        num = np.sqrt(((p - gt) ** 2).sum((2, 3, 4))); den = np.sqrt((gt ** 2).sum((2, 3, 4))) + 1e-12
        return 100 * (num / den).mean()
    r = {t: rel(p) for t, p in preds.items()}
    base = r["id"]
    legal = [t for t in preds if r[t] < base * 1.15 or t == "id"]
    ens = np.mean([preds[t] for t in legal], axis=0)
    print(f"[{PFX} {fam}] per-g:", {t: round(float(v), 3) for t, v in r.items()},
          f" legalK={len(legal)} single={base:.3f} ENS={rel(ens):.3f}", flush=True)
print("COMBINE_DONE", flush=True)
