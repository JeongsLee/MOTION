"""Direct equivariance defect from EXISTING per-g dumps (no inference).

  eps_equiv(T) = || T^{-1} G(T u) - G(u) || / || G(u) ||      (T norm-preserving)

G(T u) is exactly what each per-g run saved (prediction on the transformed input,
in the transformed frame); G(u) is the identity run. Ground truth is NOT used.
Files: /eu/_{bcat,prose}symE_{tag}_{family}.npz  with keys pred, gt.
"""
import numpy as np, os, json
GD = {"id": (0,0,0), "fr": (0,1,0), "fc": (0,0,1), "r180": (0,1,1),
      "d1": (1,0,0), "r90": (1,1,0), "r270": (1,0,1), "d2": (1,1,1)}

def inv(p, g):
    """map a prediction made in the T-frame back to the reference frame."""
    t, fr, fc = g; C = p.shape[-1]
    if fr:
        p = np.flip(p, axis=2).copy()
        if C >= 2: p[..., 0] = -p[..., 0]
    if fc:
        p = np.flip(p, axis=3).copy()
        if C >= 2: p[..., 1] = -p[..., 1]
    if t:
        p = np.swapaxes(p, 2, 3).copy()
        if C >= 2:
            sw = list(range(C)); sw[0], sw[1] = 1, 0
            p = p[..., sw]
    return p

FAMS = ["shallow_water", "com_ns", "incom_ns", "incom_ns_arena", "cfdbench", "incom_ns_arena_u"]
out = {}
for model, pref in (("BCAT", "_bcatsymE"), ("PROSE-FD", "_prosesymE")):
    for fam in FAMS:
        f0 = f"/eu/{pref}_id_{fam}.npz"
        if not os.path.exists(f0): continue
        P0 = np.load(f0)["pred"].astype(np.float64)
        den = np.sqrt((P0 ** 2).sum(axis=(2, 3, 4)))           # per (sample, frame)
        row = {}
        for tag, g in GD.items():
            if tag == "id": continue
            f = f"/eu/{pref}_{tag}_{fam}.npz"
            if not os.path.exists(f): continue
            Pg = inv(np.load(f)["pred"].astype(np.float64), g)
            num = np.sqrt(((Pg - P0) ** 2).sum(axis=(2, 3, 4)))
            row[tag] = float(np.median((num / (den + 1e-12)).ravel()) * 100)
        out[f"{model}|{fam}"] = row
        print(f"[{model} {fam}] eps_equiv (%):",
              {k: round(v, 2) for k, v in sorted(row.items(), key=lambda kv: kv[1])}, flush=True)
json.dump(out, open("/seoul/fig3_assets/equiv_defect_baselines.json", "w"), indent=1)
print("SAVED", flush=True)
