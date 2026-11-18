import numpy as np, os, json
GD = {"fr": (0,1,0), "fc": (0,0,1), "r180": (0,1,1), "d1": (1,0,0), "r90": (1,1,0), "r270": (1,0,1), "d2": (1,1,1)}
def inv(p, g):
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
            sw = list(range(C)); sw[0], sw[1] = 1, 0; p = p[..., sw]
    return p
FAMS = ["shallow_water", "com_ns", "incom_ns", "incom_ns_arena", "cfdbench", "incom_ns_arena_u"]
res = {}
for model, pref in (("BCAT", "_bcatsymE"), ("PROSE-FD", "_prosesymE")):
    for fam in FAMS:
        f0 = f"/eu/{pref}_id_{fam}.npz"
        if not os.path.exists(f0): continue
        P0 = np.load(f0)["pred"].astype(np.float64)
        den = np.sqrt((P0 ** 2).sum(axis=(2, 3, 4))) + 1e-12
        row = {}
        for tag, g in GD.items():
            f = f"/eu/{pref}_{tag}_{fam}.npz"
            if not os.path.exists(f): continue
            Pg = inv(np.load(f)["pred"].astype(np.float64), g)
            row[tag] = float((np.sqrt(((Pg - P0) ** 2).sum(axis=(2, 3, 4))) / den).mean() * 100)
        # cfd: scalar-channel row mirror (no sign flip), 2 real channels
        if fam == "cfdbench" and os.path.exists(f"/eu/{pref}_frns_cfdbench.npz"):
            A = np.load(f"/eu/{pref}_frns_cfdbench.npz")["pred"].astype(np.float64)[..., :2]
            B = P0[..., :2]; A = np.flip(A, axis=2).copy()
            d2 = np.sqrt((B ** 2).sum(axis=(2, 3, 4))) + 1e-12
            row["frns(scalar)"] = float((np.sqrt(((A - B) ** 2).sum(axis=(2, 3, 4))) / d2).mean() * 100)
        res[f"{model}|{fam}"] = row
        print(f"[{model} {fam}]", {k: round(v, 3) for k, v in sorted(row.items(), key=lambda kv: kv[1])}, flush=True)
json.dump(res, open("/seoul/fig3_assets/equiv_defect_baselines_final.json", "w"), indent=1)
print("SAVED", flush=True)
