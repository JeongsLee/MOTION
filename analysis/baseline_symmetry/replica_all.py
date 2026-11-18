"""REVISION-R1 (R1 minor 2; user 10-21): per-element replica errors of BCAT / PROSE-FD on the five non-SWE families of the
joint benchmark, from the existing /eu symmetry dumps, so that Supplementary Fig. 4 (replica accuracy ratio) covers all six
families like Fig. 4b. Inverse = the convention of eps_base_final.py (the source of the corrected Fig. 4b defects): flip /
transpose on space with the sign of ch0 (row mirror) / ch1 (column mirror) and the ch0<->ch1 swap on transpose. CFDBench also
gets the scalar-channel row mirror (frns dump, 2 real channels) as in eps_base_final.py. Pure translations: sh1 / sh8 dumps,
both roll signs evaluated, the one with the lower GT error kept (as swe_recompute v3).
Per element: err (rel-L2 vs GT over the REAL channels = GT rms > 0), defect (vs identity prediction, same channels), mean over
sample x frame, in %. SWE is taken from swe_recompute_v3.json (fixed, h only).
  python replica_all.py /eu OUT.json"""
import numpy as np, os, json, sys
ROOT, OUT = sys.argv[1], sys.argv[2]
GD = {"fr": (0, 1, 0), "fc": (0, 0, 1), "r180": (0, 1, 1), "d1": (1, 0, 0), "r90": (1, 1, 0), "r270": (1, 0, 1), "d2": (1, 1, 1)}
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
    return np.ascontiguousarray(p)
def per_sf(a, b, ch):
    a, b = a[..., ch], b[..., ch]
    num = np.sqrt(np.square(a - b).sum(axis=(2, 3, 4), dtype=np.float64)); den = np.sqrt(np.square(b).sum(axis=(2, 3, 4), dtype=np.float64))
    return num / (den + 1e-12)
FAMS = ["com_ns", "incom_ns", "incom_ns_arena", "incom_ns_arena_u", "cfdbench"]
res = {}
for model, pref in (("BCAT", "_bcatsymE"), ("PROSE-FD", "_prosesymE")):
    for fam in FAMS:
        f = lambda tag: f"{ROOT}/{pref}_{tag}_{fam}.npz"
        if not os.path.exists(f("id")): print(f"[{model} {fam}] no identity dump", flush=True); continue
        with np.load(f("id")) as z:
            G = z["gt"].astype(np.float32); P0 = z["pred"].astype(np.float32)
        ch = [c for c in range(G.shape[-1]) if np.sqrt((G[..., c] ** 2).mean()) > 0]
        row = {"shape": list(P0.shape), "real_channels": ch, "id": {"err": 100 * float(per_sf(P0, G, ch).mean())}}
        for tag, g in GD.items():
            if not os.path.exists(f(tag)): continue
            with np.load(f(tag)) as z: Pg = inv(z["pred"].astype(np.float32), g)
            row[tag] = {"err": 100 * float(per_sf(Pg, G, ch).mean()), "defect": 100 * float(per_sf(Pg, P0, ch).mean())}
            del Pg
        if fam == "cfdbench" and os.path.exists(f("frns")):
            with np.load(f("frns")) as z: A = np.flip(z["pred"].astype(np.float32), axis=2).copy()
            c2 = [c for c in ch if c < 2]
            row["frns(scalar)"] = {"err": 100 * float(per_sf(A, G, c2).mean()), "defect": 100 * float(per_sf(A, P0, c2).mean()),
                                   "id_err_2ch": 100 * float(per_sf(P0, G, c2).mean())}
            del A
        for s in (1, 8):
            if not os.path.exists(f(f"sh{s}")): continue
            with np.load(f(f"sh{s}")) as z: raw = z["pred"].astype(np.float32)
            best = None
            for sg in (+1, -1):
                Pg = np.roll(raw, (sg * s, sg * s), axis=(2, 3)); e = float(per_sf(Pg, G, ch).mean())
                if best is None or e < best[0]: best = (e, sg, float(per_sf(Pg, P0, ch).mean()))
            row[f"sh{s}"] = {"err": 100 * best[0], "defect": 100 * best[2], "roll_sign": best[1]}
            del raw
        res[f"{model}|{fam}"] = row
        print(f"[{model} {fam}] ch {ch} " + " ".join(f"{k}:{v['err']:.3f}/{v.get('defect', 0):.2f}" for k, v in row.items() if isinstance(v, dict)), flush=True)
        del G, P0
        json.dump(res, open(OUT, "w"), indent=1)
json.dump(res, open(OUT, "w"), indent=1); print("REPLICA_ALL_OK", flush=True)
