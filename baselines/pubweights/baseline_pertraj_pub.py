"""REVISION-R1 (10-22; R1-5 / R2 / R3-3 paired statistics): PER-TRAJECTORY equivariance metrics of BCAT / PROSE-FD on the
six joint-benchmark families (test-600, 100 per family) from the existing /eu symmetry dumps, for paired tests against
MOTION-NS on the same trajectories. Conventions are those of the published (corrected) Fig. 4 numbers:
  * shallow_water: h only (channel 0), scalar flips / transpose / roll (swe_recompute.py, FIXED);
  * other families: replica_all.py inverse (row mirror negates ch0, column mirror ch1, transpose swaps ch0 <-> ch1),
    error over the real channels (GT rms > 0); CFDBench adds the scalar row mirror on the 2 flow channels (frns);
  * pure shifts sh1 / sh8: both roll signs, the one with the lower GT error (mean over trajectories) kept;
  * combined D4 + shift dumps (*_s1 / *_s8): both composition orders x both signs, lowest mean GT error kept.
Per trajectory (mean over frames, %): err and defect for every available member; ensembles K8 (D4), K8xT3 (24) and K2
(identity + the admitted mirror: CFDBench frns, PDEArena fr) where all members exist.
  python baseline_pertraj.py /eu OUT.json"""
import json
import os
import sys

import numpy as np

ROOT, OUT, KIND = sys.argv[1], sys.argv[2], sys.argv[3]   # REVISION-R1 11-06: released weights (_{bcat,prose}pubE dumps)
GD = {"id": (0, 0, 0), "fr": (0, 1, 0), "fc": (0, 0, 1), "r180": (0, 1, 1),
      "d1": (1, 0, 0), "r90": (1, 1, 0), "r270": (1, 0, 1), "d2": (1, 1, 1)}
FAMS = ["shallow_water", "com_ns", "incom_ns", "incom_ns_arena", "incom_ns_arena_u", "cfdbench"]


def inv_vec(p, g):
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


def inv_scal(p, g):
    t, fr, fc = g
    if fr: p = np.flip(p, axis=2)
    if fc: p = np.flip(p, axis=3)
    if t: p = np.swapaxes(p, 2, 3)
    return np.ascontiguousarray(p)


def roll(p, s):
    return np.roll(p, (s, s), axis=(2, 3))


def rel(a, b):                                   # (n, frames)
    return np.sqrt(((a - b) ** 2).sum(axis=(2, 3, 4), dtype=np.float64)) / (
        np.sqrt((b ** 2).sum(axis=(2, 3, 4), dtype=np.float64)) + 1e-12)


res = {}
for model, pref in ((("BCAT (released)", "_bcatpubE"),) if KIND == "bcat" else (("PROSE-FD (released)", "_prosepubE"),)):
    for fam in FAMS:
        f = lambda tag: f"{ROOT}/{pref}_{tag}_{fam}.npz"
        if not os.path.exists(f("id")):
            print(f"[{model} {fam}] no identity dump", flush=True); continue
        swe = fam == "shallow_water"
        with np.load(f("id")) as z:
            G = z["gt"].astype(np.float32); P0 = z["pred"].astype(np.float32)
        ch = [0] if swe else [c for c in range(G.shape[-1]) if np.sqrt((G[..., c] ** 2).mean()) > 0]
        G, P0 = G[..., ch], P0[..., ch]
        inv = inv_scal if swe else (lambda p, g: inv_vec(p, g))

        def load(tag):
            with np.load(f(tag)) as z:
                p = z["pred"].astype(np.float32)
            return p[..., ch] if swe else p

        def pick(cands):                          # lowest mean GT error among candidate inverses
            e = {k: float(rel(v[..., :] if swe else v[..., ch], G).mean()) for k, v in cands.items()}
            b = min(e, key=e.get)
            return (cands[b] if swe else cands[b][..., ch]), b
        row = {"n": int(G.shape[0]), "channels": ch, "err": {}, "defect": {}, "choice": {}}
        acc = {"K8": 0.0, "K8xT3": 0.0, "K2": 0.0}; cnt = {k: 0 for k in acc}
        for tag, g in GD.items():
            for s in (0, 1, 8):
                name = tag if s == 0 else f"{tag}_s{s}"
                if tag == "id" and s == 0:
                    Pg = P0
                elif tag == "id":
                    if not os.path.exists(f(f"sh{s}")): continue
                    raw = load(f"sh{s}")
                    Pg, row["choice"][name] = pick({f"roll{sg:+d}": roll(raw, sg * s) for sg in (1, -1)}); del raw
                elif not os.path.exists(f(name)):
                    continue
                else:
                    raw = load(name)
                    if s == 0:
                        Pg = inv(raw, g) if swe else inv(raw, g)[..., ch]
                    else:
                        c = {}
                        for sg in (1, -1):
                            c[f"D4(roll{sg:+d})"] = roll(inv(raw, g), sg * s)
                            c[f"roll{sg:+d}(D4)"] = inv(roll(raw, sg * s), g)
                        Pg, row["choice"][name] = pick(c)
                    del raw
                row["err"][name] = (100 * rel(Pg, G).mean(1)).tolist()
                row["defect"][name] = (100 * rel(Pg, P0).mean(1)).tolist()
                if s == 0: acc["K8"] = acc["K8"] + Pg; cnt["K8"] += 1
                acc["K8xT3"] = acc["K8xT3"] + Pg; cnt["K8xT3"] += 1
                if (tag == "id" and s == 0) or (tag == "fr" and s == 0 and fam.startswith("incom_ns_arena")):
                    acc["K2"] = acc["K2"] + Pg; cnt["K2"] += 1
                del Pg
        if fam == "cfdbench" and os.path.exists(f("frns")):   # scalar row mirror, 2 flow channels
            c2 = [i for i, c in enumerate(ch) if c < 2]
            with np.load(f("frns")) as z:
                A = np.flip(z["pred"].astype(np.float32), axis=2)[..., [ch[i] for i in c2]]
            G2, P02 = G[..., c2], P0[..., c2]
            row["frns"] = dict(err=(100 * rel(A, G2).mean(1)).tolist(), defect=(100 * rel(A, P02).mean(1)).tolist(),
                               id_err_2ch=(100 * rel(P02, G2).mean(1)).tolist(),
                               K2=(100 * rel(0.5 * (A + P02), G2).mean(1)).tolist())
            del A
        row["ens"] = {k: (100 * rel(acc[k] / cnt[k], G).mean(1)).tolist()
                      for k, need in (("K8", 8), ("K8xT3", 24), ("K2", 2)) if cnt[k] == need}
        res[f"{model}|{fam}"] = row
        e0 = np.mean(row["err"]["id"])
        print(f"[{model} {fam}] n {row['n']} ch {ch} id {e0:.3f} members {len(row['err'])} ens " +
              " ".join(f"{k} {np.mean(v):.3f} ({100*(np.mean(v)-e0)/e0:+.1f}%)" for k, v in row["ens"].items()) +
              (f" | frns err {np.mean(row['frns']['err']):.3f} def {np.mean(row['frns']['defect']):.3f} "
               f"K2 {np.mean(row['frns']['K2']):.3f} vs id {np.mean(row['frns']['id_err_2ch']):.3f}" if "frns" in row else ""),
              flush=True)
        del G, P0, acc
        json.dump(res, open(OUT, "w"))
json.dump(res, open(OUT, "w")); print("BASELINE_PERTRAJ_OK", flush=True)
