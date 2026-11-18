"""Recompute BCAT / PROSE-FD shallow-water symmetry numbers from existing /eu dumps.

Dumps (pred, gt) are stored padded to C=4 with h in channel 0.
BUGGY = original inverse (sign-flips ch0/ch1 when C>=2, swaps ch0<->ch1 on transpose).
FIXED = h only (channel 0), treated as a scalar: flips / transpose / roll on space only.

Per replica: error vs GT and defect vs the identity prediction (mean and median over
sample x frame, relative L2 over space+channel, as in eps_base_final.py).
Ensembles: K4, K8 (D4), T3 (id, sh1, sh8) and K8xT3 (24 members, if *_s1/_s8 dumps exist).
For combined D4+shift dumps the composition order is not recorded, so both inverse orders
are evaluated and the one with lower GT error is taken (reported as 'order').
Memory-light: replicas are streamed, ensembles accumulated incrementally.
"""
import numpy as np, os, json, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/eu"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/ov/_lsprobe/swe_recompute.json"
FAM = "shallow_water"
GD = {"id": (0, 0, 0), "fr": (0, 1, 0), "fc": (0, 0, 1), "r180": (0, 1, 1),
      "d1": (1, 0, 0), "r90": (1, 1, 0), "r270": (1, 0, 1), "d2": (1, 1, 1)}
K4 = ("id", "fr", "fc", "r180")


def d4_inv_buggy(p, g):
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


def d4_inv_scalar(p, g):                 # same order as the original inverse, no channel ops
    t, fr, fc = g
    if fr: p = np.flip(p, axis=2)
    if fc: p = np.flip(p, axis=3)
    if t: p = np.swapaxes(p, 2, 3)
    return np.ascontiguousarray(p)


def roll_inv(p, s):                      # forward was np.roll(data, (-s,-s), axis=(H,W))
    return np.roll(p, (s, s), axis=(2, 3))


def per_sf(a, b):
    num = np.sqrt(((a - b) ** 2).sum(axis=(2, 3, 4)))
    den = np.sqrt((b ** 2).sum(axis=(2, 3, 4))) + 1e-12
    return num / den


def load_pred(path, h_only):
    with np.load(path) as z:
        p = z["pred"]
        return (p[..., :1] if h_only else p).astype(np.float64)


def stats(Pg, G, P0):
    e, d = per_sf(Pg, G), per_sf(Pg, P0)
    return {"err_mean": 100 * float(e.mean()), "defect_mean": 100 * float(d.mean()),
            "defect_median": 100 * float(np.median(d))}


res = {}
for model, pref in (("BCAT", "_bcatsymE"), ("PROSE-FD", "_prosesymE")):
    f = lambda tag: f"{ROOT}/{pref}_{tag}_{FAM}.npz"
    if not os.path.exists(f("id")):
        print(f"[{model}] missing identity dump", flush=True); continue
    with np.load(f("id")) as z:
        G4 = z["gt"].astype(np.float64); P04 = z["pred"].astype(np.float64)
    out = {"shape": list(P04.shape),
           "pred_chan_rms": [float(np.sqrt((P04[..., c] ** 2).mean())) for c in range(P04.shape[-1])],
           "gt_chan_rms": [float(np.sqrt((G4[..., c] ** 2).mean())) for c in range(G4.shape[-1])]}
    print(f"[{model}] shape {P04.shape} pred ch rms {out['pred_chan_rms']} gt ch rms {out['gt_chan_rms']}", flush=True)

    # ---- BUGGY reproduction (K4 + d1 only; all 4 channels) ----
    bug, acc = {}, np.zeros_like(P04)
    for tag in K4 + ("d1",):
        if not os.path.exists(f(tag)): continue
        Pg = P04 if tag == "id" else d4_inv_buggy(load_pred(f(tag), False), GD[tag])
        bug[tag] = stats(Pg, G4, P04)
        if tag in K4: acc += Pg
    bug_ens = {"K4": 100 * float(per_sf(acc / 4, G4).mean())}
    out["buggy"] = {"per_g": bug, "ens": bug_ens}
    print(f"  [buggy] " + str({t: (round(v['err_mean'], 3), round(v['defect_mean'], 2)) for t, v in bug.items()}) + f" {bug_ens}", flush=True)
    del G4, P04, acc

    # ---- FIXED (h only, scalar) ----
    with np.load(f("id")) as z:
        G = z["gt"][..., :1].astype(np.float64); P0 = z["pred"][..., :1].astype(np.float64)
    per, orders = {}, {}
    acc = {"K4": np.zeros_like(P0), "K8": np.zeros_like(P0), "T3": np.zeros_like(P0), "K8xT3": np.zeros_like(P0)}
    cnt = {k: 0 for k in acc}
    for tag, g in GD.items():
        for s in (0, 1, 8):
            name = tag if s == 0 else f"{tag}_s{s}"
            path = f(name)
            if tag == "id" and s == 0:
                Pg = P0
            elif tag == "id":
                # pure shifts: Aug-13 sh{s} dumps (forward roll(-s) documented in bcat_shift_runner2.sh)
                if not os.path.exists(f(f"sh{s}")): continue
                raw = load_pred(f(f"sh{s}"), True)
                cands = {f"roll{sg:+d}": roll_inv(raw, sg * s) for sg in (+1, -1)}
                errs = {k: float(per_sf(v, G).mean()) for k, v in cands.items()}
                best = min(errs, key=errs.get); Pg = cands[best]; del cands, raw
                name = f"sh{s}"
                orders[name] = best + " " + " ".join(f"{k}={100*v:.3f}" for k, v in errs.items())
            elif not os.path.exists(path):
                continue
            else:
                raw = load_pred(path, True)
                if s == 0:
                    Pg = d4_inv_scalar(raw, g)
                else:
                    # composition order and roll sign of the Aug-19 combined dumps are unrecorded:
                    # evaluate all four inverses, keep the one with the lowest GT error
                    cands = {}
                    for sg in (+1, -1):
                        cands[f"D4(roll{sg:+d})"] = roll_inv(d4_inv_scalar(raw, g), sg * s)
                        cands[f"roll{sg:+d}(D4)"] = d4_inv_scalar(roll_inv(raw, sg * s), g)
                    errs = {k: float(per_sf(v, G).mean()) for k, v in cands.items()}
                    best = min(errs, key=errs.get)
                    Pg = cands[best]; del cands
                    orders[name] = best + " " + " ".join(f"{k}={100*v:.3f}" for k, v in errs.items())
            per[name] = stats(Pg, G, P0)
            if s == 0 and tag in K4: acc["K4"] += Pg; cnt["K4"] += 1
            if s == 0: acc["K8"] += Pg; cnt["K8"] += 1
            if tag == "id": acc["T3"] += Pg; cnt["T3"] += 1
            acc["K8xT3"] += Pg; cnt["K8xT3"] += 1
            del Pg
    ens = {k: {"n": cnt[k], "err": 100 * float(per_sf(acc[k] / cnt[k], G).mean())} for k in acc if cnt[k]}
    base = per["id"]["err_mean"]
    for k in ens: ens[k]["gain_pct"] = 100 * (ens[k]["err"] - base) / base
    out["fixed"] = {"per_g": per, "ens": ens, "orders": orders}
    print(f"  [fixed] per-g (err, defect_mean, defect_median):", flush=True)
    for t, v in per.items():
        print(f"     {t:10s} {v['err_mean']:.4f}  {v['defect_mean']:.4f}  {v['defect_median']:.4f}", flush=True)
    print(f"  [fixed] ens: " + str({k: (v['n'], round(v['err'], 4), round(v['gain_pct'], 1)) for k, v in ens.items()}), flush=True)
    if orders: print(f"  [fixed] combined-order picks: {orders}", flush=True)
    res[model] = out
    del G, P0, acc

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print("SAVED", OUT, flush=True)
