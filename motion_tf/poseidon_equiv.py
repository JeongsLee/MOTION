"""REVISION-R1 (10-22): equivariance diagnostics for Poseidon (scOT), same protocol as eval_equiv.py (Fig. 4b).
Units/IC/leads as poseidon_same.py (loader SPLIT, first NEV per family, IC = frame 0, direct lead k at time k/20).
For g in D4 (8) x periodic shifts (0, 1, 8 cells): P_g = g^-1 f(g x0); per frame, physical units, Poseidon channels
(CE: rho, u, v, p; NS: u, v):
  defect_g = ||P_g - P_id|| / ||P_id||,  err_g = ||P_g - y|| / ||y||   (relative L2 over space + channels)
Ensembles (mean of back-mapped replicas): K4 = {id, fr, fc, r180}, K8 = D4, T3 = shifts, K8xT3 = all 24.
Per family: mean over leads of the per-trajectory value, then mean over trajectories (eval_equiv convention);
rel-L1 of the identity is printed too (poseidon_same.py metric, median). Vector pairing (which velocity slot lies
along array axis H) is checked PER FAMILY on ground truth (CE: continuity residual; NS: divergence).
  python poseidon_equiv.py MODEL OUT.json"""
import json, os, sys, time
import numpy as np
import torch
from scOT.model import ScOT
from data import loader
MODEL, OUTP = sys.argv[1], sys.argv[2]
FAMS = os.environ.get("FAMS", "CE-RP,CE-CRP,CE-KH,CE-Gauss,NS-Sines,NS-Gauss").split(",")
NEV = int(os.environ.get("NEV", "32")); SPLIT = os.environ.get("SPLIT", "val"); KF = 20
MEAN = np.array([0.80, 0.0, 0.0, 0.0], np.float32); STD = np.array([0.31, 0.391, 0.356, 0.185], np.float32)
PMEAN = {"CE-Gauss": 2.513, "CE-KH": 1.0, "CE-RP": 0.215, "CE-CRP": 0.553}
GD = {"id": (0, 0, 0), "fr": (0, 1, 0), "fc": (0, 0, 1), "r180": (0, 1, 1),
      "d1": (1, 0, 0), "r90": (1, 1, 0), "r270": (1, 0, 1), "d2": (1, 1, 1)}
K4 = ("id", "fr", "fc", "r180"); SHIFTS = (0, 1, 8)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = ScOT.from_pretrained(MODEL).to(dev).eval()
print(f"[poseidon-equiv] {MODEL} on {dev}", flush=True)


def phys(fam, s):
    """(T, 4, H, W) physical Poseidon channels [rho, u, v, p] (p WITH its mean)."""
    f = np.asarray(s["fields"], np.float32); dims = tuple(int(v) for v in s["dims"])
    f = f.reshape(f.shape[0], *dims, f.shape[-1])
    if fam.startswith("CE"):
        return np.stack([f[..., 3], f[..., 0], f[..., 1], f[..., 4]], 1)
    return np.stack([np.ones_like(f[..., 0]), f[..., 0], f[..., 1], np.zeros_like(f[..., 0])], 1)


# ---- vector pairing PER FAMILY (eval_equiv.family_pairing lesson: CE-RP and NS-Sines store different layouts):
# velocity channel 1 (our slot 0) along array axis H [VA1 = 1] or channel 2 [VA1 = 2]. CE: continuity residual
# dt(rho) + div(rho u) with the dt scale fitted by least squares; NS: div u.
def family_pairing(fam):
    d = lambda a, ax: np.roll(a, -1, ax) - np.roll(a, 1, ax)
    X = np.stack([phys(fam, s) for n, s in zip(range(4), loader.units_for(fam, SPLIT))]).astype(np.float64)
    out = {}
    for va in (1, 2):
        uh, uw = (X[:, :, 1], X[:, :, 2]) if va == 1 else (X[:, :, 2], X[:, :, 1])
        if fam.startswith("CE"):
            K = min(6, X.shape[1] - 1)                    # consecutive frame pairs 0..K (pooled)
            rb = 0.5 * (X[:, :K, 0] + X[:, 1:K + 1, 0]); uhb = 0.5 * (uh[:, :K] + uh[:, 1:K + 1]); uwb = 0.5 * (uw[:, :K] + uw[:, 1:K + 1])
            D = d(rb * uhb, -2) + d(rb * uwb, -1); dr = X[:, 1:K + 1, 0] - X[:, :K, 0]
            al = -(dr * D).sum() / ((D * D).sum() + 1e-30)
            out[va] = float(np.sqrt(((dr + al * D) ** 2).sum() / ((dr ** 2).sum() + 1e-30)))
        else:
            nrm = np.sqrt((d(uh, -2) ** 2).mean() + (d(uw, -1) ** 2).mean()) + 1e-30
            out[va] = float(np.sqrt(((d(uh, -2) + d(uw, -1)) ** 2).mean()) / nrm)
    va = 1 if out[1] < out[2] else 2
    print(f"[pairing] {fam}: {'continuity' if fam.startswith('CE') else 'div'} residual ch1~H {out[1]:.4f} | ch2~H {out[2]:.4f}"
          f" -> VA1 = ch{va}", flush=True)
    if not min(out.values()) < 0.7 * max(out.values()):
        # POSEIDON_VA1="FAM:ch,..." from a decisive pairing signature (single-axis reflections {fr, fc, r90, r270}
        # corrupted while {r180, d1, d2} intact); CE-RP 10-22 run 1: ch1 gave 58.6% vs 4.2% -> ch2.
        ov = dict(kv.split(":") for kv in os.environ.get("POSEIDON_VA1", "").split(",") if ":" in kv)
        assert fam in ov, f"vector pairing ambiguous for {fam}"
        va = int(ov[fam]); out["override"] = va
        print(f"[pairing] {fam}: ambiguous -> POSEIDON_VA1 override ch{va}", flush=True)
    return va, out


VA1 = None


def _flip(x, ax):                     # x (..., 4, H, W); ax = -2 (H) or -1 (W)
    x = np.flip(x, axis=ax).copy()
    c = VA1 if ax == -2 else 3 - VA1
    x[..., c, :, :] = -x[..., c, :, :]
    return x


def _transpose(x):
    x = np.swapaxes(x, -2, -1).copy()
    x[..., [1, 2], :, :] = x[..., [2, 1], :, :]
    return x


def g_fwd(x, g, s):
    t, fr, fc = GD[g]
    if s: x = np.roll(x, (s, s), axis=(-2, -1))
    if fr: x = _flip(x, -2)
    if fc: x = _flip(x, -1)
    if t: x = _transpose(x)
    return x


def g_inv(x, g, s):
    t, fr, fc = GD[g]
    if t: x = _transpose(x)
    if fc: x = _flip(x, -1)
    if fr: x = _flip(x, -2)
    if s: x = np.roll(x, (-s, -s), axis=(-2, -1))
    return x


def predict(fam, x0, T):              # x0 physical (4, H, W) -> (T, 4, H, W) physical
    pm = PMEAN.get(fam, 0.0); x = x0.copy(); x[3] -= pm
    xn = (x - MEAN[:, None, None]) / STD[:, None, None]
    xt = torch.from_numpy(np.repeat(xn[None], T, 0)).to(dev)
    t = torch.arange(1, T + 1, dtype=torch.float32, device=dev) / 20.0
    with torch.no_grad():
        out = model(pixel_values=xt, time=t).output.cpu().numpy()
    P = out * STD[None, :, None, None] + MEAN[None, :, None, None]; P[:, 3] += pm
    return P


def rel2(a, b, ch):                   # per frame
    return np.sqrt(((a[:, ch] - b[:, ch]) ** 2).sum((1, 2, 3)) / ((b[:, ch] ** 2).sum((1, 2, 3)) + 1e-12))


MEMBERS = [(g, s) for s in SHIFTS for g in GD]
res = {}; t0 = time.time()
for fam in FAMS:
    VA1, pdiag = family_pairing(fam)
    ch = [0, 1, 2, 3] if fam.startswith("CE") else [1, 2]
    err = {}; dfc = {}; ens = {k: [] for k in ("K4", "K8", "T3", "K8xT3")}; l1 = []
    for n, s in enumerate(loader.units_for(fam, SPLIT)):
        if n >= NEV: break
        X = phys(fam, s); T = min(KF, X.shape[0] - 1); Y = X[1:T + 1]
        acc = {k: 0.0 for k in ens}; P0 = None
        for g, sh in MEMBERS:
            Pg = g_inv(predict(fam, g_fwd(X[0], g, sh), T), g, sh)
            if P0 is None: P0 = Pg
            name = g if sh == 0 else f"{g}_s{sh}"
            err.setdefault(name, []).append(float(rel2(Pg, Y, ch).mean()))
            dfc.setdefault(name, []).append(float(rel2(Pg, P0, ch).mean()))
            if sh == 0 and g in K4: acc["K4"] = acc["K4"] + Pg
            if sh == 0: acc["K8"] = acc["K8"] + Pg
            if g == "id": acc["T3"] = acc["T3"] + Pg
            acc["K8xT3"] = acc["K8xT3"] + Pg
        for k, c in (("K4", 4), ("K8", 8), ("T3", 3), ("K8xT3", 24)):
            ens[k].append(float(rel2(acc[k] / c, Y, ch).mean()))
        l1.append(float((np.abs(P0[:, ch] - Y[:, ch]).sum((1, 2, 3)) / (np.abs(Y[:, ch]).sum((1, 2, 3)) + 1e-8)).mean()))
    m = lambda v: 100 * float(np.mean(v))
    base = m(err["id"])
    d4 = [m(dfc[g]) for g in GD if g != "id"]; shd = [m(dfc[f"id_s{s}"]) for s in (1, 8)]
    E = {k: m(v) for k, v in ens.items()}
    res[fam] = dict(n=len(l1), VA1=VA1, pairing=pdiag, err_id_L2=base, l1_median=100 * float(np.median(l1)), d4_defect_mean=float(np.mean(d4)),
                    d4_defect_max=float(np.max(d4)), shift_defect=shd, ens=E,
                    gain={k: 100 * (v - base) / base for k, v in E.items()},
                    per_g={k: (m(err[k]), m(dfc[k])) for k in err},
                    # REVISION-R1 (10-22): per-trajectory values (mean over leads, %) for paired tests against MOTION
                    per_traj=dict(err={k: [100 * x for x in v] for k, v in err.items()},
                                  defect={k: [100 * x for x in v] for k, v in dfc.items()},
                                  ens={k: [100 * x for x in v] for k, v in ens.items()}))
    print(f"[{fam:9s}] n={len(l1)} err(L2) {base:.3f} (L1 med {res[fam]['l1_median']:.2f}) | D4 defect mean "
          f"{np.mean(d4):.3f} max {np.max(d4):.3f} | shift defect {['%.3f' % v for v in shd]} | "
          + " ".join(f"{k} {v:.3f} ({res[fam]['gain'][k]:+.1f}%)" for k, v in E.items()) + f"  ({time.time()-t0:.0f}s)",
          flush=True)
    sig = (np.mean([m(dfc[g]) for g in ("fr", "fc", "r90", "r270")]) / max(np.mean([m(dfc[g]) for g in ("r180", "d1", "d2")]), 1e-9))
    res[fam]["pairing_signature"] = float(sig)
    print(f"     pairing signature {sig:.2f}" + ("  [WARN: vector pairing suspect]" if sig > 2 else ""), flush=True)
    print("     per-g (err, defect): " + " ".join(f"{k}=({v[0]:.2f},{v[1]:.2f})" for k, v in res[fam]["per_g"].items()
                                               if "_s" not in k), flush=True)
json.dump(dict(model=MODEL, split=SPLIT, nev=NEV, res=res), open(OUTP, "w"), indent=1)
print("POSEIDON_EQUIV_DONE", flush=True)
