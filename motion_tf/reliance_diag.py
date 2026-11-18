"""Per-trajectory PHYSICAL-REGIME diagnostics from ground truth (revision R1 #1: "head reliance tracks
physical parameters within families").

The model never receives most per-sample parameters (adapters.load encodes the spec without values,
so per_sample nu / attr:mach are masked out), so the regime is measured from the data itself, for
the SAME units and order eval_stats.py uses (loader.units_for(fam, split)[:n]). Physical units.

Diagnostics (computed where the slots exist; NaN otherwise):
  mach        rms |u| / c with c = sqrt(p / rho)   (gamma omitted: rank statistics are invariant)
  shock       mean over frames of max |grad rho| / mean rho (cell units)
  ke_decay    -(d/dt) log KE over the trajectory (fitted slope), velocity families
  grad_energy mean |grad u|^2 / mean u^2 over scored channels (effective squared wavenumber)
  amplitude   std of the scored channels at t=0 relative to their mean |.| (nonlinearity proxy)
  tchange     mean frame-to-frame relative change of the scored channels
  python reliance_diag.py --out DIAG.npz [--split test] [--n 64] [--fams ...]
"""
import argparse, json
import itertools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--fams", default="com_ns,incom_ns,pdearena_ns,CE-RP,CE-CRP,CE-KH,CE-Gauss,CE-RM,"
                                  "NS-Sines,NS-Gauss,shallow_water,diff_react,ACE")
a = ap.parse_args()
from data import loader

d = lambda u, ax: 0.5 * (np.roll(u, -1, ax) - np.roll(u, 1, ax))
out = {}
for fam in a.fams.split(","):
    units = list(itertools.islice(loader.units_for(fam, a.split), a.n))
    rows = {k: [] for k in ("mach", "shock", "ke_decay", "grad_energy", "amplitude", "tchange")}
    for s in units:
        cm = np.asarray(s["cmask"]).astype(bool)
        dims = tuple(int(v) for v in s["dims"])
        f = np.asarray(s["fields"], np.float64)
        f = f.reshape(f.shape[0], *dims, f.shape[-1])
        tm = s.get("tmask")                                  # valid frames (pdearena_uncond padding)
        if tm is not None and np.asarray(tm).ndim == 1 and np.asarray(tm).sum() > 2:
            f = f[: int(np.asarray(tm).sum())]
        vel = cm[0] and cm[1]
        ax = (1, 2)
        if vel and cm[3] and cm[4]:
            rho = np.maximum(f[..., 3], 1e-8); p = np.maximum(f[..., 4], 1e-8)
            sp = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
            rows["mach"].append(float(np.sqrt((sp ** 2).mean()) / np.sqrt((p / rho).mean())))
        else:
            rows["mach"].append(np.nan)
        if cm[3]:
            rho = f[..., 3]
            g = np.sqrt(d(rho, 1) ** 2 + d(rho, 2) ** 2)
            rows["shock"].append(float(np.mean(g.max(axis=(1, 2)) / (np.abs(rho).mean(axis=(1, 2)) + 1e-12))))
        else:
            rows["shock"].append(np.nan)
        if vel:
            ke = 0.5 * (f[..., 0] ** 2 + f[..., 1] ** 2).mean(axis=(1, 2)) + 1e-12
            t = np.arange(len(ke))
            rows["ke_decay"].append(float(-np.polyfit(t, np.log(ke), 1)[0]) if len(ke) > 2 else np.nan)
        else:
            rows["ke_decay"].append(np.nan)
        x = f[..., cm]
        ge = sum((d(x, k) ** 2).mean() for k in ax) / ((x ** 2).mean() + 1e-12)
        rows["grad_energy"].append(float(ge))
        x0 = x[0]
        rows["amplitude"].append(float(x0.std() / (np.abs(x0).mean() + 1e-12)))
        dx = np.sqrt(((x[1:] - x[:-1]) ** 2).sum(axis=(1, 2, 3))) / (np.sqrt((x[:-1] ** 2).sum(axis=(1, 2, 3))) + 1e-12)
        rows["tchange"].append(float(dx.mean()))
    for k, v in rows.items():
        out[f"{fam}|{k}"] = np.asarray(v, np.float64)
    print(f"[{fam:14s}] n={len(units)} " + " ".join(
        f"{k}={np.nanmedian(v):.3g}[{np.nanmin(v):.2g},{np.nanmax(v):.2g}]" for k, v in rows.items()
        if not np.all(np.isnan(v))), flush=True)
np.savez_compressed(a.out, **out, __meta__=json.dumps({"split": a.split, "n": a.n}))
print("DIAG_OK", a.out, flush=True)
