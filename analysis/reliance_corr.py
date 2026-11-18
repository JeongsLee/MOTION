"""Offline: does head reliance track the physical regime within a family? (revision R1 #1)

Inputs: an eval_stats.py npz with configs base + ko:<head> (per-trajectory errors, fixed seeds) and a
reliance_diag.py npz (per-trajectory regime diagnostics, same units and order).
Reliance of trajectory i on head k:  r_ik = (e_ko:k,i - e_base,i) / e_base,i   (relative KO damage).
For every (family, head, diagnostic): Spearman rho with a permutation p-value.

PRE-REGISTERED pairs (stated before looking at results; Holm-corrected as one family of tests):
  advection, shock, compressible  ~ mach        (+)   compressible families
  shock                           ~ shock       (+)   compressible families
  diffusion                       ~ ke_decay    (+)   velocity families (dissipation)
  diffusion                       ~ grad_energy (+)   all
  reaction_nl                     ~ amplitude   (+)   diff_react, ACE
All other pairs are reported as exploratory (uncorrected).
  python reliance_corr.py STATS.npz DIAG.npz [--metric phys] [--json OUT.json] [--perm 5000]
"""
import argparse, json
import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("stats"); ap.add_argument("diag")
ap.add_argument("--metric", default="phys")
ap.add_argument("--perm", type=int, default=5000)
ap.add_argument("--json", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)
S = np.load(a.stats); D = np.load(a.diag)


def traj(cfg, fam):
    e, w = S[f"{cfg}|{fam}|{a.metric}"], S[f"{cfg}|{fam}|w"]
    return ((e * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)).mean(1)


PRE = [("advection", "mach"), ("shock", "mach"), ("compressible", "mach"), ("shock", "shock"),
       ("diffusion", "ke_decay"), ("diffusion", "grad_energy"), ("reaction_nl", "amplitude")]
fams = sorted({k.split("|")[1] for k in S.files if k.startswith("base|")})
heads = sorted({k.split("|")[0][3:] for k in S.files if k.startswith("ko:")})
diags = sorted({k.split("|")[1] for k in D.files if "|" in k})
rows = []
for fam in fams:
    eb = traj("base", fam)
    for h in heads:
        if f"ko:{h}|{fam}|{a.metric}" not in S.files:
            continue
        r = (traj(f"ko:{h}", fam) - eb) / np.maximum(eb, 1e-12)
        if np.nanmax(np.abs(r)) < 1e-3:
            continue                                  # head closed (metadata mask) / no effect in this family
        for dg in diags:
            key = f"{fam}|{dg}"
            if key not in D.files:
                continue
            x = D[key][: len(r)]
            ok = np.isfinite(x) & np.isfinite(r)
            if ok.sum() < 8 or np.nanstd(x[ok]) < 0.02 * (np.abs(np.nanmean(x[ok])) + 1e-12):
                continue                              # regime ~constant within the family (e.g. CE-KH/CE-RM Mach)
            rho = spearmanr(x[ok], r[ok]).correlation
            xp = x[ok]; rp = r[ok]
            null = np.array([spearmanr(rng.permutation(xp), rp).correlation for _ in range(a.perm)])
            p = float((np.abs(null) >= abs(rho)).mean())
            rows.append({"family": fam, "head": h, "diag": dg, "rho": float(rho), "p": p, "n": int(ok.sum()),
                         "mean_reliance": float(np.mean(r)), "pre": (h, dg) in PRE})
pre = [r for r in rows if r["pre"]]
if pre:                                               # Holm over the pre-registered tests
    o = np.argsort([r["p"] for r in pre]); m = len(pre); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * pre[i]["p"]); pre[i]["p_holm"] = min(run, 1.0)
print(f"PRE-REGISTERED ({len(pre)} tests, Holm):")
for r in sorted(pre, key=lambda r: (r["family"], r["head"])):
    print(f"  {r['family']:14s} {r['head']:13s} ~ {r['diag']:11s} rho {r['rho']:+.2f}  p {r['p']:.3f}  "
          f"holm {r.get('p_holm', 1):.3f}  n={r['n']}  mean KO damage {100*r['mean_reliance']:+.1f}%"
          + ("  *" if r.get("p_holm", 1) < 0.05 else ""))
print("EXPLORATORY |rho| >= 0.4, p < 0.01 (uncorrected):")
for r in sorted(rows, key=lambda r: -abs(r["rho"])):
    if not r["pre"] and abs(r["rho"]) >= 0.4 and r["p"] < 0.01:
        print(f"  {r['family']:14s} {r['head']:13s} ~ {r['diag']:11s} rho {r['rho']:+.2f}  p {r['p']:.3f}  n={r['n']}")
if a.json:
    json.dump(rows, open(a.json, "w"), indent=1)
