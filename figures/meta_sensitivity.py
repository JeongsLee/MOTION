"""R1-2: sensitivity of MOTION-S to wrong or missing metadata (evaluation only; test split, 128 trajectories/family).
Paired per-trajectory comparison with the correct metadata (ev20-m1f-stats base). Writes a table (LaTeX) and JSON.
  python meta_sensitivity.py DATA_DIR      (DATA_DIR holds m1f/stats.npz and m1f_meta_<mode>/stats.npz)
"""
import json, sys, os
import numpy as np
from scipy.stats import wilcoxon

D = sys.argv[1]
MODES = [("all_on", "all operator tokens on"), ("none", "no operator token"), ("swap", "another family's tokens"),
         ("drop1", "one true token removed"), ("add1", "one spurious token added")]
SH = {"shallow_water": "SWE", "diff_react": "Diff-React", "com_ns": "Comp. NS", "incom_ns": "Incomp. NS",
      "pdearena_ns": "PDEArena", "pdearena_uncond": "PDEArena-U", "Poisson-Gauss": "Poisson",
      "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": "3D M0.1", "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": "3D M1.0",
      "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": "3D Turb"}
Z0 = np.load(f"{D}/m1f/stats.npz")


def err(Z, f):
    e = Z[f"base|{f}|phys"][:, 0]; w = Z[f"base|{f}|w"][:, 0]
    return 100 * (e * w).sum(1) / w.sum(1)


FAMS = sorted({k.split("|")[1] for k in Z0.files if k.startswith("base|")})
out = {"families": FAMS, "modes": {}}
for m, lab in MODES:
    p = f"{D}/m1f_meta_{m}/stats.npz"
    if not os.path.exists(p): continue
    Z = np.load(p); rows = {}
    for f in FAMS:
        if f"base|{f}|phys" not in Z.files: continue
        a, b = err(Z0, f), err(Z, f); n = min(len(a), len(b)); a, b = a[:n], b[:n]
        pv = wilcoxon(b, a).pvalue if np.any(a != b) else 1.0
        rows[f] = {"true": float(a.mean()), "meta": float(b.mean()), "rel": float(100 * (b.mean() / a.mean() - 1)), "p": float(pv)}
    nf = len(rows); holm = sorted(rows, key=lambda f: rows[f]["p"])
    for k, f in enumerate(holm):
        rows[f]["sig"] = rows[f]["p"] < 0.05 / (nf - k)                    # Holm step-down (approximate, sequential)
    rel = np.array([r["rel"] for r in rows.values()])
    out["modes"][m] = {"label": lab, "rows": rows, "median_rel": float(np.median(rel)), "max_rel": float(rel.max()),
                       "n_worse": int(sum(r["sig"] and r["rel"] > 0 for r in rows.values())),
                       "n_better": int(sum(r["sig"] and r["rel"] < 0 for r in rows.values())),
                       "class_avg_true": float(np.mean([r["true"] for r in rows.values()])),
                       "class_avg_meta": float(np.mean([r["meta"] for r in rows.values()]))}
json.dump(out, open(f"{D}/meta_sensitivity.json", "w"), indent=1)
for m, v in out["modes"].items():
    print(f"{m:7s} class avg {v['class_avg_true']:.2f} -> {v['class_avg_meta']:.2f} | median {v['median_rel']:+.1f}% max {v['max_rel']:+.1f}% "
          f"| worse {v['n_worse']} better {v['n_better']} of {len(v['rows'])}")
    top = sorted(v["rows"].items(), key=lambda x: -x[1]["rel"])[:4]
    print("        largest: " + ", ".join(f"{SH.get(f, f)} {r['rel']:+.0f}%" for f, r in top))
