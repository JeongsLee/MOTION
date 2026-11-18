"""Offline statistics on eval_stats.py outputs (no model needed).

Per-trajectory error = frame-weighted mean of the per-frame rel-L2 (weights = valid-frame mask,
or (k+1) for steady relaxation), averaged over point-sampling draws. Then:
  * per-family mean with 95% bootstrap CI (resample trajectories)
  * class average with a hierarchical bootstrap CI (resample trajectories within each family)
  * paired comparison vs a reference run: per-family mean difference with paired-bootstrap CI,
    Wilcoxon signed-rank p-value, Holm-adjusted across families; class-average difference CI
  python stats_compare.py REF.npz [OTHER.npz ...] [--cfg base] [--metric phys] [--B 2000]
"""
import argparse, json, sys
import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--cfg", default="base")
ap.add_argument("--ref_cfg", default=None, help="config inside the REF file (default = --cfg)")
ap.add_argument("--metric", default="phys")
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--json", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)


def load(path, cfg):
    z = np.load(path, allow_pickle=False)
    fams = sorted({k.split("|")[1] for k in z.files if k.startswith(cfg + "|")})
    out = {}
    for f in fams:
        e, w = z[f"{cfg}|{f}|{a.metric}"], z[f"{cfg}|{f}|w"]            # (n, draws, k_seg)
        out[f] = ((e * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)).mean(1)  # (n,)
    return out


def boot_mean(x):
    i = rng.integers(0, len(x), (a.B, len(x)))
    return np.percentile(x[i].mean(1), [2.5, 97.5])


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


ref = load(a.files[0], a.ref_cfg or a.cfg)
report = {"ref": a.files[0], "metric": a.metric, "families": {}}
print(f"REF {a.files[0]} [{a.ref_cfg or a.cfg}]  metric={a.metric}")
fams = sorted(ref)
for f in fams:
    lo, hi = boot_mean(ref[f])
    print(f"  {f:44s} {100*ref[f].mean():7.3f}  [{100*lo:.3f}, {100*hi:.3f}]  n={len(ref[f])}")
# hierarchical class-average CI
def class_avg_boot(d):
    cs = []
    for _ in range(a.B):
        cs.append(np.mean([d[f][rng.integers(0, len(d[f]), len(d[f]))].mean() for f in fams]))
    return np.percentile(cs, [2.5, 97.5])
ca = np.mean([ref[f].mean() for f in fams]); clo, chi = class_avg_boot(ref)
print(f"  CLASS-AVG {100*ca:.3f}  [{100*clo:.3f}, {100*chi:.3f}]")
report["ref_class_avg"] = [ca, clo, chi]

for path in a.files[1:]:
    oth = load(path, a.cfg)
    common = [f for f in fams if f in oth and len(oth[f]) == len(ref[f])]
    print(f"\nVS {path} [{a.cfg}]  (paired on {len(common)} families; diff = other - ref, % points)")
    ps, rows = [], []
    for f in common:
        d = oth[f] - ref[f]
        lo, hi = boot_mean(d)
        try:
            p = wilcoxon(oth[f], ref[f]).pvalue
        except ValueError:
            p = 1.0
        ps.append(p); rows.append((f, d.mean(), lo, hi, p, oth[f].mean(), ref[f].mean()))
    adj = holm(ps)
    for (f, dm, lo, hi, p, mo, mr), pa in zip(rows, adj):
        flag = "*" if pa < 0.05 else " "
        print(f"  {f:44s} {100*mr:7.3f} -> {100*mo:7.3f}  diff {100*dm:+7.3f} [{100*lo:+.3f},{100*hi:+.3f}]  "
              f"p={p:.2e} holm={pa:.2e} {flag}")
    # class-average paired difference
    diffs = []
    for _ in range(a.B):
        tot = 0.0
        for f in common:
            i = rng.integers(0, len(ref[f]), len(ref[f]))
            tot += (oth[f][i] - ref[f][i]).mean()
        diffs.append(tot / len(common))
    dm = np.mean([(oth[f] - ref[f]).mean() for f in common])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  CLASS-AVG diff {100*dm:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}]")
    report["families"][path] = {"rows": [[r[0], r[1], r[2], r[3], r[4], pa] for r, pa in zip(rows, adj)],
                                "class_avg_diff": [dm, lo, hi]}
if a.json:
    json.dump(report, open(a.json, "w"), indent=1, default=float)
