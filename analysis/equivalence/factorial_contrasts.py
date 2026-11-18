"""REVISION-R1 11-03 (user): equivariance of all matched-budget variants as factor contrasts. Axial Transformer runs
BR (MOTION-S design), V1a (generic bank), V3 (decoder instead of closed-form integration), V2 (no metadata), V1aV3,
V1aN, V1aV3N; FNO runs FNO_BR, FNO_V1a, FNO_V1aV3. Same per-trajectory dihedral defect, family counts, geometric-mean
ratio and two-sided family-level Wilcoxon as backbone_fixed_bank.py."""
import os
import json, numpy as np
from scipy import stats
D = os.environ.get("MOTION_RESULTS", "results") + "/A14_factorial/data"
D4 = ["fr", "fc", "r180", "d1", "r90", "r270", "d2"]
RUNS = ["BR", "V1a", "V3", "V2", "V1aV3", "V1aN", "V1aV3N", "FNO_BR", "FNO_V1a", "FNO_V1aV3"]
Z = {r: np.load(f"{D}/{r}/equiv.npz", allow_pickle=True) for r in RUNS}
FAMS = list(json.loads(str(Z["BR"]["__meta__"]))["summary"])
def per_traj(z, f):
    w = z[f"{f}|w"]; tm = lambda a: 100 * (a * w).sum(1) / w.sum(1)
    return np.mean([tm(z[f"{f}|{g}|defect"]) for g in D4], 0), tm(z[f"{f}|id|err"])
def comp(a, b):
    L = S = 0; md = []; me = []
    for f in FAMS:
        da, ea = per_traj(Z[a], f); db, eb = per_traj(Z[b], f)
        lr = np.log(db / da); p = stats.wilcoxon(db, da).pvalue
        L += p < .05 and np.median(lr) > 0; S += p < .05 and np.median(lr) < 0
        md.append(np.median(lr)); me.append(np.median(np.log(eb / ea)))
    md = np.array(md)
    return dict(larger=int(L), smaller=int(S), unchanged=len(FAMS) - int(L) - int(S), defect_ratio=float(np.exp(md.mean())),
                p=float(stats.wilcoxon(md).pvalue), p_greater=float(stats.wilcoxon(md, alternative="greater").pvalue),
                error_ratio=float(np.exp(np.mean(me))), md=md.tolist())
C = {"vs MOTION-S: w/o mechanism": ("BR", "V1a"), "vs MOTION-S: w/o integration": ("BR", "V3"),
     "vs MOTION-S: w/o metadata": ("BR", "V2"), "vs MOTION-S: w/o mechanism+integration": ("BR", "V1aV3"),
     "vs MOTION-S: w/o mechanism+metadata": ("BR", "V1aN"), "vs MOTION-S: w/o all": ("BR", "V1aV3N"),
     "bank | closed (Tr)": ("BR", "V1a"), "bank | decoder (Tr)": ("V3", "V1aV3"),
     "integration | prescribed (Tr)": ("BR", "V3"), "integration | generic (Tr)": ("V1a", "V1aV3"),
     "metadata | prescribed+closed": ("BR", "V2"), "metadata | generic+closed": ("V1a", "V1aN"),
     "metadata | generic+decoder": ("V1aV3", "V1aV3N"),
     "bank | closed (FNO)": ("FNO_BR", "FNO_V1a"), "integration | generic (FNO)": ("FNO_V1a", "FNO_V1aV3"),
     "FNO: w/o mechanism+integration": ("FNO_BR", "FNO_V1aV3")}
out = {k: comp(*v) for k, v in C.items()}
inter = np.array(out["bank | closed (Tr)"]["md"]) - np.array(out["bank | decoder (Tr)"]["md"])
out["interaction bank x integration (Tr)"] = dict(ratio=float(np.exp(inter.mean())), p=float(stats.wilcoxon(inter).pvalue))
for k, v in out.items():
    if "larger" in v:
        print(f"{k:42s} {v['larger']}/{v['smaller']}/{v['unchanged']}  defect x{v['defect_ratio']:.3f}  p {v['p']:.3f} (>: {v['p_greater']:.3f})  err x{v['error_ratio']:.3f}")
    else: print(k, v)
json.dump(out, open("factorial_contrasts.json", "w"), indent=1)
