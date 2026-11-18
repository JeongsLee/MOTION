"""REVISION-R1 11-02 (user): does the prescribed bank lower the dihedral equivariance defect independently of the
backbone? 2 x 2 design from the matched-budget runs: backbone {axial Transformer, FNO} x bank {prescribed, generic}
(BR, V1a, FNO_BR, FNO_V1a; 20,000 steps, test trajectories, 12 periodic square-domain families).
Per trajectory: dihedral defect = mean over the 7 non-identity D4 members of the frame-weighted defect (%), error =
untransformed error. Per family: paired Wilcoxon of the generic against the prescribed bank (count larger/smaller at
p<0.05) and the median log ratio. Pooled: sign test of the family directions within and across backbones; family-
level mixed model log(defect) ~ bank * backbone (+ log error) with family random intercepts (statsmodels if present,
else a paired family-level t / Wilcoxon on log ratios)."""
import os
import json, sys
import numpy as np
from scipy import stats
D = os.environ.get("MOTION_RESULTS", "results") + "/A14_factorial/data"
RUNS = {("Transformer", "prescribed"): "BR", ("Transformer", "generic"): "V1a",
        ("FNO", "prescribed"): "FNO_BR", ("FNO", "generic"): "FNO_V1a"}
D4 = ["fr", "fc", "r180", "d1", "r90", "r270", "d2"]
Z = {k: np.load(f"{D}/{v}/equiv.npz", allow_pickle=True) for k, v in RUNS.items()}
FAMS = list(json.loads(str(Z[("Transformer", "prescribed")]["__meta__"]))["summary"])
def per_traj(z, f):
    w = z[f"{f}|w"]; tm = lambda a: 100 * (a * w).sum(1) / w.sum(1)
    return np.mean([tm(z[f"{f}|{g}|defect"]) for g in D4], 0), tm(z[f"{f}|id|err"])
res = {"families": FAMS, "per_backbone": {}}
rows = []
for bb in ("Transformer", "FNO"):
    out = {}; L = S = 0; dirs = []
    for f in FAMS:
        dp, ep = per_traj(Z[(bb, "prescribed")], f); dg, eg = per_traj(Z[(bb, "generic")], f)
        assert len(dp) == len(dg)
        p = stats.wilcoxon(dg, dp).pvalue; lr = np.log(dg / dp)
        sig = p < 0.05; up = np.median(lr) > 0
        L += sig and up; S += sig and not up; dirs.append(np.median(lr))
        out[f] = dict(n=len(dp), defect_presc=float(dp.mean()), defect_gen=float(dg.mean()), err_presc=float(ep.mean()),
                      err_gen=float(eg.mean()), median_log_ratio=float(np.median(lr)), p=float(p),
                      median_log_ratio_q=float(np.median(np.log((dg / eg) / (dp / ep)))))
        for b, d, e in (("prescribed", dp, ep), ("generic", dg, eg)):
            rows.append(dict(backbone=bb, bank=b, family=f, ldef=float(np.log(d.mean())), lerr=float(np.log(e.mean()))))
    dirs = np.array(dirs)
    res["per_backbone"][bb] = dict(larger=int(L), smaller=int(S), n_fam=len(FAMS), fam=out,
                                   sign_p=float(stats.binomtest(int(L), int(L + S), alternative="greater").pvalue),
                                   wilcoxon_fam_p=float(stats.wilcoxon(dirs, alternative="greater").pvalue),
                                   mean_ratio=float(np.exp(dirs.mean())))
    print(f"{bb}: larger {L} / smaller {S} / n.s. {len(FAMS)-L-S}; sign p {res['per_backbone'][bb]['sign_p']:.4f}; "
          f"family Wilcoxon (median log ratio > 0) p {res['per_backbone'][bb]['wilcoxon_fam_p']:.4f}; "
          f"geo-mean ratio {res['per_backbone'][bb]['mean_ratio']:.3f}")
Lt = sum(res["per_backbone"][b]["larger"] for b in res["per_backbone"]); St = sum(res["per_backbone"][b]["smaller"] for b in res["per_backbone"])
res["pooled_sign_p"] = float(stats.binomtest(Lt, Lt + St, alternative="greater").pvalue)
print(f"pooled sign test {Lt}/{St}: p {res['pooled_sign_p']:.4g}")
# family-level two-way analysis on log mean defect: within-family contrasts
import itertools
fam_eff = {}
for f in FAMS:
    g = {(r["backbone"], r["bank"]): r for r in rows if r["family"] == f}
    bank_T = g[("Transformer", "generic")]["ldef"] - g[("Transformer", "prescribed")]["ldef"]
    bank_F = g[("FNO", "generic")]["ldef"] - g[("FNO", "prescribed")]["ldef"]
    bb_P = g[("FNO", "prescribed")]["ldef"] - g[("Transformer", "prescribed")]["ldef"]
    bb_G = g[("FNO", "generic")]["ldef"] - g[("Transformer", "generic")]["ldef"]
    fam_eff[f] = dict(bank=(bank_T + bank_F) / 2, backbone=(bb_P + bb_G) / 2, interaction=bank_F - bank_T)
for k in ("bank", "backbone", "interaction"):
    v = np.array([fam_eff[f][k] for f in FAMS])
    t = stats.ttest_1samp(v, 0); w = stats.wilcoxon(v)
    res[f"effect_{k}"] = dict(mean_log=float(v.mean()), ratio=float(np.exp(v.mean())), ci=[float(np.exp(v.mean() - 2.201 * v.std(ddof=1) / np.sqrt(len(v)))), float(np.exp(v.mean() + 2.201 * v.std(ddof=1) / np.sqrt(len(v))))],
                              t_p=float(t.pvalue), wilcoxon_p=float(w.pvalue), n_pos=int((v > 0).sum()))
    print(f"{k:12s} ratio {np.exp(v.mean()):.3f} 95% CI [{res[f'effect_{k}']['ci'][0]:.3f}, {res[f'effect_{k}']['ci'][1]:.3f}] "
          f"t p {t.pvalue:.4f} Wilcoxon p {w.pvalue:.4f}  positive in {(v>0).sum()}/{len(v)}")
try:
    import pandas as pd, statsmodels.formula.api as smf
    df = pd.DataFrame(rows); df["gen"] = (df.bank == "generic").astype(int); df["fno"] = (df.backbone == "FNO").astype(int)
    for form in ("ldef ~ gen * fno", "ldef ~ gen * fno + lerr"):
        m = smf.mixedlm(form, df, groups=df["family"]).fit(reml=True)
        print(form); print(m.summary().tables[1].to_string())
        res[f"mixed[{form}]"] = {k: dict(coef=float(m.params[k]), p=float(m.pvalues[k])) for k in m.params.index if k != "Group Var"}
except Exception as e:
    print("mixed model skipped:", e)
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "bank_backbone.json", "w"), indent=1)
