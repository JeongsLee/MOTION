"""Pooled tests over the 24 family x backbone strata (generic vs prescribed bank, per-trajectory log defect ratio):
one-sided Wilcoxon on the 24 stratum medians; stratified sign-flip permutation of the paired per-trajectory log
ratios (statistic = mean of stratum medians); family-clustered bootstrap CI of the pooled ratio; backbone x bank
interaction as the paired difference of the two backbones' family medians."""
import json, numpy as np
from scipy import stats
exec(open("bank_backbone.py").read().split("res = {")[0])
rng = np.random.default_rng(0)
LR = {}
for bb in ("Transformer", "FNO"):
    for f in FAMS:
        dp, _ = per_traj(Z[(bb, "prescribed")], f); dg, _ = per_traj(Z[(bb, "generic")], f)
        LR[(bb, f)] = np.log(dg / dp)
med = np.array([np.median(v) for v in LR.values()])
obs = med.mean()
null = np.array([np.mean([np.median(v * rng.choice([-1, 1], len(v))) for v in LR.values()]) for _ in range(20000)])
p_perm = max((null >= obs).mean(), 1 / 20000)
boot = []
for _ in range(10000):
    fs = rng.choice(FAMS, len(FAMS))
    boot.append(np.mean([np.median(LR[(bb, f)][rng.integers(0, len(LR[(bb, f)]), len(LR[(bb, f)]))]) for f in fs for bb in ("Transformer", "FNO")]))
ci = np.exp(np.percentile(boot, [2.5, 97.5]))
inter = np.array([np.median(LR[("FNO", f)]) - np.median(LR[("Transformer", f)]) for f in FAMS])
out = dict(n_strata=len(med), n_pos=int((med > 0).sum()), wilcoxon_one_sided_p=float(stats.wilcoxon(med, alternative="greater").pvalue),
           perm_one_sided_p=float(p_perm), pooled_ratio=float(np.exp(obs)), pooled_ci95=[float(ci[0]), float(ci[1])],
           interaction_wilcoxon_p=float(stats.wilcoxon(inter).pvalue), interaction_mean_ratio=float(np.exp(inter.mean())))
print(json.dumps(out, indent=1)); json.dump(out, open("pooled.json", "w"), indent=1)
