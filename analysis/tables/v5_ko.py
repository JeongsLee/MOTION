# Knockout reliance on the pointwise heads (reaction, reaction_nl, coefficient) in B-R vs V5 (no H0).
import json, numpy as np
from scipy.stats import wilcoxon
A = '../A14_factorial/data'
br, v5 = (np.load(f"{A}/{x}/stats.npz", allow_pickle=True) for x in ("BR", "V5"))
rows = {r['fam']: r for r in json.load(open('v5_h0.json'))}
def traj(z, cfg, f):
    v, w = z[f"{cfg}|{f}|phys"], z[f"{cfg}|{f}|w"]; return ((v*w).sum(-1)/np.maximum(w.sum(-1), 1e-6)).mean(1)
def rel(z, h, f):  # per-trajectory relative KO increase
    return traj(z, f"ko:{h}", f) / traj(z, "base", f) - 1
for f, r in rows.items():
    for h in ("reaction", "reaction_nl", "coefficient"):
        a, b = rel(br, h, f), rel(v5, h, f)
        r[f"ko_{h}"] = [100*np.mean(a), 100*np.mean(b), (wilcoxon(b, a).pvalue if np.any(a != b) else 1.0)]
json.dump(list(rows.values()), open('v5_h0.json', 'w'), indent=1, default=float)
for f, r in sorted(rows.items(), key=lambda t: -t[1]['h0share']):
    print(f"{f[:22]:22s} H0 {r['h0share']:.2f} dErr {r['rel']:+5.1f}% pH {r['p_holm']:.0e} | " + " ".join(
        f"{h[:5]} {r['ko_'+h][0]:+5.0f}->{r['ko_'+h][1]:+5.0f} (p {r['ko_'+h][2]:.0e})" for h in ("reaction", "reaction_nl", "coefficient")))
