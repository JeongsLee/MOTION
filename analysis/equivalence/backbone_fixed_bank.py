"""Backbone effect at a fixed bank (FNO vs axial Transformer), compared on the same scale with the bank effect at a
fixed backbone (generic vs prescribed): per family paired Wilcoxon (larger/smaller/unchanged), geometric-mean ratio of
the dihedral defect and of the defect-to-error ratio q = defect / error."""
import json, numpy as np
from scipy import stats
exec(open("bank_backbone.py").read().split("res = {")[0])
def comp(a, b):          # ratio b / a per family
    L=S=0; md=[]; mq=[]; me=[]
    for f in FAMS:
        da, ea = per_traj(Z[a], f); db, eb = per_traj(Z[b], f)
        lr = np.log(db / da); p = stats.wilcoxon(db, da).pvalue
        L += p < .05 and np.median(lr) > 0; S += p < .05 and np.median(lr) < 0
        md.append(np.median(lr)); mq.append(np.median(np.log((db / eb) / (da / ea)))); me.append(np.median(np.log(eb / ea)))
    md, mq, me = map(np.array, (md, mq, me))
    return dict(larger=int(L), smaller=int(S), unchanged=len(FAMS) - int(L) - int(S),
                defect_ratio=float(np.exp(md.mean())), p_defect=float(stats.wilcoxon(md).pvalue),
                q_ratio=float(np.exp(mq.mean())), p_q=float(stats.wilcoxon(mq).pvalue), error_ratio=float(np.exp(me.mean())),
                abs_log_defect=float(np.abs(md).mean()))
out = {"backbone (prescribed): FNO vs Transformer": comp(("Transformer", "prescribed"), ("FNO", "prescribed")),
       "backbone (generic): FNO vs Transformer": comp(("Transformer", "generic"), ("FNO", "generic")),
       "bank (Transformer): generic vs prescribed": comp(("Transformer", "prescribed"), ("Transformer", "generic")),
       "bank (FNO): generic vs prescribed": comp(("FNO", "prescribed"), ("FNO", "generic"))}
for k, v in out.items():
    print(f"{k:44s} L/S/U {v['larger']}/{v['smaller']}/{v['unchanged']}  defect x{v['defect_ratio']:.3f} (p {v['p_defect']:.3f})  "
          f"defect/err x{v['q_ratio']:.3f} (p {v['p_q']:.3f})  error x{v['error_ratio']:.3f}  mean|log| {v['abs_log_defect']:.3f}")
json.dump(out, open("backbone_fixed_bank.json", "w"), indent=1)
