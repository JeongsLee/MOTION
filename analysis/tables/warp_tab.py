# B-R (MOTION-S) vs B-R-W (+ live carrier): error, carrier knockout, advection/upwind-head knockout reliance.
import json, numpy as np
from scipy.stats import wilcoxon
A = '../A14_factorial/data'
br, bw = (np.load(f"{A}/{x}/stats.npz", allow_pickle=True) for x in ("BR", "BRW"))
def traj(z, c, f):
    v, w = z[f"{c}|{f}|phys"], z[f"{c}|{f}|w"]; return ((v*w).sum(-1)/np.maximum(w.sum(-1), 1e-6)).mean(1)
def rel(z, c, f): return 100*(traj(z, c, f).mean()/traj(z, "base", f).mean()-1)
out = {}
fams = sorted({k.split("|")[1] for k in bw.files if k.startswith("base|")})
for f in fams:
    eb, ew = traj(br, "base", f), traj(bw, "base", f)
    r = dict(err_br=100*eb.mean(), err_bw=100*ew.mean(), p=wilcoxon(ew, eb).pvalue, warp_off=rel(bw, "warp_off", f),
             adv=[rel(br, "ko:advection", f), rel(bw, "ko:advection", f)], shock=[rel(br, "ko:shock", f), rel(bw, "ko:shock", f)])
    out[f] = r
    print(f"{f[:22]:22s} E {r['err_br']:6.2f}->{r['err_bw']:6.2f} p {r['p']:.0e} warp_off {r['warp_off']:+6.1f} adv {r['adv'][0]:+5.1f}->{r['adv'][1]:+5.1f} upw {r['shock'][0]:+5.1f}->{r['shock'][1]:+5.1f}")
json.dump(out, open('warp_tab.json', 'w'), indent=1, default=float)
