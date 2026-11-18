# V5 (no H0) vs B-R: per-family paired error difference, H0 contribution share, head-share shift.
import json, numpy as np, sys
from scipy.stats import wilcoxon, spearmanr
A = sys.argv[1]
def traj(z, cfg, fam, m):
    v, w = z[f"{cfg}|{fam}|{m}"], z[f"{cfg}|{fam}|w"]
    return ((v * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)).mean(1)
br, v5 = (np.load(f"{A}/{r}/stats.npz", allow_pickle=True) for r in ("BR", "V5"))
cb, cv = (json.load(open(f"{A}/{r}/contrib.json")) for r in ("BR", "V5"))
fams = sorted({k.split("|")[1] for k in br.files if k.startswith("base|")})
rows = []
for f in fams:
    m = "phys"
    eb, ev = traj(br, "base", f, m), traj(v5, "base", f, m)
    p = wilcoxon(ev, eb).pvalue
    C = cb["C"].get(f, {}); tot = sum(C.values())
    h0 = C.get("H0", 0) / tot if tot else np.nan
    sb, sv = cb["share"].get(f, {}), cv["share"].get(f, {})
    shift = {k: sv.get(k, 0) - sb.get(k, 0) for k in sb}
    top = sorted(shift.items(), key=lambda t: -t[1])[:2]
    lab = cb["labels"].get(f, {})
    rows.append(dict(fam=f, n=len(eb), err_br=100*eb.mean(), err_v5=100*ev.mean(), d=100*(ev-eb).mean(),
                     rel=(ev.mean()/eb.mean()-1)*100, p=p, h0share=h0, gain_top=top,
                     src=int(any(lab.get(k, 0) for k in ("reaction", "reaction_nl", "buoyancy"))),
                     ops=[k for k, v in lab.items() if v]))
# Holm
ps = np.array([r["p"] for r in rows]); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
for i, j in enumerate(o): run = max(run, (m - i) * ps[j]); adj[j] = min(run, 1)
for r, a in zip(rows, adj): r["p_holm"] = a
for r in sorted(rows, key=lambda r: -r["d"]):
    print(f"{r['fam'][:26]:26s} n={r['n']:3d} BR {r['err_br']:6.2f} V5 {r['err_v5']:6.2f} d {r['d']:+6.2f} ({r['rel']:+5.1f}%) "
          f"pH {r['p_holm']:.1e} H0sh {r['h0share']:.3f} src {r['src']} top+ {[(k, round(v,3)) for k,v in r['gain_top']]}")
print("mean d over families %.3f" % np.mean([r["d"] for r in rows]))
x = [r["h0share"] for r in rows]; y = [r["rel"] for r in rows]
print("Spearman H0share vs rel. change:", spearmanr(x, y))
json.dump(rows, open(sys.argv[2], "w"), indent=1, default=float)
