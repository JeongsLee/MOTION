"""Hybrid (typed + generic heads) analysis (revision R1; A13).
  python hybrid_analysis.py REF_STATS.npz HY_STATS.npz [--json OUT]
 REF = the model with the same warp but no generic heads (BRW). Reports: (1) generic collapse: single-generic KO damage,
 all-generic KO damage, sum-of-singles vs joint (compensation); (2) bypass: typed-head KO matrix HY vs REF (Spearman,
 key physical cells, total typed damage); (3) accuracy split by family."""
import argparse, json
import numpy as np
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("hy"); ap.add_argument("--json"); a = ap.parse_args()
ZR, ZH = np.load(a.ref), np.load(a.hy)
def tr(Z, cfg, f):
    e, w = Z[f"{cfg}|{f}|phys"], Z[f"{cfg}|{f}|w"]; return ((e * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)).mean(1)
D = lambda Z, h, f: 100 * (tr(Z, "ko:" + h, f).mean() / tr(Z, "base", f).mean() - 1)
FAMS = sorted({k.split("|")[1] for k in ZH.files if k.startswith("base|")})
cfgs = sorted({k.split("|")[0] for k in ZH.files if k.startswith("ko:")})
GEN = sorted([c[3:] for c in cfgs if c[3:].startswith("generic") and "+" not in c])
ALLG = [c[3:] for c in cfgs if "+" in c and c[3:].startswith("generic")]
TYPED = sorted({k.split("|")[0][3:] for k in ZR.files if k.startswith("ko:") and "+" not in k})
OUT = {}
sh = lambda f: f.replace("pdebench3d_cns_", "3D_")[:16]
print("== (1) generic heads: KO damage (%) per family")
print("%-16s %8s %8s %8s %6s  %s" % ("family", "allgen", "sumsgl", "maxsgl", "PR", "base HY / REF"))
rows = {}
for f in FAMS:
    s = np.array([D(ZH, g, f) for g in GEN]); ag = D(ZH, ALLG[0], f) if ALLG else np.nan
    p = np.clip(s, 0, None); pr = (p.sum() ** 2 / (p ** 2).sum()) if p.sum() > 0 else 0
    rows[f] = {"allgen": ag, "singles": s.tolist(), "sum": float(s.sum()), "max": float(s.max()), "argmax": GEN[int(s.argmax())], "PR": float(pr),
               "base_hy": float(100 * tr(ZH, "base", f).mean()), "base_ref": float(100 * tr(ZR, "base", f).mean())}
    print("%-16s %8.1f %8.1f %8.1f %6.2f  %6.2f / %6.2f  top %s" % (sh(f), ag, s.sum(), s.max(), pr, rows[f]["base_hy"], rows[f]["base_ref"], rows[f]["argmax"]))
OUT["generic"] = rows
tops = [rows[f]["argmax"] for f in FAMS]
print("  most damaging generic per family:", {g: tops.count(g) for g in set(tops)})
print("  all-generic KO: HY error after KO vs REF base (does it fall back to the no-generic model?)")
for f in FAMS:
    ko = 100 * tr(ZH, "ko:" + ALLG[0], f).mean() if ALLG else np.nan
    print("   %-16s HY %6.2f -> KO-all %6.2f | REF %6.2f" % (sh(f), rows[f]["base_hy"], ko, rows[f]["base_ref"]))
print("== (2) typed heads: KO matrix HY vs REF")
KR = np.array([[D(ZR, h, f) for f in FAMS] for h in TYPED]); KH = np.array([[D(ZH, h, f) for f in FAMS] for h in TYPED])
rho = spearmanr(KR.ravel(), KH.ravel()).correlation
print(f"  Spearman over typed head x family: {rho:.3f}; top typed head agrees in {(KR.argmax(0) == KH.argmax(0)).sum()}/{len(FAMS)}")
print("  total typed damage (clipped): REF %.0f  HY %.0f" % (np.clip(KR, 0, None).sum(), np.clip(KH, 0, None).sum()))
CELLS = [("advection", "NS-Sines"), ("advection", "NS-Gauss"), ("advection", "incom_ns"), ("advection", "CE-RM"),
         ("diffusion", "NS-Sines"), ("diffusion", "ACE"), ("diffusion", "diff_react"), ("compressible", "CE-CRP"),
         ("compressible", "shallow_water"), ("gradvec", "NS-Sines"), ("gradvec", "CE-RP"), ("reaction_nl", "ACE"),
         ("wave", "Wave-Layer"), ("spectral", "CE-Gauss"), ("shock", "NS-Sines"), ("shock", "CE-RM"), ("geometry", "Poisson-Gauss")]
for h, f in CELLS:
    if h in TYPED and f in FAMS:
        print("   %-13s %-16s REF %7.1f  HY %7.1f" % (h, sh(f), KR[TYPED.index(h), FAMS.index(f)], KH[TYPED.index(h), FAMS.index(f)]))
print("  per head total (REF / HY):")
for k, h in enumerate(TYPED):
    a_, b_ = np.clip(KR[k], 0, None).sum(), np.clip(KH[k], 0, None).sum()
    if a_ + b_ > 2: print("   %-13s %7.1f %7.1f" % (h, a_, b_))
OUT["typed"] = {"spearman": rho, "KR": KR.tolist(), "KH": KH.tolist(), "heads": TYPED, "fams": FAMS}
if a.json: json.dump(OUT, open(a.json, "w"), indent=1, default=float)
