"""Group knockouts (revision R1-2, V2 'limited sharing' hypothesis). Input: groupko.npz from run_groups.sh.
 Interaction I(a,b) = D(a+b) - D(a) - D(b) (D = % error increase). I > 0: compensation (shared function),
 I < 0: overlap of damage. Within-group (same a-priori operator class) vs across-group pairs.
 Group-KO selectivity: AUC of group-KO damage, families whose a-priori operator list contains the group's class
 vs the rest.   python group_analysis.py V2.npz BR.npz [--json OUT]"""
import argparse, json
import numpy as np
from scipy.stats import mannwhitneyu
def roc_auc_score(y, d):
    return mannwhitneyu(d[y], d[~y]).statistic / (y.sum() * (~y).sum())
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--json"); a = ap.parse_args()
rng = np.random.default_rng(0)
OPS = {"shallow_water": ("advection", "divergence"), "diff_react": ("diffusion", "reaction"),
       "com_ns": ("advection", "diffusion", "divergence", "pressure_grad"),
       "incom_ns": ("advection", "diffusion", "pressure_projection"),
       "pdearena_ns": ("advection", "diffusion", "pressure_projection", "buoyancy"),
       "pdearena_uncond": ("advection", "diffusion", "pressure_projection", "buoyancy"),
       "CE-RP": ("advection", "divergence", "pressure_grad"), "CE-CRP": ("advection", "divergence", "pressure_grad"),
       "CE-KH": ("advection", "divergence", "pressure_grad", "shear"), "CE-Gauss": ("advection", "divergence", "pressure_grad"),
       "CE-RM": ("advection", "divergence", "pressure_grad"), "NS-Sines": ("advection", "diffusion", "pressure_projection"),
       "NS-Gauss": ("advection", "diffusion", "pressure_projection"), "ACE": ("diffusion", "reaction"),
       "Wave-Layer": ("wave",), "Poisson-Gauss": ("laplacian",)}
GROUPS = {"diffusion+wave": {"diffusion", "wave", "laplacian"},                         # second-order
          "advection+shock+dilatation": {"advection", "divergence"},                    # transport / compression
          "gradvec+shear+buoyancy": {"pressure_grad", "shear", "buoyancy"},             # gradient / forcing
          "reaction+reaction_nl+elliptic": {"reaction", "laplacian", "pressure_projection"},  # local / global
          "spectral+coefficient": {"pressure_projection", "laplacian"},                # nonlocal (exploratory)
          "geometry+compressible": {"divergence", "pressure_grad"}}                     # compressible coupling
CLASS = {h: g for g in GROUPS for h in g.split("+")}
WITHIN = "diffusion+wave advection+shock advection+dilatation shock+dilatation gradvec+shear gradvec+buoyancy shear+buoyancy reaction+reaction_nl spectral+coefficient".split()
ACROSS = "diffusion+advection wave+gradvec shock+reaction diffusion+gradvec advection+reaction_nl shear+spectral wave+shock dilatation+coefficient".split()
OUT = {}
for path in a.files:
    Z = np.load(path); tag = path.split("/")[-1].split("_")[0]
    fams = sorted({k.split("|")[1] for k in Z.files if k.startswith("base|")})
    def tr(cfg, f):
        e, w = Z[f"{cfg}|{f}|phys"], Z[f"{cfg}|{f}|w"]; return ((e * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)).mean(1)
    D = lambda cfg, f: 100 * (tr("ko:" + cfg, f).mean() / tr("base", f).mean() - 1)
    res = {"pairs": {}}
    for kind, P in (("within", WITHIN), ("across", ACROSS)):
        vals = []
        for p in P:
            x, y = p.split("+"); I = []
            for f in fams:
                da, db = D(x, f), D(y, f)
                if max(abs(da), abs(db)) < 1: continue                  # both heads inert/closed for f
                I.append(D(p, f) - da - db)
            res["pairs"][p] = (float(np.mean(I)) if I else np.nan, len(I)); vals += I
        v = np.array(vals); bi = rng.integers(0, len(v), (2000, len(v))) if len(v) else None
        res[kind] = (float(v.mean()) if len(v) else np.nan, *(np.percentile(v[bi].mean(1), [2.5, 97.5]).tolist() if len(v) else [np.nan] * 2), len(v),
                     float(np.mean(np.abs(v))) if len(v) else np.nan)
    res["group_auc"] = {}
    for g, cls in GROUPS.items():
        d = np.array([D(g, f) for f in fams]); y = np.array([bool(cls & set(OPS[f])) for f in fams])
        auc = roc_auc_score(y, d) if 0 < y.sum() < len(y) else np.nan
        res["group_auc"][g] = (float(auc), float(d[y].mean()), float(d[~y].mean()), int(y.sum()))
    res["top_group"] = {}
    for f in fams:
        dg = {g: D(g, f) for g in GROUPS}; top = max(dg, key=dg.get)
        res["top_group"][f] = (top, round(dg[top], 1), bool(GROUPS[top] & set(OPS[f])))
    OUT[tag] = res
    print(f"===== {tag}")
    for kind in ("within", "across"):
        m, lo, hi, n, ab = res[kind]; print(f"  {kind:6s} interaction mean {m:+6.2f} [{lo:+.2f},{hi:+.2f}]  |I| {ab:5.2f}  n={n}")
    for p, (m, n) in res["pairs"].items(): print(f"     {p:24s} {m:+7.2f} (n={n})")
    print("  group-KO selectivity (AUC, mean damage relevant / other):")
    for g, (auc, dr, do, n) in res["group_auc"].items(): print(f"     {g:30s} AUC {auc:.2f}  {dr:7.1f} / {do:7.1f}  (n_rel={n})")
    nt = sum(v[2] for v in res["top_group"].values())
    print(f"  top group is a-priori relevant in {nt}/{len(fams)} families")
    for f, v in res["top_group"].items(): print(f"     {f:16s} {v[0]:30s} {v[1]:7.1f}  {'ok' if v[2] else '--'}")
if a.json: json.dump(OUT, open(a.json, "w"), indent=1, default=float)
