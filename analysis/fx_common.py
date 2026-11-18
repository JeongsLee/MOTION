"""Shared definitions for the specialisation-criteria figures (fig_factorial.py / fig_metadata.py; A14)."""
import json, os
import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = f"{R}/results/A14_factorial/data"
OPS = {"shallow_water": ("advection", "divergence"), "diff_react": ("diffusion", "reaction"),
       "com_ns": ("advection", "diffusion", "divergence", "pressure_grad"),
       "incom_ns": ("advection", "diffusion", "pressure_projection"),
       "cfdbench": ("advection", "diffusion", "pressure_projection"),   # REVISION-R1 10-21: = data/registry.py operators (incompressible NS with walls; PROSE-FD App. A.3); used only by the NS benchmark model
       "pdearena_ns": ("advection", "diffusion", "pressure_projection", "buoyancy"),
       "pdearena_uncond": ("advection", "diffusion", "pressure_projection", "buoyancy"),
       "CE-RP": ("advection", "divergence", "pressure_grad"), "CE-CRP": ("advection", "divergence", "pressure_grad"),
       "CE-KH": ("advection", "divergence", "pressure_grad", "shear"), "CE-Gauss": ("advection", "divergence", "pressure_grad"),
       "CE-RM": ("advection", "divergence", "pressure_grad"), "NS-Sines": ("advection", "diffusion", "pressure_projection"),
       "NS-Gauss": ("advection", "diffusion", "pressure_projection"), "ACE": ("diffusion", "reaction"),
       "Wave-Layer": ("wave",), "Poisson-Gauss": ("laplacian",)}
for _f in ("pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08", "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08",
           "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08"):
    OPS[_f] = ("advection", "diffusion", "divergence", "pressure_grad")
OPEN_CLASS = {"compressible": {"divergence", "pressure_grad"}, "gradvec": {"pressure_grad", "pressure_projection"},
              "spectral": {"pressure_projection", "laplacian"}, "coefficient": {"wave", "diffusion", "laplacian"},
              "geometry": set()}
SAME = [("diffusion", "wave"), ("advection", "shock"), ("advection", "dilatation"), ("shock", "dilatation"),
        ("gradvec", "shear"), ("gradvec", "buoyancy"), ("reaction", "reaction_nl"), ("spectral", "coefficient")]
WITHIN = ["diffusion+wave", "advection+shock", "advection+dilatation", "shock+dilatation", "gradvec+shear",
          "gradvec+buoyancy", "shear+buoyancy", "reaction+reaction_nl", "spectral+coefficient"]
SHORT = {"shallow_water": "SWE", "diff_react": "Diff-React", "com_ns": "Comp. NS", "incom_ns": "Incomp. NS",
         "pdearena_ns": "PDEArena", "pdearena_uncond": "PDEArena-u", "Poisson-Gauss": "Poisson",
         "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": "3D M0.1", "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": "3D M1.0",
         "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": "3D Turb"}
sh = lambda f: SHORT.get(f, f)


def load(c, name):
    p = f"{D}/{c}/{name}"
    if not os.path.exists(p): return None
    return json.load(open(p)) if p.endswith(".json") else np.load(p)


def traj(z, cfg, f, m="phys"):
    e, w = z[f"{cfg}|{f}|{m}"], z[f"{cfg}|{f}|w"]
    return ((e * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)).mean(1)


def ko_matrix(z, heads, fams):
    return np.array([[100 * (traj(z, "ko:" + h, f).mean() / traj(z, "base", f).mean() - 1) for f in fams] for h in heads])


def criteria(K, heads, fams, rel=None, gko=None):
    """K: (heads, fams) KO damage (%). Returns per-unit arrays (families / pairs) for C1..C7."""
    P = np.clip(K, 0, None)
    out = {"C1": P.max(0)}
    oi = [heads.index(h) for h in OPEN_CLASS if h in heads]
    Q = P[oi]; keep = Q.sum(0) > 1
    Qn = Q[:, keep] / np.linalg.norm(Q[:, keep], axis=0, keepdims=True)
    S = Qn.T @ Qn; iu = np.triu_indices(S.shape[0], 1); out["C2"] = 1 - S[iu]
    c3 = []
    for j, f in enumerate(fams):
        tot = P[oi, j].sum()
        if tot < 1: continue
        c3.append(sum(P[heads.index(h), j] for h in OPEN_CLASS if h in heads and OPEN_CLASS[h] & set(OPS[f])) / tot)
    out["C3"] = np.array(c3)
    out["C4"] = None
    if rel is not None:
        pre = [r for r in rel if r["pre"]]; sig = [r for r in pre if r.get("p_holm", 1) < .05]
        out["C4"] = (len([r for r in sig if r["rho"] > 0]), len(sig))
    out["C5"] = np.array([(P[:, j].sum() ** 2 / (P[:, j] ** 2).sum()) if P[:, j].sum() > 0 else np.nan
                          for j in range(len(fams))])
    c6 = []
    for a, b in SAME:
        if a in heads and b in heads:
            x, y = P[heads.index(a)], P[heads.index(b)]
            if x.sum() > 1 and y.sum() > 1:
                c6.append(1 - x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
            elif x.sum() > 1 or y.sum() > 1:
                c6.append(1.0)                         # only one of the two is used: fully separated
    out["C6"] = np.array(c6)
    out["C7"] = None
    if gko is not None:
        fz = sorted({k.split("|")[1] for k in gko.files if k.startswith("base|")})
        Dg = lambda cfg, f: 100 * (traj(gko, "ko:" + cfg, f).mean() / traj(gko, "base", f).mean() - 1)
        I = []
        for p in WITHIN:
            x, y = p.split("+")
            for f in fz:
                da, db = Dg(x, f), Dg(y, f)
                if max(abs(da), abs(db)) >= 1:
                    I.append(Dg(p, f) - da - db)
        out["C7"] = np.array(I)
    return out


def summ(v, rng):
    if v is None: return None
    if isinstance(v, tuple): return {"value": v[0] / max(v[1], 1), "n": list(v)}
    v = v[np.isfinite(v)]
    if not len(v): return None
    b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
    return {"value": float(v.mean()), "ci": [float(x) for x in np.percentile(b, [2.5, 97.5])]}
