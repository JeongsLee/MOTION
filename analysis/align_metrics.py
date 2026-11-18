"""Revision R1: mechanism-differentiation metrics on RELATIVE RELIANCE (head contribution shares), replacing the
knockout-damage D/A of the first draft (A had no power against a relabeling null: BR .48 vs .45 by chance).

Basis: eval_contrib.py contrib.json, share[f][k] = mean |sigma_k . head_k(h)| into the latent rate W / sum over heads.

D  differentiation. Families' reliance profiles over the five always-on heads (never gated by metadata), renormalised
   per family; D = mean over family pairs of 1 - cos. Reference lines:
     D_PHYS  the differentiation physics asks for: every family relies uniformly on the always-on heads whose operator
             class occurs in its equation (shared mechanisms keep this well below 1; 0.43 for the 19-family corpus);
     D_MAX   the geometric maximum with five heads (balanced one-hot assignment; 0.84 for 19 families).
AUC operator selectivity, per family over the heads OPEN for that family (metadata-opened + always-on; every typed
   head in models trained without metadata). Each head's share is first standardised over the families in which it is
   open (removes the head's natural contribution scale), then per family the AUC of the standardised shares of the
   open heads whose operator occurs in the equation against the open heads whose operator does not; mean over
   families. Chance = 0.5; permutation p (family labels shuffled within each head). With metadata, which heads are
   open already follows the equation, so a generic bank inherits part of this selectivity; the decisive comparison
   is between models trained without metadata. (First version, always-on heads only: align_metrics.py.bak_alwayson.)
  python align_metrics.py [cells ...]        (prints and writes 03_analyses/A14_factorial/align_metrics.json)
"""
import json, os, sys
import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fx_common import OPS, OPEN_CLASS, D as DDIR

TOK = {"reaction": "reaction", "diffusion": "diffusion", "buoyancy": "buoyancy", "shear": "shear", "wave": "wave",
       "advection": "advection", "shock": "advection", "elliptic": "elliptic", "bernoulli": "pressure_projection",
       "dilatation": "divergence", "reaction_nl": "reaction"}               # gated heads -> metadata token (banks.OP_INDEX)
ALWAYS = {"compressible": {"divergence", "pressure_grad"}, "gradvec": {"pressure_grad", "pressure_projection"},
          "spectral": {"pressure_projection", "laplacian"}, "coefficient": {"wave", "diffusion", "laplacian"},
          "kinematic": {"advection", "shear"}}                              # always-on heads with an operator class
CLS = {**{h: {t} for h, t in TOK.items()}, **ALWAYS}
OPEN5 = list(OPEN_CLASS)                                                    # D profile heads (incl. geometry)
NOMETA = {"V2", "V1aN", "V1aV3N"}
AUC_CHANCE = 0.5


def shares(c):
    p = f"{DDIR}/{c}/contrib.json"
    if not os.path.exists(p): return None
    d = json.load(open(p)); heads = d["heads"]
    fams = [f for f in d["share"] if f in OPS]
    S = np.array([[(d["share"][f][i] if isinstance(d["share"][f], list) else d["share"][f].get(h, 0.0))
                   for i, h in enumerate(heads)] for f in fams], float)
    return heads, fams, S


def dmean(Q):                                                               # Q (fams, heads) >= 0
    keep = Q.sum(1) > 0
    Qn = Q[keep] / np.linalg.norm(Q[keep], axis=1, keepdims=True)
    G = Qn @ Qn.T; iu = np.triu_indices(G.shape[0], 1)
    return float(np.mean(1 - G[iu])), int(keep.sum())


def d_refs(fams):
    L = np.array([[1.0 if OPEN_CLASS[h] & set(OPS[f]) else 0.0 for h in OPEN5] for f in fams])
    n = len(fams); g = [n // 5 + (1 if i < n % 5 else 0) for i in range(5)]
    return dmean(L)[0], 1 - sum(k * (k - 1) // 2 for k in g) / (n * (n - 1) // 2)


def _auc(S, lab):
    a = []
    for j in range(S.shape[1]):
        y = lab[:, j]
        if y.all() or (~y).all(): continue
        a.append(mannwhitneyu(S[y, j], S[~y, j]).statistic / (y.sum() * (~y).sum()))
    return (float(np.mean(a)) if a else float("nan")), len(a)


ZERO = {"vortex", "boundary_layer"}                                    # identically zero in 2D / without an SDF


def _open_match(heads, fams, nometa):
    O = np.zeros((len(fams), len(heads)), bool); M = np.zeros_like(O)
    for i, f in enumerate(fams):
        ops = set(OPS[f])
        for j, h in enumerate(heads):
            if h in ZERO or h.startswith("generic"): continue
            if h in TOK:
                if nometa or TOK[h] in ops: O[i, j] = True; M[i, j] = TOK[h] in ops
            elif h in ALWAYS: O[i, j] = True; M[i, j] = bool(ALWAYS[h] & ops)
            elif h == "geometry": O[i, j] = True
    return O, M


def _fam_auc(Z, O, M):
    v = []
    for i in range(Z.shape[0]):
        ok = O[i] & np.isfinite(Z[i]); m = M[i][ok]; x = Z[i][ok]
        if m.all() or (~m).all(): continue
        a, b = x[m][:, None], x[~m][None, :]
        v.append(((a > b) + 0.5 * (a == b)).mean())
    return float(np.mean(v)) if v else float("nan"), len(v)


def metrics(c, nperm=1000, seed=0):
    r = shares(c)
    if r is None: return None
    heads, fams, S = r
    oi = [heads.index(h) for h in OPEN5 if h in heads]
    Dv, nf = dmean(S[:, oi] / np.maximum(S[:, oi].sum(1, keepdims=True), 1e-12))
    dphys, dmax = d_refs(fams)
    O, M = _open_match(heads, fams, c in NOMETA)
    Z = np.full_like(S, np.nan)
    for j in range(S.shape[1]):
        col = S[O[:, j], j]
        if len(col) > 2 and col.std() > 0: Z[O[:, j], j] = (col - col.mean()) / col.std()
    auc, nfa = _fam_auc(Z, O, M)
    rng = np.random.default_rng(seed); null = []
    for _ in range(nperm):
        Zp = Z.copy()
        for j in range(S.shape[1]):
            idx = np.where(O[:, j] & np.isfinite(Z[:, j]))[0]; Zp[idx, j] = Z[rng.permutation(idx), j]
        null.append(_fam_auc(Zp, O, M)[0])
    return {"D_contrib": Dv, "D_phys": dphys, "D_max": dmax, "n_fams": nf,
            "AUC": auc, "AUC_p": float((np.sum(np.array(null) >= auc) + 1) / (nperm + 1)), "AUC_fams": nfa,
            "open_heads_per_family": float(O.sum(1).mean()), "metadata": c not in NOMETA}


if __name__ == "__main__":
    cells = sys.argv[1:] or sorted(d for d in os.listdir(DDIR) if os.path.exists(f"{DDIR}/{d}/contrib.json"))
    out = {}
    for c in cells:
        m = metrics(c)
        if m is None: continue
        out[c] = m
        print(f"{c:10s} AUC {m['AUC']:.2f} p {m['AUC_p']:.3f} ({m['open_heads_per_family']:.1f} open heads/family, "
              f"metadata {m['metadata']}) | D refs phys {m['D_phys']:.2f} max {m['D_max']:.2f}", flush=True)
    json.dump(out, open(os.path.join(DDIR, "..", "align_metrics.json"), "w"), indent=1)
