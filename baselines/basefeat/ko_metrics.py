"""D / AUC of the feature-group knockouts of the B1/P1 feature baselines (2026-11-05).

  python ko_metrics.py <joblog> [<joblog> ...]     (vesslctl job logs of the *-feat-ko jobs, SUMMARY lines)
  -> prints the tables and writes 03_analyses/A30_basefeat_ko/ko_metrics.json

Damage of group g in family f: 100 * (error with g zeroed / error without knockout - 1), physical rel-L2 FULLTEST on
the first 100 test trajectories per family (final checkpoint). Reliance = max(damage, 0).
D: mean over family pairs of 1 - cos of the families' reliance profiles over the four groups (as for MOTION, where the
profile runs over the five always-open heads). D_phys: equal reliance on the groups whose operator class occurs in the
family's equation (fx_common.OPS; advection->adv, diffusion->lap, divergence->div, pressure_grad /
pressure_projection->grad). D_max: geometric maximum with four groups and six families.
AUC: damage of each group standardised over families, then per family the AUC of matched against unmatched groups;
mean over the families that have both (com_ns has all four classes and drops out); one-sided permutation p, family
labels permuted within each group (exhaustive over 10,000 random permutations).
CI: the evaluation prints running means in batches of six trajectories; batch means are recovered from them and
resampled (paired across knockouts, 2,000 bootstrap draws) for 95% intervals of each damage.
CAVEAT: the raw input channels stay in the model, so a group knockout measures reliance on the precomputed feature,
not on the operation; a model can recompute the same derivative from the raw fields.
"""
import json, os, re, sys
import numpy as np

FAMS = ["shallow_water", "com_ns", "incom_ns", "incom_ns_arena", "cfdbench", "incom_ns_arena_u"]
OPSF = {"shallow_water": ("advection", "divergence"), "com_ns": ("advection", "diffusion", "divergence", "pressure_grad"),
        "incom_ns": ("advection", "diffusion", "pressure_projection"), "cfdbench": ("advection", "diffusion", "pressure_projection"),
        "incom_ns_arena": ("advection", "diffusion", "pressure_projection", "buoyancy"),
        "incom_ns_arena_u": ("advection", "diffusion", "pressure_projection", "buoyancy")}   # = fx_common.OPS
GROUPS = ["grad", "lap", "adv", "div"]
GMAP = {"grad": {"pressure_grad", "pressure_projection"}, "lap": {"diffusion"}, "adv": {"advection"}, "div": {"divergence"}}
OUT = os.environ.get("MOTION_RESULTS", "results") + "/A30_basefeat_ko"
rng = np.random.default_rng(0)


def parse(path):
    S = {}
    for line in open(path, errors="ignore"):
        m = re.search(r"SUMMARY (\S+) (\S+) (\S+)", line)
        if m:
            pts = [tuple(map(float, p.split(":"))) for p in m.group(3).strip(",").split(",") if p]
            n = np.array([p[0] for p in pts]); mu = np.array([p[1] for p in pts])
            o = np.argsort(n); n, mu = n[o], mu[o]
            tot = n * mu; bs = np.diff(np.r_[0, tot]); bn = np.diff(np.r_[0, n])
            S[(m.group(1), m.group(2))] = (bs / bn, bn)          # batch means, batch sizes
    return S


def dmean(Q):
    Q = np.asarray(Q, float); keep = Q.sum(1) > 0
    Qn = Q[keep] / np.linalg.norm(Q[keep], axis=1, keepdims=True)
    G = Qn @ Qn.T; iu = np.triu_indices(G.shape[0], 1)
    return float(np.mean(1 - G[iu]))


def matched(f):
    return np.array([bool(GMAP[g] & set(OPSF[f])) for g in GROUPS])


def auc(dmg):                                                    # dmg (fams, groups)
    Z = (dmg - dmg.mean(0)) / (dmg.std(0) + 1e-12)
    vals = []
    for i, f in enumerate(FAMS):
        m = matched(f)
        if m.all() or not m.any(): continue
        a, b = Z[i, m], Z[i, ~m]
        vals.append(np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b]))
    return float(np.mean(vals)), len(vals)


def main(paths):
    S = {}
    for p in paths: S.update(parse(p))
    tags = ["none"] + GROUPS + ["grad_lap_adv_div"]
    W = {}
    for t in tags:
        for f in FAMS:
            bm, bn = S[(t, f)]
            W[(t, f)] = (bm, bn)
    base = np.array([np.average(W[("none", f)][0], weights=W[("none", f)][1]) for f in FAMS])
    err = {t: np.array([np.average(W[(t, f)][0], weights=W[(t, f)][1]) for f in FAMS]) for t in tags}
    dmg = np.stack([100 * (err[g] / base - 1) for g in GROUPS], 1)            # (fams, groups)
    alld = 100 * (err["grad_lap_adv_div"] / base - 1)
    R = np.clip(dmg, 0, None)
    L = np.array([matched(f).astype(float) for f in FAMS])
    n = len(FAMS); g = [n // 4 + (1 if i < n % 4 else 0) for i in range(4)]
    dmax = 1 - sum(k * (k - 1) // 2 for k in g) / (n * (n - 1) // 2)
    D, Dphys = dmean(R), dmean(L)
    kept = R.sum(1) > 0; Dphys_kept = dmean(L[kept])            # reference on the families with nonzero reliance (= those entering D)
    A, na = auc(dmg)
    perm = []
    for _ in range(10000):
        P = np.stack([dmg[rng.permutation(n), j] for j in range(4)], 1)
        perm.append(auc(P)[0])
    p = float((1 + np.sum(np.array(perm) >= A - 1e-12)) / (1 + len(perm)))
    # paired batch bootstrap of the damages
    ci = np.zeros((n, 4, 2)); Db = []
    for i, f in enumerate(FAMS):
        bb, bn = W[("none", f)]; k = len(bb)
        draws = rng.integers(0, k, (2000, k))
        b0 = (bb[draws] * bn[draws]).sum(1) / bn[draws].sum(1)
        for j, gname in enumerate(GROUPS):
            bk, _ = W[(gname, f)]
            bk_ = (bk[draws] * bn[draws]).sum(1) / bn[draws].sum(1)
            ci[i, j] = np.percentile(100 * (bk_ / b0 - 1), [2.5, 97.5])
    res = {"families": FAMS, "groups": GROUPS, "base_err": base.tolist(), "damage": dmg.tolist(), "damage_ci": ci.tolist(),
           "all_groups_damage": alld.tolist(), "D": D, "D_phys": Dphys, "D_phys_kept": Dphys_kept, "fams_in_D": [f for f, k in zip(FAMS, kept) if k], "D_max": dmax, "AUC": A, "AUC_n_fams": na, "AUC_p": p}
    print("base err:", dict(zip(FAMS, np.round(base, 3))))
    print("damage % (fams x grad lap adv div):")
    for i, f in enumerate(FAMS):
        print(f"  {f:18s}", "  ".join(f"{dmg[i, j]:7.2f} [{ci[i, j, 0]:.1f},{ci[i, j, 1]:.1f}]" for j in range(4)),
              f"| all {alld[i]:.1f}", "matched", "".join("x" if m else "." for m in matched(f)))
    print(f"D={D:.3f} over {int(kept.sum())} fams (phys same fams {Dphys_kept:.3f}; all six {Dphys:.3f}, max {dmax:.3f})  AUC={A:.3f} over {na} fams, p={p:.3f}")
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for path in sys.argv[1:]:
        kind = "bcat" if "bcat" in os.path.basename(path) else "prose"
        print("=====", kind); out[kind] = main([path])
    json.dump(out, open(f"{OUT}/ko_metrics.json", "w"), indent=1)
