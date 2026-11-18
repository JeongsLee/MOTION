"""REVISION-R1 (10-22, DRAFT; R1-5 bootstrap intervals, R2 paired tests, R3-3 error bars): trajectory-level equivariance
statistics for the three released weights (MOTION-S, MOTION-NS, later MOTION-IVP-S) and their baselines on the same
held-out trajectories.

Per trajectory (frame / lead mean, %), over the ADMITTED members of the family (Supplementary Note 4):
  replica ratio  rho = mean_g err_g / err_id   (the submitted SI 'replica accuracy ratio'; 1 = equivariant)
  defect ratio   q   = mean_g defect_g / err_id (Fig. 4b of the submitted version)
  gain           G   = (err_ens - err_id) / err_id, ens = the full admitted set (K8xT3 / K8 / K2)
Admitted sets: D4 x {0, 1, 8}-cell shifts (23 members) when translation is admitted, D4 (7) otherwise; one mirror for
CFDBench (scalar row mirror) and PDEArena (row mirror). A pair uses the members present for BOTH models.
Within each model and family: median rho with a bootstrap 95% CI, one-sided Wilcoxon for G < 0, fraction of trajectories
with q > 2. Paired (same trajectories): two-sided Wilcoxon on log rho, log q and G, Holm over the families of the pair,
stratified sign-flip permutation and family-clustered bootstrap of the pooled median difference, TOST on
log(err_id ratio) with margin log 2.
  python stats_equiv3.py OUT_PREFIX   (paths below)"""
import json
import os
import sys

import numpy as np
from scipy import stats

A = os.environ.get("MOTION_RESULTS", "results")
SRC = {   # model -> list of (kind, path)
    "MOTION-S": [("npz", f"{A}/equiv_full/m1f_all/equiv.npz")],
    "MOTION-S (pch)": [("npz", f"{A}/equiv_full/m1f_pch/equiv.npz")],
    "MOTION-NS": [("npz", f"{A}/nss_equiv_zsc/run4/equiv.npz"), ("npz", f"{A}/nss_equiv_zsc/equiv_run3_anisotropic_ok.npz", ("cfdbench", "pdearena_ns", "pdearena_uncond"))],
    # run3 periodic rows are contaminated (fr_ns in the D4 list, REVISION_LOG 10-21) -> periodic families from run4 only
    "Poseidon-T": [("poseidon", f"{A}/poseidon_equiv/Poseidon-T_equiv_test_n128.json")],
    "MOTION": [("npz", f"{A}/equiv_full/m2s2_all/equiv.npz")],
    "MOTION (pch)": [("npz", f"{A}/equiv_full/m2s2_pch/equiv.npz")],
    "Poseidon-B": [("poseidon", f"{A}/poseidon_equiv/Poseidon-B_equiv_test_n128.json")],
    "MOTION-IVP": [("npz", f"{A}/ivps_equiv_gn3w/equiv.npz")],   # REVISION-R1 10-28: gn3w, test 128, rho,u,v,p
    "BCAT": [("baseline", f"{A}/equiv_full/baseline_pertraj.json")],
    "PROSE-FD": [("baseline", f"{A}/equiv_full/baseline_pertraj.json")],
}
PAIRS = [("MOTION-IVP", "Poseidon-T"), ("MOTION-IVP", "Poseidon-B"), ("MOTION-S (pch)", "Poseidon-T"), ("MOTION (pch)", "Poseidon-B"), ("MOTION (pch)", "Poseidon-T"), ("MOTION-NS", "BCAT"), ("MOTION-NS", "PROSE-FD")]
BASE_FAM = {"shallow_water": "shallow_water", "com_ns": "com_ns", "incom_ns": "incom_ns",
            "pdearena_ns": "incom_ns_arena", "pdearena_uncond": "incom_ns_arena_u", "cfdbench": "cfdbench"}
D4 = ["id", "fr", "fc", "r180", "d1", "r90", "r270", "d2"]
rng = np.random.default_rng(0)


def load(model):
    """-> {family: {'err': {member: (n,)}, 'defect': {...}, 'ens': (n,), 'ens_name': str}} (identity member 'id')."""
    out = {}
    for src in SRC[model]:
        kind, path, only = (src + (None,))[:3]
        if not os.path.exists(path):
            print(f"  [{model}] missing {path}"); continue
        if kind == "npz":
            z = np.load(path, allow_pickle=True); S = json.loads(str(z["__meta__"]))["summary"]
            for fam, s in S.items():
                if only and fam not in only: continue
                w = z[f"{fam}|w"]; tm = lambda a: 100 * (a * w).sum(1) / w.sum(1)
                names = [k.split("|")[1] for k in z.files if k.startswith(f"{fam}|") and k.endswith("|err")
                         and not k.split("|")[1].startswith("ens_")]
                if not s.get("shift_admissible", False):
                    names = [n for n in names if "_s" not in n]
                ens = ("K2" if "K2" in s["ens"] else "K8xT3" if ("K8xT3" in s["ens"] and s.get("shift_admissible", False))
                       else "K8")                     # eval_equiv forms K8xT3 even when translation is not admitted
                out[fam] = dict(err={n: tm(z[f"{fam}|{n}|err"]) for n in names},
                                defect={n: tm(z[f"{fam}|{n}|defect"]) for n in names},
                                ens=tm(z[f"{fam}|ens_{ens}|err"]), ens_name=ens)
                if fam == "cfdbench" and "fr_ns" in out[fam]["err"]:          # the admitted scalar mirror only
                    for k in ("err", "defect"):
                        out[fam][k] = {n: v for n, v in out[fam][k].items() if n in ("id", "fr_ns")}
                if fam.startswith("pdearena"):
                    for k in ("err", "defect"):
                        out[fam][k] = {n: v for n, v in out[fam][k].items() if n in ("id", "fr")}
        elif kind == "poseidon":
            for fam, r in json.load(open(path))["res"].items():
                t = r["per_traj"]
                out[fam] = dict(err={k: np.array(v) for k, v in t["err"].items()},
                                defect={k: np.array(v) for k, v in t["defect"].items()},
                                ens=np.array(t["ens"]["K8xT3"]), ens_name="K8xT3")
        else:
            J = json.load(open(path))
            for fam, bf in BASE_FAM.items():
                r = J.get(f"{model}|{bf}")
                if r is None: continue
                if fam == "cfdbench":
                    f = r["frns"]
                    out[fam] = dict(err={"id": np.array(f["id_err_2ch"]), "fr_ns": np.array(f["err"])},
                                    defect={"id": 0 * np.array(f["err"]), "fr_ns": np.array(f["defect"])},
                                    ens=np.array(f["K2"]), ens_name="K2")
                    continue
                mem = ["id", "fr"] if fam.startswith("pdearena") else list(r["err"])
                ens = "K2" if fam.startswith("pdearena") else "K8xT3" if "K8xT3" in r["ens"] else "K8"
                out[fam] = dict(err={k: np.array(r["err"][k]) for k in mem},
                                defect={k: np.array(r["defect"][k]) for k in mem},
                                ens=np.array(r["ens"][ens]), ens_name=ens)
    return out


def metrics(d, members):
    e0 = d["err"]["id"]
    eg = np.mean([d["err"][m] for m in members], 0); dg = np.mean([d["defect"][m] for m in members], 0)
    return e0, np.log(eg / e0), np.log(dg / e0), (d["ens"] - e0) / e0


def boot_median(x, n=10000):
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], 1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


OUTP = sys.argv[1] if len(sys.argv) > 1 else f"{A}/equiv_full/stats3"
data = {m: load(m) for m in SRC}
res = {"within": {}, "pairs": {}}
for m, D in data.items():                                   # ---- within-model
    res["within"][m] = {}
    for fam, d in D.items():
        mem = [k for k in d["err"] if k != "id"]
        e0, lr, lq, G = metrics(d, mem)
        res["within"][m][fam] = dict(n=int(len(e0)), members=len(mem), ens=d["ens_name"], err_mean=float(e0.mean()),
                                     rho_median=float(np.exp(np.median(lr))), rho_ci=list(np.exp(boot_median(lr))),
                                     gain_median=float(100 * np.median(G)), gain_mean=float(100 * G.mean()),
                                     p_gain_lt0=float(stats.wilcoxon(G, alternative="less").pvalue),
                                     frac_q_gt2=float((np.exp(lq) > 2).mean()))
for a, b in PAIRS:                                          # ---- paired
    DA, DB = data[a], data[b]; fams = [f for f in DA if f in DB]
    if not fams: print(f"[pair {a} / {b}] no common family yet"); continue
    rows, diffs = {}, {"log_rho": [], "log_q": [], "gain": []}
    for fam in fams:
        mem = [k for k in DA[fam]["err"] if k != "id" and k in DB[fam]["err"]]
        ea, *ma = metrics(DA[fam], mem); eb, *mb = metrics(DB[fam], mem)
        if len(ea) != len(eb):
            print(f"  [{a}/{b} {fam}] n mismatch {len(ea)} vs {len(eb)}; skipped"); continue
        r = {"n": int(len(ea)), "members": len(mem)}
        for k, xa, xb in zip(diffs, ma, mb):
            diffs[k].append(xa - xb)
            r[k] = dict(a_median=float(np.median(xa)), b_median=float(np.median(xb)), p=float(stats.wilcoxon(xa, xb).pvalue),
                        a_better=int((xa < xb).sum()))
        le = np.log(ea / eb)
        r["err_ratio"] = float(np.exp(le.mean()))
        r["tost_p"] = float(max(stats.ttest_1samp(le, -np.log(2), alternative="greater").pvalue,
                                stats.ttest_1samp(le, np.log(2), alternative="less").pvalue))
        rows[fam] = r
    for k in diffs:
        for f, q in zip(rows, holm([rows[f][k]["p"] for f in rows])):
            rows[f][k]["p_holm"] = float(q)
    pooled = {}
    for k, ds in diffs.items():
        obs = np.mean([np.median(d) for d in ds])
        null = np.array([np.mean([np.median(d * rng.choice([-1, 1], len(d))) for d in ds]) for _ in range(20000)])
        boot = np.array([np.mean([np.median(ds[i][rng.integers(0, len(ds[i]), len(ds[i]))])
                                  for i in rng.integers(0, len(ds), len(ds))]) for _ in range(20000)])
        pooled[k] = dict(diff=float(obs), p_perm=float(max((np.abs(null) >= abs(obs)).mean(), 1 / 20000)),
                         ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                         families_a_better=int(sum(np.median(d) < 0 for d in ds)), families=len(ds))
    res["pairs"][f"{a} | {b}"] = dict(families=rows, pooled=pooled)
json.dump(res, open(OUTP + ".json", "w"), indent=1)
for m, W in res["within"].items():
    print(f"== {m}")
    for f, r in W.items():
        print(f"   {f:16s} n {r['n']:3d} mem {r['members']:2d} err {r['err_mean']:7.3f} rho {r['rho_median']:.3f} "
              f"[{r['rho_ci'][0]:.3f}, {r['rho_ci'][1]:.3f}] gain {r['gain_median']:+6.1f}% (p<0 {r['p_gain_lt0']:.1e}) "
              f"q>2 {100*r['frac_q_gt2']:.0f}%")
for pk, P in res["pairs"].items():
    print(f"== pair {pk}")
    for f, r in P["families"].items():
        print(f"   {f:16s} n {r['n']} rho {np.exp(r['log_rho']['a_median']):.3f}/{np.exp(r['log_rho']['b_median']):.3f} "
              f"(p {r['log_rho']['p_holm']:.1e}) gain {100*r['gain']['a_median']:+.1f}/{100*r['gain']['b_median']:+.1f} "
              f"(p {r['gain']['p_holm']:.1e}) q {np.exp(r['log_q']['a_median']):.2f}/{np.exp(r['log_q']['b_median']):.2f} "
              f"(p {r['log_q']['p_holm']:.1e}) err x{r['err_ratio']:.2f} TOST {r['tost_p']:.2g}")
    for k, v in P["pooled"].items():
        print(f"   pooled {k}: {v}")
