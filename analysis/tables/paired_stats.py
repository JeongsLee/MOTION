# REVISION-R1 10-30 (user): paired per-trajectory comparison of the fine-tuned MOTION-IVP with Poseidon-T / Poseidon-B on
# the fifteen downstream tasks (official test split; same rows in the same order in both scorers).
# Inputs: <task>.json from run_ft15_rescore.sh (MOTION-IVP), poseidon/<model>/<task>/score_test_{ar,direct}.json.
# Rule B of make_fig4b.py: MOTION-IVP hop-1 AR on time-dependent tasks, relaxation on steady ones; Poseidon in the
# rollout of Herde et al. (2024) Table 6. Per task: Wilcoxon signed-rank on the paired log errors, Holm over 15 tasks.
import json, os, sys
import numpy as np
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__))
TASKS = ["NS-PwC", "NS-SVS", "NS-BB", "NS-SL", "NS-Tracer-PwC", "FNS-KF", "CE-RPUI", "CE-RM", "GCE-RT",
         "Wave-Gauss", "Wave-Layer", "ACE", "Poisson-Gauss", "Helmholtz", "SE-AF"]
STEADY = {"Poisson-Gauss", "Helmholtz", "SE-AF"}
T6_AR = {"NS-PwC", "NS-SVS", "NS-BB", "NS-SL", "NS-Tracer-PwC", "CE-RPUI"}


def pos(model, t):
    mode = "ar" if t in T6_AR else "direct"
    return np.array(json.load(open(f"{H}/poseidon/{model}/{t}/score_test_{mode}.json"))["per_traj_fulltraj"])


def mot(t):
    f = f"{H}/{t}.json"
    if not os.path.exists(f): return None
    d = json.load(open(f))
    return np.array(d["direct_fulltraj"] if t in STEADY else d["ar_fulltraj"])


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj


out = {}
for model in ("Poseidon-T", "Poseidon-B"):
    rows = []
    for t in TASKS:
        P = pos(model, t); M = mot(t)
        if M is None: rows.append((t, None)); continue
        P, M = 100 * P, 100 * M                                   # fractions -> percent
        assert len(M) == len(P), (t, len(M), len(P))
        d = np.log(M) - np.log(P)
        w = stats.wilcoxon(d)
        rows.append((t, dict(n=len(M), med_m=float(np.median(M)), med_p=float(np.median(P)),
                             ratio=float(np.exp(np.median(d))), frac_better=float((d < 0).mean()),
                             p=float(w.pvalue), rho=float(stats.spearmanr(M, P)[0]))))
    ok = [r for r in rows if r[1]]
    adj = holm([r[1]["p"] for r in ok])
    for (t, r), a in zip(ok, adj): r["p_holm"] = float(a)
    print(f"\n== MOTION-IVP vs {model}")
    print(f"{'task':15s} {'n':>4s} {'MOT':>7s} {'POS':>7s} {'ratio':>6s} {'%better':>7s} {'p':>9s} {'p_holm':>9s} {'rho':>5s}")
    for t, r in rows:
        if r is None: print(f"{t:15s}  (missing)"); continue
        print(f"{t:15s} {r['n']:4d} {r['med_m']:7.2f} {r['med_p']:7.2f} {r['ratio']:6.2f} {100*r['frac_better']:6.0f}% "
              f"{r['p']:9.2e} {r['p_holm']:9.2e} {r['rho']:5.2f}")
    sig_w = sum(1 for _, r in ok if r["p_holm"] < 0.05 and r["ratio"] < 1)
    sig_l = sum(1 for _, r in ok if r["p_holm"] < 0.05 and r["ratio"] > 1)
    print(f"significant (Holm 0.05): MOTION-IVP better {sig_w}, worse {sig_l}, n.s. {len(ok) - sig_w - sig_l}")
    out[model] = dict(rows)
json.dump(out, open(f"{H}/paired_stats.json", "w"), indent=1)
