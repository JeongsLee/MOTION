# REVISION-R1 10-27: Fig. 4b (Poseidon downstream, six task groups) and the SI 15-task table, generated from one
# data block so that finished FT runs only change the numbers below.
# Metric: physical relative L1, full trajectory, median over the official test trajectories (%).
# Group value = geometric mean of the task errors in the group (task errors span 0.6-50%).
# RULE "B": MOTION-IVP hop-1 AR on every time-dependent task (its pretraining mode), relaxation (K=1) on steady
#           tasks; Poseidon-T in the rollout of Herde et al. (2024), Table 6.
# RULE "flow": both models AR on flow tasks, direct / single-shot on ACE and the wave tasks, relaxation on steady.
# None = not yet available; PROV holds the stand-in printed in red until the run finishes.
import math, os, sys

RULE = sys.argv[1] if len(sys.argv) > 1 else "B"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")

TASKS = ["NS-PwC", "NS-SVS", "NS-BB", "NS-SL", "NS-Tracer-PwC", "FNS-KF", "CE-RPUI", "CE-RM", "GCE-RT",
         "Wave-Gauss", "Wave-Layer", "ACE", "Poisson-Gauss", "Helmholtz", "SE-AF"]
GROUPS = [("NS", TASKS[0:6]), ("CE", TASKS[6:9]), ("Wave", TASKS[9:11]), ("ACE", ["ACE"]),
          ("Elliptic", ["Poisson-Gauss", "Helmholtz"]), ("SE-AF", ["SE-AF"])]
FLOW = set(TASKS[0:9]); STEADY = {"Poisson-Gauss", "Helmholtz", "SE-AF"}
# Poseidon Table 6 rollout (AR on these, direct on the other time-dependent tasks)
T6_AR = {"NS-PwC", "NS-SVS", "NS-BB", "NS-SL", "NS-Tracer-PwC", "CE-RPUI"}

# MOTION-IVP (stage-3 C init, 3,200 updates; CE-RPUI / CE-RM with the energy channel as in the pretraining (5 targets,
# 4 scored), user 10-27)
M_AR = {"NS-PwC": 2.18, "NS-SVS": 2.42, "NS-BB": 1.28, "NS-SL": 2.63, "NS-Tracer-PwC": 4.47,
        "FNS-KF": 2.23, "CE-RPUI": 6.74, "CE-RM": 7.77, "GCE-RT": 2.66,
        "Wave-Gauss": 27.16, "Wave-Layer": 25.59, "ACE": 0.91}
M_SS = {"FNS-KF": 13.84, "CE-RM": 19.01, "GCE-RT": 1.61, "Wave-Gauss": 50.08, "Wave-Layer": 50.37, "ACE": 0.63}
M_STEADY = {"Poisson-Gauss": 6.76, "Helmholtz": 32.04, "SE-AF": 1.15}
# stand-ins until the AR / 4-channel runs finish (val track or the 5-channel run)
PROV = {"FNS-KF": 2.27, "CE-RPUI": 6.74, "CE-RM": 7.77, "GCE-RT": 1.61, "Wave-Gauss": 50.08,
        "Wave-Layer": 50.37, "ACE": 1.44}
# Poseidon-T (our FT15 runs), direct / AR
P_DIR = {"NS-PwC": 2.73, "NS-SVS": 2.45, "NS-BB": 1.81, "NS-SL": 4.93, "NS-Tracer-PwC": 9.42, "FNS-KF": 7.03,
         "CE-RPUI": 8.41, "CE-RM": 13.02, "GCE-RT": 2.27, "Wave-Gauss": 32.17, "Wave-Layer": 23.18, "ACE": 1.47}
# REVISION-R1 11-01: NS-SVS 2.44 -> 2.43 (median 2.4347); B_DIR Wave-Gauss 25.775 -> 25.7753 (displayed 25.78)
P_AR = {"NS-PwC": 1.71, "NS-SVS": 2.43, "NS-BB": 1.51, "NS-SL": 2.75, "NS-Tracer-PwC": 11.35, "FNS-KF": 12.94,
        "CE-RPUI": 7.20, "CE-RM": 7.21, "GCE-RT": 1.47, "Wave-Gauss": 64.95, "Wave-Layer": 54.43, "ACE": 5.69}
P_STEADY = {"Poisson-Gauss": 8.01, "Helmholtz": 33.27, "SE-AF": 1.25}
# REVISION-R1 10-29 (user): Poseidon-B (our FT15 runs, same recipe as Poseidon-T), direct / AR -- SI table only
B_DIR = {"NS-PwC": 1.636, "NS-SVS": 1.882, "NS-BB": 1.357, "NS-SL": 3.844, "NS-Tracer-PwC": 9.649, "FNS-KF": 7.422,
         "CE-RPUI": 7.686, "CE-RM": 17.847, "GCE-RT": 2.147, "Wave-Gauss": 25.7753, "Wave-Layer": 19.554, "ACE": 1.060}
B_AR = {"NS-PwC": 1.113, "NS-SVS": 1.216, "NS-BB": 1.079, "NS-SL": 1.547, "NS-Tracer-PwC": 9.959, "FNS-KF": 11.251,
        "CE-RPUI": 5.730, "CE-RM": 6.446, "GCE-RT": 1.143, "Wave-Gauss": 61.205, "Wave-Layer": 58.314, "ACE": 3.308}
B_STEADY = {"Poisson-Gauss": 11.191, "Helmholtz": 35.377, "SE-AF": 1.446}


def motion(t):
    if t in STEADY: return M_STEADY[t], False
    use_ar = RULE == "B" or t in FLOW
    v = M_AR.get(t) if use_ar else M_SS.get(t)
    return (v, False) if v is not None else (PROV[t], True)


def poseidon(t):
    if t in STEADY: return P_STEADY[t]
    ar = (t in T6_AR) if RULE == "B" else (t in FLOW)
    return P_AR[t] if ar else P_DIR[t]


def poseidon_b(t):
    if t in STEADY: return B_STEADY[t]
    ar = (t in T6_AR) if RULE == "B" else (t in FLOW)
    return B_AR[t] if ar else B_DIR[t]


def fmt(v, prov):
    s = f"${v:.2f}$"
    return r"\textcolor{red}{" + s + "}" if prov else s


def gmean(xs): return math.exp(sum(math.log(x) for x in xs) / len(xs))


# REVISION-R1 10-27 (user): main Fig. 4b = in-distribution (operators of the pretraining corpus; Herde et al. 2024) vs
# out-of-distribution tasks, geometric means, error ratio (geometric mean of MOTION-IVP / Poseidon-T per task) and wins
ID = ["NS-PwC", "NS-SVS", "NS-BB", "NS-SL", "CE-RPUI", "CE-RM"]
OOD = [t for t in TASKS if t not in ID]
brow = []
for nm, ts in (("ID", ID), ("OOD", OOD), ("All", TASKS)):
    mv = [motion(t) for t in ts]; pv = [poseidon(t) for t in ts]
    prov = any(p for _, p in mv)
    ratio = gmean([v / q for (v, _), q in zip(mv, pv)]); w = sum(v < q for (v, _), q in zip(mv, pv))
    if nm == "All": brow.append(r"  \midrule")
    brow.append(f"  {nm} & ${len(ts)}$ & {fmt(gmean([v for v, _ in mv]), prov)} & ${gmean(pv):.2f}$ & ${ratio:.2f}$ & ${w}/{len(ts) - w}$ \\\\")
tabb = "\n".join([r"\begin{tabular}{lccccc}", r"  \toprule",
                  r"  \multirow{2}{*}{tasks} & \multirow{2}{*}{$n$} & MOTION-IVP & Poseidon-T & \multirow{2}{*}{ratio} & \multirow{2}{*}{W/L} \\",
                  r"   & & (25.6M) & (21M) & & \\", r"  \midrule"] + brow + [r"  \bottomrule", r"\end{tabular}"])
open(os.path.join(OUT, "tab_fig4b_idood.tex"), "w").write(tabb + "\n")

rows = []
for g, ts in GROUPS:
    mv = [motion(t) for t in ts]; pv = [poseidon(t) for t in ts]
    gm, gp = gmean([v for v, _ in mv]), gmean(pv)
    prov = any(p for _, p in mv)
    rows.append(f"  {g} ({len(ts)}) & {fmt(gm, prov)} & ${gp:.2f}$ \\\\")
allm = [motion(t) for t in TASKS]
rows.append(r"  \midrule")
rows.append(f"  All (15) & {fmt(gmean([v for v, _ in allm]), any(p for _, p in allm))} & "
            f"${gmean([poseidon(t) for t in TASKS]):.2f}$ \\\\")
tab = "\n".join([r"\begin{tabular}{lcc}", r"  \toprule",
                 r"  \multirow{2}{*}{group ($n$)} & MOTION-IVP & Poseidon-T \\", r"   & (25.6M) & (21M) \\", r"  \midrule"] + rows +
                [r"  \bottomrule", r"\end{tabular}"])
open(os.path.join(OUT, "tab_fig4b.tex"), "w").write(tab + "\n")

# SI: all fifteen tasks (REVISION-R1 10-29: + Poseidon-B column and the mean row)
si = [r"\begin{tabular}{llcccc}", r"  \toprule", r"  group & task & ID/OOD & MOTION-IVP & Poseidon-T & Poseidon-B \\",
      r"   & & & (25.6M) & (21M) & (158M) \\", r"  \midrule"]
for g, ts in GROUPS:
    for i, t in enumerate(ts):
        v, p = motion(t)
        si.append(f"  {g if i == 0 else ''} & {t} & {'ID' if t in ID else 'OOD'} & {fmt(v, p)} & ${poseidon(t):.2f}$ & ${poseidon_b(t):.2f}$ \\\\")
si += [r"  \midrule", f"  Mean (15) & & & ${sum(motion(t)[0] for t in TASKS) / 15:.2f}$ & ${sum(poseidon(t) for t in TASKS) / 15:.2f}$ & "
       f"${sum(poseidon_b(t) for t in TASKS) / 15:.2f}$ \\\\"]
si += [r"  \bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_ft15_all.tex"), "w").write("\n".join(si) + "\n")
print(tab)

# REVISION-R1 10-29 (user): main Fig. 4b in the format of Fig. 4a: six task groups, ARITHMETIC mean of the task errors
# per group and over all fifteen tasks (Fig. 4a also switches to the arithmetic mean).
am = lambda xs: sum(xs) / len(xs)
rows = []
for g, ts in GROUPS:   # REVISION-R1 10-29 (user): + Poseidon-B column
    mv = [motion(t) for t in ts]; pv = [poseidon(t) for t in ts]; bv = [poseidon_b(t) for t in ts]
    rows.append(f"  {g} ({len(ts)}) & {fmt(am([v for v, _ in mv]), any(p for _, p in mv))} & ${am(pv):.2f}$ & ${am(bv):.2f}$ \\\\")
allm = [motion(t) for t in TASKS]
rows += [r"  \midrule", f"  Mean (15) & {fmt(am([v for v, _ in allm]), any(p for _, p in allm))} & "
         f"${am([poseidon(t) for t in TASKS]):.2f}$ & ${am([poseidon_b(t) for t in TASKS]):.2f}$ \\\\"]
tab = "\n".join([r"\begin{tabular}{lccc}", r"  \toprule", r"  Task group & MOTION-IVP & Poseidon-T & Poseidon-B \\", r"  (tasks) & (25.6M) & (21M) & (158M) \\",
                 r"  \midrule"] + rows + [r"  \bottomrule", r"\end{tabular}"])
open(os.path.join(OUT, "tab_fig4b_arith.tex"), "w").write(tab + "\n")
print(tab)
