# REVISION-R1 11-01 (user): SI table of the per-trajectory paired comparison (paired_stats.json -> tab_ft15_paired.tex).
# Columns per baseline: median of the per-trajectory error ratio MOTION-IVP / baseline, share of trajectories on which
# MOTION-IVP is more accurate, Holm-adjusted Wilcoxon p (fifteen tasks).
import json, os, math
H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "..", "..", "figures", "out", "tab_ft15_paired.tex")
S = json.load(open(f"{H}/paired_stats.json"))
def pf(p):
    if p >= 0.01: return f"${p:.2f}$"
    e = math.floor(math.log10(p)); m = p / 10 ** e
    return f"${m:.0f}\\times10^{{{e}}}$" if round(m) < 10 else f"$1\\times10^{{{e+1}}}$"
L = [r"\begin{tabular}{lrcccccc}", r"  \toprule",
     r"  & & \multicolumn{3}{c}{vs Poseidon-T} & \multicolumn{3}{c}{vs Poseidon-B} \\",
     r"  \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r"  task & $n$ & ratio & better & $p_\mathrm{Holm}$ & ratio & better & $p_\mathrm{Holm}$ \\", r"  \midrule"]
for t in S["Poseidon-T"]:
    a, b = S["Poseidon-T"][t], S["Poseidon-B"][t]
    cell = lambda r: (f"$\\mathbf{{{r['ratio']:.2f}}}$" if r["ratio"] < 1 else f"${r['ratio']:.2f}$",
                      f"${100 * r['frac_better']:.0f}\\%$", pf(r["p_holm"]))
    L.append(f"  {t} & ${a['n']}$ & " + " & ".join(cell(a)) + " & " + " & ".join(cell(b)) + r" \\")
L += [r"  \bottomrule", r"\end{tabular}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
