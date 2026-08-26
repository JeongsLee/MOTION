"""Fig3a (gain curve) + Fig3c v3 (2x2 error spectra, samples matched to fig3b v2).
a: class-averaged accuracy gain AG(x) = err_baseline(x)/err_MOTION(x) at equal training
   samples x, over the early-budget window 16k..200k (16k grid shared by all three logs).
c: GT signal spectrum vs |pred-GT| error spectra, 2x2 grid, lowest-error samples.
Outputs -> /code-vol/motion_fv2/figs_out/{fig3a_gain.png, fig3c_spectra_v3.png}
"""
import os, re
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FuncFormatter

OUT = "/code-vol/motion_fv2/figs_out"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 17, "axes.labelsize": 19, "axes.titlesize": 18, "legend.fontsize": 14.5,
    "xtick.labelsize": 15.5, "ytick.labelsize": 15.5,
    "axes.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.6, "grid.color": "#9aa0a6",
    "legend.frameon": False, "savefig.dpi": 220,
    "axes.spines.top": True, "axes.spines.right": True,
})
C_OUR, C_BCAT, C_PROSE = "#E4572E", "#2E5EAA", "#8A8A8A"
FAMS = ["shallow_water", "com_ns", "incom_ns", "pdearena_ns", "cfdbench", "pdearena_uncond"]
INV = {"shallow_water": "shallow_water", "com_ns": "com_ns", "incom_ns": "incom_ns",
       "incom_ns_arena": "pdearena_ns", "cfdbench": "cfdbench",
       "incom_ns_arena_u": "pdearena_uncond"}


def parse_ours(p):
    S = {}
    for L in open(p, errors="ignore"):
        m = re.search(r"\[EVAL step (\d+)\].*overall=([\d.]+)%", L)
        if m:
            S[int(m.group(1)) * 8] = float(m.group(2))
    return S


def parse_fair(p):
    ep, per = -1, {}
    for L in open(p, errors="ignore"):
        m = re.search(r"End of epoch (\d+)", L)
        if m:
            ep = int(m.group(1)); continue
        m = re.search(r"FULLTEST_RELNORM\s+(\S+)\s+n=100\s+mean=([\d.]+)%", L)
        if m and m.group(1) in INV:
            per.setdefault(ep, {}).setdefault(INV[m.group(1)], []).append(float(m.group(2)))
    S = {}
    for e in sorted(per):
        d = per[e]
        if e < 0 or not all(f in d for f in FAMS):
            continue
        S[(e + 1) * 16000] = float(np.mean([d[f][0] for f in FAMS]))
    return S


ours = parse_ours("/eu/results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa/train.log")
bcat = parse_fair("/eu/results/p1_bcat_fair/v1/train.log")
prose = parse_fair("/eu/results/p1_prose_fair/v1/train.log")
xs = sorted(x for x in set(ours) & set(bcat) & set(prose) if x <= 200000)
gb = [bcat[x] / ours[x] for x in xs]
gp = [prose[x] / ours[x] for x in xs]
for x, a, b in zip(xs, gb, gp):
    print(f"x={x} AG_bcat={a:.2f} AG_prose={b:.2f}", flush=True)

fig, ax = plt.subplots(figsize=(6.4, 6.6))
ax.axhline(1.0, color="#666", lw=1.2, ls=":", zorder=2)
ax.plot(xs, gb, "-s", color=C_BCAT, lw=2.2, ms=7, mec="white", mew=0.7,
        label="vs BCAT (matched retrain)", zorder=4)
ax.plot(xs, gp, "-.^", color=C_PROSE, lw=2.2, ms=7.5, mec="white", mew=0.7,
        label="vs PROSE-FD (matched retrain)", zorder=3)
ax.set_xscale("log")
ax.set_xticks([2e4, 5e4, 1e5, 2e5])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:g}k"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(0.9, 4.1)
ax.set_xlabel("training samples seen")
ax.set_ylabel(r"accuracy gain of MOTION ($\times$)")
ax.text(xs[0] * 1.02, 1.06, "parity", color="#666", fontsize=14)
ax.legend(loc="upper right", handlelength=2.2)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3a_gain.png", bbox_inches="tight")
print("FIG3A_GAIN_DONE", flush=True)
plt.close(fig)

print("GAIN_ONLY_DONE", flush=True)
