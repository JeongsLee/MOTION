"""Supplementary Note 1: the two-stage pretraining, at both model sizes.

Class-averaged relative L2 in physical space -- the convention every number in the paper
is reported in -- against optimizer step, for MOTION and for the scale-reduced MOTION-S.
Both runs follow the same protocol: an amplitude stage graded by the full-field norm, then
a fluctuation stage graded by the mean-removed norm, restarted from the amplitude optimum
with a fresh cosine schedule. The dashed rule marks that restart.

Evaluations are every 2,000 steps, so the step axis is reconstructed from the evaluation
index; the stage boundaries this produces (MOTION-S at 134k, MOTION at 106k) match the run
ledger.

  python render_s1_history.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm

for _f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _p = f"/mnt/c/Windows/Fonts/{_f}"
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs")
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(HERE, "refdata", "m1m2_stages.json")))
EVERY = 2000

RUNS = [("MOTION", "m2", "#1a4f8a", 156.9), ("MOTION-S", "m1", "#d97706", 20.7)]

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 8.6, "axes.linewidth": 0.8, "savefig.dpi": 400})
fig, ax = plt.subplots(figsize=(5.0, 3.1))
fig.subplots_adjust(left=0.105, right=0.995, top=0.955, bottom=0.145)

summary = []
for name, key, col, npar in RUNS:
    stages = [st for st in D[key] if st["stage"].startswith(("S1", "S2"))]
    x, y, bound = [], [], None
    step = 0
    for si, st in enumerate(stages):
        for v in st["phys"]:
            step += EVERY
            x.append(step); y.append(v)
        if si == 0:
            bound = step
    ax.plot(x, y, "-", color=col, lw=1.4, label=f"{name}  ({npar:.1f}M)")
    ax.axvline(bound, color=col, lw=0.8, ls=(0, (4, 3)), alpha=0.55)
    b1 = stages[0]["phys_best"]; b2 = stages[1]["phys_best"]
    ax.plot([x[-1]], [b2], "o", ms=4.5, mfc=col, mec="white", mew=0.8, zorder=5)
    ax.annotate(f"{b2:.2f}\\%".replace("\\", ""), (x[-1], b2), textcoords="offset points",
                xytext=(6, -2), fontsize=8.2, color=col, va="center")
    summary.append((name, npar, bound, b1, b2, len(x)))

ax.set_yscale("log")
ax.set_ylim(6.2, 30)
ax.minorticks_off()
ax.set_yticks([7, 8, 10, 15, 20, 25])
ax.set_yticklabels(["7", "8", "10", "15", "20", "25"])
ax.set_xlim(0, 172000)
ax.set_xticks([0, 40000, 80000, 120000, 160000])
ax.set_xticklabels(["0", "40k", "80k", "120k", "160k"])
ax.set_xlabel("optimizer step")
ax.set_ylabel("class-averaged relative $L_2$ (\\%, physical)".replace("\\%", "%"))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=3, width=0.8)
ax.legend(frameon=False, fontsize=8.4, loc="upper right", handlelength=1.6)
ax.text(0.5, 0.055, "dashed rule: the amplitude stage ends and the fluctuation stage "
        "restarts", transform=ax.transAxes, ha="center", fontsize=7.6, color="0.35",
        style="italic")

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figS1_history.{ext}"), bbox_inches="tight")
print("wrote figS1_history.pdf")
for name, npar, bound, b1, b2, n in summary:
    print(f"  {name:9s} {npar:6.1f}M  amplitude ends {bound//1000}k  "
          f"best S1 {b1:.2f}  best S2 {b2:.2f}  ({n} evaluations)")
