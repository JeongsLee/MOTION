"""Supplementary figure (REVISION-R1, 10-22): class-averaged test error of MOTION-NS, BCAT and PROSE-FD against training
samples seen (linear x, log y), as in the training dashboard. refdata/curves_ns.json (MOTION-NS every 5k updates x
effective batch 8; baselines every 2k updates). In-training test-set monitors, no checkpoint selection."""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
for _f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _p = f"/mnt/c/Windows/Fonts/{_f}"
    if os.path.exists(_p): fm.fontManager.addfont(_p)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.7,
                     "savefig.dpi": 400})
cv = json.load(open(os.path.join(HERE, "refdata", "curves_ns.json")))
fig, ax = plt.subplots(figsize=(4.4, 2.9))
for key, lab, c, ls in (("ours", "MOTION-NS (21.8M)", "#E4572E", "-"), ("bcat", "BCAT (156M)", "#2E5EAA", "--"),
                        ("prose", "PROSE-FD (165M)", "#8A8A8A", "-.")):
    x = np.array([t[0] for t in cv[key]], float); y = np.array([t[1] for t in cv[key]], float)
    ax.plot(x, y, ls, color=c, lw=1.4, label=lab); ax.plot([x[-1]], [y[-1]], "o", color=c, ms=3.5)
    ax.text(x[-1] * 1.01, y[-1], f"{y[-1]:.2f}", fontsize=6.5, color=c, va="center")
ax.set_yscale("log"); ax.set_xlim(0, 1.36e6)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_yticks([3, 5, 10, 20, 40])
ax.set_xlabel("training samples seen"); ax.set_ylabel("class-averaged relative $L_2$ (%)")
ax.legend(frameon=False, fontsize=7); ax.grid(alpha=0.2, lw=0.5)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
for ext in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"figS_ns_curves.{ext}"), bbox_inches="tight")
print("FIGS_NS_CURVES_DONE")
