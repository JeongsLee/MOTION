"""Fig3 a/b — Times New Roman, compact height, repositioned legends.
Reads figscripts/refdata/curves.json (extracted from /eu train logs)."""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
for f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    fm.fontManager.addfont(f"/mnt/c/Windows/Fonts/{f}")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs")

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": 24, "axes.labelsize": 28, "axes.titlesize": 25, "legend.fontsize": 19,
    "xtick.labelsize": 22, "ytick.labelsize": 22,
    "axes.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.6, "grid.color": "#9aa0a6",
    "legend.frameon": False, "savefig.dpi": 220,
})
C_OUR, C_BCAT, C_PROSE = "#E4572E", "#2E5EAA", "#8A8A8A"

cv = json.load(open(os.path.join(HERE, "refdata", "curves.json")))
ours = [tuple(t) for t in cv["ours"]]; bcat = [tuple(t) for t in cv["bcat"]]; prose = [tuple(t) for t in cv["prose"]]

# ---------------- panel a ----------------
xo = np.array([s[0] for s in ours], float); yo = np.array([s[1] for s in ours], float)
CROSS = {}
for nm, S in (("BCAT", bcat), ("PROSE", prose)):
    tgt = S[-1][1]
    k = int(np.argmax(yo <= tgt))
    xat = float(np.exp(np.interp(np.log(tgt), np.log([yo[k], yo[k - 1]]), np.log([xo[k], xo[k - 1]])))) \
        if k > 0 and yo[k - 1] > tgt else float(xo[k])
    CROSS[nm] = (xat, tgt)

fig, ax = plt.subplots(figsize=(5.6, 5.0))
for S, c, ls, mk, lab, z in [(ours, C_OUR, "-", "o", "MOTION", 5),
                             (bcat, C_BCAT, "--", "s", "BCAT", 4),
                             (prose, C_PROSE, "-.", "^", "PROSE-FD", 3)]:
    x = [s[0] for s in S]; y = [s[1] for s in S]
    ax.plot(x, y, ls, color=c, lw=2.3, label=lab, zorder=z)
    ax.plot([x[-1]], [y[-1]], marker=mk, color=c, ms=8.5, mec="white", mew=0.8, zorder=6)
for nm, c in (("BCAT", C_BCAT), ("PROSE", C_PROSE)):
    xat, tgt = CROSS[nm]
    ax.plot([xat, 1.28e6], [tgt, tgt], ":", color=c, lw=1.4, zorder=2)
    ax.plot([xat], [tgt], marker="|", color=c, ms=13, mew=2.2, zorder=6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([2e4, 1e5, 1e6])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v in (3, 5) else ""))
ax.set_xlabel("Training samples seen")
ax.set_ylabel("Class-averaged\nrel-$L_2$ (%)")
ax.legend(loc="lower left", handlelength=1.1, fontsize=19.5, labelspacing=0.22,
          handletextpad=0.4, borderaxespad=0.08, borderpad=0.12)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3a_overall.png", bbox_inches="tight")
plt.close(fig)
print("FIG3A_DONE")

# ---------------- panel b ----------------
do = dict(ours); db = dict(bcat); dp = dict(prose)
xs = sorted(x for x in set(do) & set(db) & set(dp) if x <= 200000)
gb = [db[x] / do[x] for x in xs]
gp = [dp[x] / do[x] for x in xs]

fig, ax = plt.subplots(figsize=(5.6, 5.0))
ax.axhline(1.0, color="#666", lw=1.2, ls=":", zorder=2)
ax.plot(xs, gb, "-s", color=C_BCAT, lw=2.2, ms=7, mec="white", mew=0.7,
        label="vs BCAT", zorder=4)
ax.plot(xs, gp, "-.^", color=C_PROSE, lw=2.2, ms=7.5, mec="white", mew=0.7,
        label="vs PROSE-FD", zorder=3)
ax.set_xscale("log")
ax.set_xticks([2e4, 1e5, 2e5])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:g}k"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(0.9, 4.1)
ax.set_xlabel("Training samples seen")
ax.set_ylabel("Accuracy gain")
ax.text(xs[-1], 1.08, "parity", color="#666", fontsize=19, ha="right", va="bottom")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), handlelength=1.1,
          fontsize=19.5, labelspacing=0.22, handletextpad=0.4, borderpad=0.12)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3a_gain.png", bbox_inches="tight")
plt.close(fig)
print("FIG3B_DONE")
