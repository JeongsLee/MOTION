"""Fig3c — Wave-Layer anchor sweep: "refuses vs pattern-completes".

Two sub-axes, shared y scale:
  left  = official protocol (single-frame IC, u only -> ill-posed off anchor 0), full horizon
  right = repaired protocol (two-field IC [u, v0, c]), lead <= 12
Data: refdata/wave_anchor_sweep.csv (VESSL job logs 2026-07-14, see CSV header).
Style: manuscript shared style (render_fig3_v3.py lineage); MOTION #E4572E,
baseline blue #2E5EAA (palette CVD-validated).
Runs locally: python3 render_fig3c_anchor.py -> ../figs/fig3c_anchor.png
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "refdata", "wave_anchor_sweep.csv")
OUT = os.path.join(HERE, "..", "figs", "fig3d_anchor.png")

import matplotlib.font_manager as _fm
for _f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _fm.fontManager.addfont(f"/mnt/c/Windows/Fonts/{_f}")
plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": 18, "axes.labelsize": 19, "axes.titlesize": 16.5,
    "legend.fontsize": 16, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 1.1,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.6,
    "grid.color": "#9aa0a6", "legend.frameon": False, "savefig.dpi": 220,
})
C_OUR, C_POS = "#E4572E", "#2E5EAA"

D = {}
with open(CSV) as f:
    for row in csv.DictReader(r for r in f if not r.startswith("#")):
        D.setdefault((row["protocol"], row["model"]), []).append(
            (int(row["anchor"]), float(row["rel_l1_fulltraj"])))
for k in D:
    D[k].sort()

fig, axs = plt.subplots(2, 1, figsize=(3.9, 4.05), sharey=True, sharex=True)
titles = {"official": "ill-posed ($u$ only)",
          "repaired": "closed ($u,\\,v_0$ restored)"}
for ax, proto in zip(axs, ["official", "repaired"]):
    for model, c, mk, z in [("Poseidon-B", C_POS, "s", 4), ("MOTION", C_OUR, "o", 5)]:
        a, y = zip(*D[(proto, model)])
        xi = range(len(a))
        ax.plot(xi, y, "-", color=c, lw=2.8, marker=mk, ms=10.5,
                mec="white", mew=0.8, zorder=z)
    ax.set_xticks(range(5), [str(v) for v in (0, 1, 2, 4, 8)])
    ax.set_title(titles[proto], pad=4)
    ax.set_ylim(0, 50)

# the one place the official benchmark is well-posed
axs[0].annotate("released benchmark\nscores here", xy=(0.06, 17.5),
                xytext=(0.42, 4.2), fontsize=13.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
axs[1].set_xlabel("anchor frame")
fig.supylabel("median rel. $L_1$ (%)", fontsize=18, x=0.015)

# direct labels on the curves (identity never by colour alone)
axs[0].text(1.55, 42.0, "Poseidon-B", color=C_POS, fontsize=16, fontweight="bold", va="center")
axs[0].text(2.2, 22.0, "MOTION", color=C_OUR, fontsize=16, fontweight="bold", va="center")
axs[1].text(0.45, 23.0, "Poseidon-B", color=C_POS, fontsize=16, fontweight="bold", va="center")
axs[1].text(2.15, 4.0, "MOTION", color=C_OUR, fontsize=16, fontweight="bold", va="center")

# the one place the official benchmark is well-posed
axs[0].annotate("released benchmark\nscores here", xy=(0.06, 17.5),
                xytext=(0.42, 4.2), fontsize=13.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
axs[1].set_xlabel("anchor frame")
fig.supylabel("median rel. $L_1$ (%)", fontsize=18, x=0.015)

fig.tight_layout(h_pad=0.15, rect=(0.015, 0.0, 1.0, 0.995), pad=0.2)
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
