"""Fig4g — Allen-Cahn attractor audit: tail exceedance beyond the bistable fixed points.

The reaction term of ACE has stable fixed points at u = +-1, so a solution can
approach but never leave [-1, 1]; the ground truth saturates at exactly 1.000.
The curve is P(|u| > x) over all predicted cells (16 held-out samples x 19 frames),
log scale, so the tail past the attractor is the physics violation.
Data: refdata/ace_bistable.npz (built locally from fig3_local/preds/ACE_{motion,scot}.npz).
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
for f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    fm.fontManager.addfont(f"/mnt/c/Windows/Fonts/{f}")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOC = "/mnt/d/working/2026/neuraladaf/v6/fig3_local"
OUT = os.path.join(HERE, "..", "figs", "fig4g_bistable.png")

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": 19, "axes.labelsize": 20, "legend.fontsize": 17,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "axes.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6, "grid.color": "#9aa0a6",
    "legend.frameon": False, "savefig.dpi": 220,
})
C_OUR, C_POS, C_GT = "#E4572E", "#2E5EAA", "#333333"

dm = np.load(f"{LOC}/preds/ACE_motion.npz")
sc = dm["std"][:16][:, None, None, None, :]; mc = dm["mean"][:16][:, None, None, None, :]
P = dm["pred16"] * sc + mc; G = dm["gt16"] * sc + mc
u_m = np.abs(P[:, 1:20, ..., 2]).ravel()
u_g = np.abs(G[:, 1:20, ..., 2]).ravel()
u_s = np.abs(np.load(f"{LOC}/preds/ACE_scot.npz")["pred16"][:, :, 0]).ravel()

x = np.linspace(0.96, 1.28, 160)
def exc(u):
    return np.array([(u > t).mean() for t in x]) * 100

fig, ax = plt.subplots(figsize=(3.9, 2.45))
ax.axvline(1.0, color="#888", lw=1.3, ls=":", zorder=1)
for u, c, ls, lab, z in ((u_g, C_GT, "-", "Ground truth", 5),
                         (u_m, C_OUR, "-", "MOTION", 4),
                         (u_s, C_POS, "--", "Poseidon-B", 3)):
    y = exc(u)
    ax.plot(x, np.where(y > 0, y, np.nan), ls, color=c, lw=2.4, label=lab, zorder=z)
ax.set_yscale("log")
ax.set_xlim(0.96, 1.28)
ax.set_ylim(8e-4, 60)
ax.set_xlabel(r"$|u|$")
ax.set_ylabel("cells above $|u|$ (%)", fontsize=17)
ax.annotate("attractor", xy=(1.0, 12), xytext=(1.035, 22), fontsize=16, color="#555",
            arrowprops=dict(arrowstyle="->", color="#777", lw=1.1))
ax.legend(loc="upper right", frameon=False, fontsize=14, handlelength=1.2,
          labelspacing=0.18, handletextpad=0.4, borderaxespad=0.15)
fig.tight_layout(pad=0.25)
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
print("max |u|: GT %.3f MOTION %.3f Poseidon %.3f" % (u_g.max(), u_m.max(), u_s.max()))
for t in (1.0, 1.02, 1.05):
    print(f"  >|{t}|: GT {(u_g>t).mean()*100:.3f}%  MOTION {(u_m>t).mean()*100:.3f}%  Poseidon {(u_s>t).mean()*100:.3f}%")
