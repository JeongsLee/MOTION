"""Fig 4b/c — physical-equivalence readouts as paired heatmaps.

Left block  : equivariance defect eps_equiv (%), median over samples and frames,
              worst legal transform per family; sequential scale, log-spaced.
Right block : ensemble gain (%p), absolute error change of the legal-group average
              against the same run's single pass; diverging scale (blue helps).

Data: refdata/equiv_readouts.json  (assembled from the measurement jobs; every
number traceable to FIGURES.md). Rows are families/tasks, columns are models.
"""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
for f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    fm.fontManager.addfont(f"/mnt/c/Windows/Fonts/{f}")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs")
D = json.load(open(os.path.join(HERE, "refdata", "equiv_readouts.json")))

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": 17, "axes.linewidth": 0.9, "savefig.dpi": 220,
})
DIV = "—"            # em dash for illegal / not applicable


def panel(key, fname, figsize):
    blk = D[key]
    rows, models = blk["rows"], blk["models"]
    def _num(x):
        return np.nan if isinstance(x, str) else float(x)
    eps = np.array([[_num(blk["eps"][r].get(m, np.nan)) for m in models] for r in rows], float)
    gain = np.array([[_num(blk["gain"][r].get(m, np.nan)) for m in models] for r in rows], float)
    div = {(r, m) for r in rows for m in models
           if isinstance(blk["gain"][r].get(m), str) and blk["gain"][r][m] == "div"}

    fig, axs = plt.subplots(1, 2, figsize=figsize,
                            gridspec_kw={"wspace": 0.08, "left": 0.30, "right": 0.995,
                                         "top": 0.80, "bottom": 0.02})
    # ---- defect (sequential, log) ----
    ax = axs[0]
    e = np.where(np.isfinite(eps), eps, np.nan)
    lo = max(0.05, np.nanmin(e[e > 0]) * 0.8)
    im = ax.imshow(np.ma.masked_invalid(e), cmap="YlOrRd", norm=LogNorm(vmin=lo, vmax=250), aspect="auto")
    ax.set_title(r"Equivariance defect $\epsilon_{\rm equiv}$ (%)", fontsize=17, pad=9)
    for i, r in enumerate(rows):
        for j, m in enumerate(models):
            v = eps[i, j]
            if not np.isfinite(v):
                ax.text(j, i, DIV, ha="center", va="center", color="#777", fontsize=15); continue
            txt = f"{v:.2f}" if v < 10 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=14.5,
                    color="white" if v > 12 else "#222")
    # ---- gain (diverging) ----
    ax2 = axs[1]
    g = np.where(np.isfinite(gain), gain, np.nan)
    lim = max(0.7, np.nanmax(np.abs(g)))
    ax2.imshow(np.ma.masked_invalid(g), cmap="RdBu_r", aspect="auto",
               norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    ax2.set_title(r"Ensemble gain (\%p)" if False else "Ensemble gain (%p)", fontsize=17, pad=9)
    for i, r in enumerate(rows):
        for j, m in enumerate(models):
            v = gain[i, j]
            if (r, m) in div:
                ax2.text(j, i, "div.", ha="center", va="center", fontsize=14, color="#b2182b"); continue
            if not np.isfinite(v):
                ax2.text(j, i, DIV, ha="center", va="center", color="#777", fontsize=15); continue
            ax2.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=14.5, color="#222")

    for k, ax_ in enumerate(axs):
        ax_.set_xticks(range(len(models)))
        ax_.set_xticklabels(models, fontsize=15.5)
        ax_.set_yticks(range(len(rows)))
        ax_.set_yticklabels(rows if k == 0 else [""] * len(rows), fontsize=15.5)
        ax_.tick_params(length=0)
        ax_.set_xticks(np.arange(-.5, len(models), 1), minor=True)
        ax_.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax_.grid(which="minor", color="white", linewidth=1.4)
        for sp in ax_.spines.values():
            sp.set_color("#999")
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", fname)


panel("joint", "fig4b_equiv.png", (7.6, 3.5))
panel("ivp", "fig4c_equiv.png", (7.0, 3.1))
