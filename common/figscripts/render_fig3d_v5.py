"""SUPERSEDED by render_fig3d_v6.py -- do not run this file.
It writes fig3d_contour_v4.png, the SAME name render_fig3d_v4.py uses, and its 4x5 grid
silently dropped v4's 6th diagnostic column (2026-08-12). v6 = this typography + that
column, under a new output name.

Fig3d v5: v4 layout, LOCAL render — Times New Roman, mechanism row labels moved
clear of the images (wider left margin + labelpad). Reads refdata/fig3d_sel.npz
(selected samples extracted from /eu dumps; selection identical to v4)."""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
for f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    fm.fontManager.addfont(f"/mnt/c/Windows/Fonts/{f}")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs")
plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": 20, "axes.linewidth": 1.0, "savefig.dpi": 220,
    "xtick.direction": "in", "ytick.direction": "in",
})
C_OUR, C_BCAT = "#E4572E", "#2E5EAA"

ROWS = [("shallow_water", "Gravity waves"),
        ("com_ns", "Compressibility"),
        ("pdearena_ns", "Turbulent transport"),
        ("cfdbench", "Advective wake")]
cols = ["Ground truth", "MOTION", "BCAT", r"$|$MOTION $-$ GT$|$", r"$|$BCAT $-$ GT$|$"]

D = np.load(os.path.join(HERE, "refdata", "fig3d_sel.npz"))


def coherence(p, g, nb=16):
    P = np.fft.fftshift(np.fft.fft2(p - p.mean()))
    G = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    n = p.shape[0]
    ky, kx = np.indices(P.shape)
    kr = np.sqrt((kx - n / 2) ** 2 + (ky - n / 2) ** 2)
    kb = np.linspace(1, n / 2, nb + 1)
    ks, cs = [], []
    for i in range(nb):
        m = (kr >= kb[i]) & (kr < kb[i + 1])
        num = np.abs((np.conj(G[m]) * P[m]).sum())
        den = (np.abs(G[m]) * np.abs(P[m])).sum() + 1e-12
        ks.append(0.5 * (kb[i] + kb[i + 1])); cs.append(num / den)
    return np.array(ks), np.array(cs)


fig, axs = plt.subplots(4, 5, figsize=(15.5, 3.15 * 4),
                        gridspec_kw={"wspace": 0.05, "hspace": 0.10,
                                     "left": 0.065, "right": 0.99,
                                     "top": 0.955, "bottom": 0.045})
for r, (fa, lab) in enumerate(ROWS):
    gtc = D[f"{fa}_gt"]; a = D[f"{fa}_a"]; b = D[f"{fa}_b"]
    rla = float(D[f"{fa}_rla"]); rlb = float(D[f"{fa}_rlb"])
    c00 = gtc.mean(); amp = np.abs(gtc - c00).max() * 0.92
    emax = max(np.abs(a - gtc).max(), np.abs(b - gtc).max()) * 0.85 + 1e-9
    ims = [(gtc, "RdBu_r", c00 - amp, c00 + amp), (a, "RdBu_r", c00 - amp, c00 + amp),
           (b, "RdBu_r", c00 - amp, c00 + amp),
           (np.abs(a - gtc), "inferno", 0, emax), (np.abs(b - gtc), "inferno", 0, emax)]
    for c, (im, cm, lo, hi) in enumerate(ims):
        axs[r, c].imshow(im, cmap=cm, vmin=lo, vmax=hi)
        axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        for sp in axs[r, c].spines.values():
            sp.set_linewidth(1.0); sp.set_color("#555")
        if r == 0:
            axs[r, c].set_title(cols[c], fontsize=21, pad=9)
    axs[r, 0].set_ylabel(lab, fontsize=19, rotation=90, va="center", labelpad=11)
    bb = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    axs[r, 3].text(0.5, 0.06, f"{rla:.1f}%", color="k", fontsize=17, ha="center",
                   transform=axs[r, 3].transAxes, bbox=bb)
    axs[r, 4].text(0.5, 0.06, f"{rlb:.1f}%", color="k", fontsize=17, ha="center",
                   transform=axs[r, 4].transAxes, bbox=bb)


fig.savefig(f"{OUT}/fig3d_contour_v4.png", bbox_inches="tight")
print("FIG3D_DONE")
