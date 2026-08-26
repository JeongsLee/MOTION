"""Supplementary Note 2, rebuilt on three axes instead of one.

The point the note has to support is not "MOTION is better" -- it is *which instrument
separates the two operators*, because the three do not agree:

  signal energy E_pred(k)/E_GT(k) : does NOT separate. The baseline sits at or above the
                                    ground truth at every band on three of four cases
                                    (1.53x on the wave); the deficit is OURS.
  error power                     : decisive on the wave and the wake (up to ~2 decades),
                                    nearly blind on the two turbulent cases.
  phase coherence                 : decisive on exactly those two (2x the decorrelated
                                    fraction on the smoke; 8.6x at k=3 on the pressure).

So the figure is 4 cases x 3 axes, and the caption says which column decides which row.
Local render off refdata/fig3d_sel.npz -- same worst-case samples and frame as Fig. 2d.
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
OUT = os.path.join(HERE, "..", "figs")
plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": 13, "axes.linewidth": 0.9, "savefig.dpi": 240,
    "xtick.direction": "in", "ytick.direction": "in",
})
C_OUR, C_BCAT, C_GT = "#E4572E", "#2E5EAA", "#222222"
NB = 16                                    # same binning as Fig. 2d's diagnostic column

ROWS = [("shallow_water", "Gravity waves"),
        ("com_ns", "Compressibility"),
        ("pdearena_ns", "Turbulent transport"),
        ("cfdbench", "Advective wake")]

D = np.load(os.path.join(HERE, "refdata", "fig3d_sel.npz"))


def _rad(f):
    F = np.fft.fftshift(np.fft.fft2(f - f.mean()))
    n = f.shape[0]
    iy, ix = np.indices(F.shape)
    kr = np.sqrt((ix - n / 2) ** 2 + (iy - n / 2) ** 2)
    kb = np.linspace(1, n / 2, NB + 1)
    return F, kr, kb


def power(f):
    F, kr, kb = _rad(f)
    P = np.abs(F) ** 2
    ks = np.array([0.5 * (kb[i] + kb[i + 1]) for i in range(NB)])
    es = np.array([P[(kr >= kb[i]) & (kr < kb[i + 1])].mean() for i in range(NB)])
    return ks, es


def coherence(p, g):
    P, kr, kb = _rad(p)
    G, _, _ = _rad(g)
    ks, cs = [], []
    for i in range(NB):
        m = (kr >= kb[i]) & (kr < kb[i + 1])
        ks.append(0.5 * (kb[i] + kb[i + 1]))
        cs.append(abs((np.conj(G[m]) * P[m]).sum()) / ((abs(G[m]) * abs(P[m])).sum() + 1e-12))
    return np.array(ks), np.array(cs)


TITLES = [r"signal energy  $E_{\rm pred}(k)/E_{\rm GT}(k)$",
          r"error power  $E_{\rm err}(k)$",
          r"phase coherence  $c(k)$"]

fig, axs = plt.subplots(4, 3, figsize=(8.7, 1.92 * 4),
                        gridspec_kw={"wspace": 0.30, "hspace": 0.28,
                                     "left": 0.085, "right": 0.985,
                                     "top": 0.945, "bottom": 0.055})
LOG = []
for r, (fa, lab) in enumerate(ROWS):
    g = D[f"{fa}_gt"]; a = D[f"{fa}_a"]; b = D[f"{fa}_b"]
    ks, eg = power(g); _, ea = power(a); _, eb = power(b)
    _, pa = power(a - g); _, pb = power(b - g)
    _, ca = coherence(a, g); _, cb = coherence(b, g)

    # --- column 1: signal energy, relative to the ground truth -------------------
    ax = axs[r, 0]
    ax.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.55)
    ax.plot(ks, ea / eg, color=C_OUR, lw=2.0, label="MOTION")
    ax.plot(ks, eb / eg, color=C_BCAT, ls="--", lw=2.0, label="BCAT")
    ax.set_yscale("log"); ax.set_ylim(2e-2, 4)
    ax.set_ylabel(lab, fontsize=14, labelpad=8)

    # --- column 2: error power, against the ground-truth signal ------------------
    ax = axs[r, 1]
    ax.plot(ks, eg, color=C_GT, lw=1.4, alpha=0.55, label="GT signal")
    ax.plot(ks, pa, color=C_OUR, lw=2.0, label="MOTION error")
    ax.plot(ks, pb, color=C_BCAT, ls="--", lw=2.0, label="BCAT error")
    ax.set_yscale("log")

    # --- column 3: phase coherence ----------------------------------------------
    ax = axs[r, 2]
    ax.plot(ks, ca, color=C_OUR, lw=2.0, label="MOTION")
    ax.plot(ks, cb, color=C_BCAT, ls="--", lw=2.0, label="BCAT")
    ax.set_ylim(0, 1.06)

    for c in range(3):
        ax = axs[r, c]
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 40, 64])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
        ax.grid(alpha=0.18, lw=0.5)
        ax.tick_params(labelsize=11)
        if r == 0:
            ax.set_title(TITLES[c], fontsize=14, pad=8)
            ax.legend(fontsize=10.5, frameon=False,
                      loc="lower left" if c == 2 else "best")
        if r == 3:
            ax.set_xlabel("wavenumber $k$", fontsize=13)

    hi = ks >= 20
    LOG.append((lab,
                float((eb / eg).max()), float((ea / eg)[hi].mean()), float((eb / eg)[hi].mean()),
                float((pb / pa).max()), float(((1 - cb) / np.maximum(1 - ca, 1e-9)).max())))

fig.savefig(f"{OUT}/fig_si_amp_err_phase.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_si_amp_err_phase.pdf", bbox_inches="tight")
print("%-22s %8s %9s %9s %10s %10s" % ("case", "B Emax", "M E k>=20", "B E k>=20",
                                       "errP B/M", "(1-c) B/M"))
for row in LOG:
    print("%-22s %8.2f %9.2f %9.2f %10.1f %10.2f" % row)
print("SI_AMP_ERR_PHASE_DONE")
