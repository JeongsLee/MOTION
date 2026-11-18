"""Supplementary Fig. 2 (REVISION-R1, 2026-11-06): knockout signatures of MOTION-S on the test split, three rows.

  a  spectral structure on buoyant turbulence: energy spectrum | phase coherence, side by side
  b  spatial structure on two compressible-Euler problems: full model + three knockouts (contours), one row each
  c  error spectrum E_err(k)/E_gt(k) of the two problems, side by side

Same data and same quantities as the previous two-panel version (render_fig1d_ko.py KO_FINAL and render_fig1e_ko.py
of the release code, sample 0, lead 5, 16 test trajectories): data 03_analyses/A31_figS2_ko/{fig1d_ko,fig1e_ko}.npz
(dump job fig1-m1f-test, MOTION-S ckpt_best).

  python render_figS2_ko.py [data_dir]   -> ../figs/figS2_ko.{pdf,png}
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "A31_figS2_ko")
OUT = os.path.join(HERE, "out")
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "figure.dpi": 300,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

# ------------------------------------------------------------------ a: spectral structure (pdearena_ns, velocity)
Zd = np.load(os.path.join(D, "fig1d_ko.npz"))


def radial_bins(n):
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    return np.sqrt(KX ** 2 + KY ** 2).astype(int)


def spec_coh(gt, pr):
    n = gt.shape[1]
    kr = radial_bins(n)
    G = np.fft.fft2(gt, axes=(1, 2)); P = np.fft.fft2(pr, axes=(1, 2))
    Pg = (np.abs(G) ** 2).sum((0, -1)); Pp = (np.abs(P) ** 2).sum((0, -1)); cr = (P * np.conj(G)).sum((0, -1))
    Eg = np.zeros(n // 2); Ep = np.zeros(n // 2); C = np.zeros(n // 2)
    for k in range(1, n // 2):
        m = kr == k
        if not m.any():
            continue
        Eg[k], Ep[k] = Pg[m].mean(), Pp[m].mean()
        C[k] = np.abs(cr[m].sum()) / (np.sqrt(Pg[m].sum() * Pp[m].sum()) + 1e-12)
    return Eg, Ep, C


fam = "pdearena_ns"
band = slice(15, 36)
g_ = Zd[f"{fam}__gt"][..., :2]
SEL = [("full", "#2a78d6", "-", "full model"), ("spectral", "#eb6834", "--", "w/o spectral nonlocal"),
       ("shock", "#3f9b57", "--", "w/o upwind transport"), ("gradvec", "#8a5cd6", "--", "w/o directional gradient")]
curves = {m: spec_coh(g_, Zd[f"{fam}__{m}"][..., :2]) for m, _, _, _ in SEL}
Eg = curves["full"][0]
kk = np.arange(1, len(Eg))

# ------------------------------------------------------------------ b, c: compressible Euler (fig1e_v3 quantities)
Ze = np.load(os.path.join(D, "fig1e_ko.npz"))
HEADS = ["shock", "advection", "compressible"]
ROWS = ["CE-RM", "CE-CRP"]
HLAB = {"shock": "w/o upwind transport", "advection": "w/o advection", "compressible": "w/o density coupling"}
FLAB = {"CE-RM": "Richtmyer–Meshkov\n(density)", "CE-CRP": "curved Riemann\n(pressure)"}
FTIT = {"CE-RM": "Richtmyer–Meshkov (density)", "CE-CRP": "curved Riemann (pressure)"}
DCH = {"CE-RM": 2, "CE-CRP": 3}
MATCH = {"CE-RM": "shock", "CE-CRP": "compressible"}
KCOL = {"full": "#2a78d6", "shock": "#eb6834", "advection": "#3f9b57", "compressible": "#8a5cd6"}


def radial(F2):
    n = F2.shape[0]
    kr = radial_bins(n)
    return np.array([F2[kr == k].mean() if (kr == k).any() else 0 for k in range(n // 2)])


cols = ["full"] + HEADS
fig = plt.figure(figsize=(7.0, 9.0))
outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.85, 1.0], hspace=0.42)

# row a
ga = outer[0].subgridspec(1, 2, wspace=0.3)
axE, axC = fig.add_subplot(ga[0]), fig.add_subplot(ga[1])
axE.loglog(kk, Eg[1:], "-", color="k", lw=1.3, label="ground truth")
for m, c, ls, lab in SEL:
    _, Ep, C = curves[m]
    axE.loglog(kk, Ep[1:], ls, color=c, lw=1.2, label=lab)
    axC.semilogx(kk, C[1:], ls, color=c, lw=1.2, label=lab)
for ax in (axE, axC):
    ax.axvspan(band.start, band.stop - 1, color="#dfe3e8", alpha=0.65, zorder=0)
    ax.set_xlabel("wavenumber $k$")
lo = min(np.min(v[1][1:]) for v in curves.values()); hi = max(np.max(v[1][1:]) for v in curves.values())
axE.set_ylim(lo * 0.5, hi * 2)
axE.set_ylabel("energy $E(k)$")
axE.legend(fontsize=7, loc="lower left", frameon=False)
axC.set_ylim(-0.05, 1.05)
axC.set_ylabel("phase coherence $\\gamma(k)$")

# row b
gb = outer[1].subgridspec(2, 4, wspace=0.05, hspace=0.12)
for r, fam in enumerate(ROWS):
    ch = DCH[fam]
    gt = Ze[f"{fam}__gt"][0][..., ch]
    v0, v1 = np.percentile(gt, 1), np.percentile(gt, 99)
    for c, kname in enumerate(cols):
        ax = fig.add_subplot(gb[r, c])
        f = Ze[f"{fam}__{kname}"][0][..., ch]
        ax.imshow(f.T, origin="lower", cmap="viridis", vmin=v0, vmax=v1)
        rel = np.linalg.norm(f - gt) / (np.linalg.norm(gt) + 1e-12)
        ax.text(0.03, 0.03, f"{100 * rel:.1f}%", transform=ax.transAxes, fontsize=7.5, color="w", ha="left",
                va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, ec="none"))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("full model" if kname == "full" else HLAB[kname], fontsize=8.5)
        if c == 0:
            ax.set_ylabel(FLAB[fam], fontsize=8)
        if kname == MATCH[fam]:
            for sp in ax.spines.values():
                sp.set_edgecolor("#d43d2a"); sp.set_linewidth(1.8)

# row c
gc = outer[2].subgridspec(1, 2, wspace=0.3)
for r, fam in enumerate(ROWS):
    ax = fig.add_subplot(gc[r])
    ch = DCH[fam]
    gt = Ze[f"{fam}__gt"][0][..., ch]
    Egt = radial(np.abs(np.fft.fft2(gt)) ** 2)
    k = np.arange(1, len(Egt))
    for kname in cols:
        f = Ze[f"{fam}__{kname}"][0][..., ch]
        Ee = radial(np.abs(np.fft.fft2(f - gt)) ** 2)
        ax.loglog(k, (Ee / (Egt + 1e-12))[1:], "-" if kname == "full" else "--", color=KCOL[kname],
                  lw=1.4 if kname == MATCH[fam] else 1.0, label="full model" if kname == "full" else HLAB[kname])
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
    ax.set_title(FTIT[fam], fontsize=8.5)
    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel("$E_{\\mathrm{err}}(k)/E_{\\mathrm{gt}}(k)$")
    if r == 0:
        ax.legend(fontsize=6.8, loc="lower right", frameon=False)

FIRST = {}
for ax in fig.axes:
    FIRST.setdefault(round(ax.get_position().y1, 3), ax)
tops = sorted({round(ax.get_position().y1, 2) for ax in fig.axes}, reverse=True)
x0 = min(ax.get_position().x0 for ax in fig.axes) - 0.07
row_top = {"a": axE.get_position().y1, "b": max(a.get_position().y1 for a in fig.axes if a.images),
           "c": max(a.get_position().y1 for a in fig.axes[-2:])}
for lab, y in row_top.items():
    fig.text(x0, y + 0.025, lab, fontsize=10, fontweight="bold", va="bottom")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figS2_ko.{ext}"))
print("written", os.path.join(OUT, "figS2_ko.pdf"))
