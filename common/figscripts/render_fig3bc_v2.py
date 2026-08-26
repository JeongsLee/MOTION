"""Fig3 panel b+c, v2: sample = lowest combined error among the 4 GT-matched trajectories
(user request), spectra panel = GT signal spectrum vs ERROR spectra of both models.
Outputs -> /code-vol/motion_fv2/figs_out/fig3b_contour_v2.png, fig3c_spectra_v2.png
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

OUT = "/code-vol/motion_fv2/figs_out"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 15, "legend.fontsize": 12.5,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.6, "grid.color": "#9aa0a6",
    "legend.frameon": False, "savefig.dpi": 220,
    "axes.spines.top": True, "axes.spines.right": True,
})
C_OUR, C_BCAT = "#E4572E", "#2E5EAA"

ROWS = [("shallow_water", "shallow_water", [5], 0, 10, "Shallow Water\n($h$)", "Shallow Water ($h$)"),
        ("com_ns", "com_ns", [0, 1, 3, 4], 3, 10, "Compressible NS\n(pressure)", "Compressible NS (pressure)"),
        ("pdearena_uncond", "incom_ns_arena_u", [0, 1, 2], 2, 4, "PDEArena NS\n(uncond., smoke)", "PDEArena NS (uncond., smoke)"),
        ("cfdbench", "cfdbench", [0, 1, 2], 0, 10, "CFDBench\n($V_x$)", "CFDBench ($V_x$)")]
cols = ["ground truth", "MOTION", "BCAT", r"$|$MOTION $-$ GT$|$", r"$|$BCAT $-$ GT$|$"]


def spec1d(f):
    f = f - f.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    n = f.shape[0]
    ky, kx = np.indices(F.shape)
    kr = np.sqrt((kx - n / 2) ** 2 + (ky - n / 2) ** 2).astype(int)
    kmax = n // 2
    E = np.bincount(kr.ravel(), F.ravel(), minlength=kmax + 1)[:kmax + 1]
    cnt = np.bincount(kr.ravel(), minlength=kmax + 1)[:kmax + 1]
    return np.arange(kmax + 1), E / np.maximum(cnt, 1)


def espec(f, ref):
    d = f - ref
    d = d - d.mean() + 0  # keep DC of error? remove for consistency with spec1d
    F = np.abs(np.fft.fftshift(np.fft.fft2(d))) ** 2
    n = d.shape[0]
    ky, kx = np.indices(F.shape)
    kr = np.sqrt((kx - n / 2) ** 2 + (ky - n / 2) ** 2).astype(int)
    kmax = n // 2
    E = np.bincount(kr.ravel(), F.ravel(), minlength=kmax + 1)[:kmax + 1]
    cnt = np.bincount(kr.ravel(), minlength=kmax + 1)[:kmax + 1]
    return np.arange(kmax + 1), E / np.maximum(cnt, 1)


DATA = []
for fa, fb, slots, pc, F, lab, lab1 in ROWS:
    A = np.load(f"/eu/_bfa_{fa}.npz")
    aP = A["pred"][..., slots][:, :F]; aG = A["gt"][..., slots][:, :F]
    Bz = np.load(f"/eu/_bcatf_{fb}.npz"); nc = len(slots)
    bP = Bz["pred"][..., :nc][:, :F]; bG = Bz["gt"][..., :nc][:, :F]
    ns = min(len(aP), len(bP)); aP, aG, bP = aP[:ns], aG[:ns], bP[:ns]
    fr = F - 1

    def rl2(p, g, i):
        o = p[i, fr]; t = g[i, fr]
        return 100 * np.sqrt(((o - t) ** 2).sum()) / np.sqrt((t ** 2).sum())
    errs = [(rl2(aP, aG, i), rl2(bP, aG, i)) for i in range(ns)]
    for i, (ea, eb) in enumerate(errs):
        print(f"  {fa} sample{i}: MOTION {ea:.2f}% BCAT {eb:.2f}%", flush=True)
    s = int(np.argmin([ea + eb for ea, eb in errs]))
    print(f"{fa}: chosen sample={s} (MOTION {errs[s][0]:.2f}% BCAT {errs[s][1]:.2f}%)", flush=True)
    DATA.append((fa, slots, pc, F, fr, s, lab, lab1, aP, aG, bP, errs[s]))

# ---------------- panel b v2 ----------------
fig, axs = plt.subplots(4, 5, figsize=(14.4, 2.9 * 4))
for r, (fa, slots, pc, F, fr, s, lab, lab1, aP, aG, bP, (ea, eb)) in enumerate(DATA):
    gtc = aG[s, fr, ..., pc]; a = aP[s, fr, ..., pc]; b = bP[s, fr, ..., pc]
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
            axs[r, c].set_title(cols[c], fontsize=15, pad=8)
    axs[r, 0].set_ylabel(lab, fontsize=13.5, rotation=90, va="center")
    bb = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    axs[r, 3].text(0.5, 0.06, f"{ea:.1f}%", color="k", fontsize=12.5, ha="center",
                   transform=axs[r, 3].transAxes, bbox=bb)
    axs[r, 4].text(0.5, 0.06, f"{eb:.1f}%", color="k", fontsize=12.5, ha="center",
                   transform=axs[r, 4].transAxes, bbox=bb)
fig.tight_layout(w_pad=0.4, h_pad=0.6)
fig.savefig(f"{OUT}/fig3b_contour_v2.png", bbox_inches="tight")
print("FIG3B2_DONE", flush=True)
plt.close(fig)

# ---------------- panel c v2: GT spectrum vs error spectra ----------------
fig, axs = plt.subplots(1, 4, figsize=(15.0, 3.7))
for ax, (fa, slots, pc, F, fr, s, lab, lab1, aP, aG, bP, _) in zip(axs, DATA):
    gtc = aG[s, fr, ..., pc]; a = aP[s, fr, ..., pc]; b = bP[s, fr, ..., pc]
    k, Eg = spec1d(gtc)
    _, Ea = espec(a, gtc)
    _, Eb = espec(b, gtc)
    ax.loglog(k[1:], Eg[1:] + 1e-30, "-", color="k", lw=2.6, label="ground truth", zorder=3)
    ax.loglog(k[1:], Ea[1:] + 1e-30, "-", color=C_OUR, lw=2.0, label="MOTION error", zorder=5)
    ax.loglog(k[1:], Eb[1:] + 1e-30, "--", color=C_BCAT, lw=2.0, label="BCAT error", zorder=4)
    ax.set_title(lab1, pad=7)
    ax.set_xlabel("wavenumber $k$")
    ax.set_xlim(1, 64)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.15)
axs[0].set_ylabel("$E(k)$")
axs[0].legend(loc="lower left", handlelength=2.0)
fig.tight_layout(w_pad=1.4)
fig.savefig(f"{OUT}/fig3c_spectra_v2.png", bbox_inches="tight")
print("FIG3C2_DONE", flush=True)
