"""Render Fig-1e from fig1e_ko.npz: CE-only knockout selectivity grid + error-spectrum
diagnostic column (user design 08-11: incompressible story lives in Fig-1d).

  python render_fig1e_ko.py /code-vol/motion_fv2/figs_m1s2b

Rows = CE families (R-M density / curved-Riemann pressure), columns = full model +
three knockouts (contours, matched cell framed, per-panel rel-L2) + a final
"where the error lives" panel: radially binned E_err(k)/E_gt(k) per knockout --
the R-M row separates only for -shock (mid-k, the sharp front), the CRP row only
for -compressibility (low-k, the smooth pressure structures).
Env: KO_ROWS=CE-RM,CE-CRP  KO_SAMPLE="CE-RM:0,CE-CRP:0"
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/code-vol/motion_fv2/figs_m1s2b"
Z = np.load(os.path.join(D, "fig1e_ko.npz"))
M = json.load(open(os.path.join(D, "fig1e_ko_meta.json")))
HEADS = [h for h in M["heads"] if h in ("shock", "advection", "compressible")]
ROWS = [f for f in os.environ.get("KO_ROWS", "CE-RM,CE-CRP").split(",") if f in M["families"]]

plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.titlesize": 9.5,
                     "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

HLAB = {"shock": "w/o shock (upwind)", "advection": "w/o advection",
        "compressible": "w/o density coupling"}
FLAB = {"CE-RM": "CE Richtmyer–Meshkov\n(density)", "CE-CRP": "CE curved Riemann\n(pressure)"}
# display channel in the compact active-channel array ([vx,vy,rho,p,E]); chosen by the
# 08-11 channel sweep: RM shock damage lives on density, CRP compressibility damage on
# pressure (density there is shock-confounded).
DCH = {"CE-RM": 2, "CE-CRP": 3}
MATCH = {"CE-RM": "shock", "CE-CRP": "compressible"}
KCOL = {"full": "#2a78d6", "shock": "#eb6834", "advection": "#3f9b57",
        "compressible": "#8a5cd6"}

SAMP = {}
for tok in os.environ.get("KO_SAMPLE", "").split(","):
    if ":" in tok:
        f, i = tok.rsplit(":", 1)
        SAMP[f] = int(i)


def radial(F2):
    n = F2.shape[0]
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    kr = np.sqrt(KX ** 2 + KY ** 2).astype(int)
    return np.array([F2[kr == k].mean() if (kr == k).any() else 0 for k in range(n // 2)])


cols = ["full"] + HEADS
NC = len(cols) + 1                                    # + diagnostic column
fig, axes = plt.subplots(len(ROWS), NC, figsize=(1.85 * NC, 1.95 * len(ROWS)),
                         gridspec_kw={"hspace": 0.18, "wspace": 0.08,
                                      "width_ratios": [1] * len(cols) + [1.25]})
for r, fam in enumerate(ROWS):
    s = SAMP.get(fam, 0)
    ch = DCH[fam]
    gt = Z[f"{fam}__gt"][s][..., ch]
    v0, v1 = np.percentile(gt, 1), np.percentile(gt, 99)
    Eg = radial(np.abs(np.fft.fft2(gt)) ** 2)
    k = np.arange(1, len(Eg))
    for c, kname in enumerate(cols):
        ax = axes[r, c]
        f = Z[f"{fam}__{kname}"][s][..., ch]
        ax.imshow(f.T, origin="lower", cmap="viridis", vmin=v0, vmax=v1)
        rel = np.linalg.norm(f - gt) / (np.linalg.norm(gt) + 1e-12)
        ax.text(0.03, 0.03, f"{100*rel:.1f}%", transform=ax.transAxes, fontsize=7.5,
                color="w", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, ec="none"))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("full model" if kname == "full" else HLAB.get(kname, kname),
                         fontsize=9)
        if c == 0:
            ax.set_ylabel(FLAB.get(fam, fam), fontsize=8)
        if kname == MATCH.get(fam):
            for sp in ax.spines.values():
                sp.set_edgecolor("#d43d2a"); sp.set_linewidth(1.8)
    # diagnostic column: error-spectrum ratio (where the lost information lived)
    ax = axes[r, len(cols)]
    for kname in cols:
        f = Z[f"{fam}__{kname}"][s][..., ch]
        Ee = radial(np.abs(np.fft.fft2(f - gt)) ** 2)
        ax.loglog(k, (Ee / (Eg + 1e-12))[1:], "-" if kname == "full" else "--",
                  color=KCOL[kname], lw=1.3 if kname == MATCH.get(fam) else 1.0,
                  label="full" if kname == "full" else HLAB[kname])
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
    ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
    ax.set_ylabel("$E_{\\mathrm{err}}(k)/E_{\\mathrm{gt}}(k)$", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    if r == 0:
        pass   # no title: the caption already says what this column shows
        ax.legend(fontsize=5.6, loc="lower right", frameon=False, borderpad=0.1,
                  handlelength=1.4)
    if r == len(ROWS) - 1:
        ax.set_xlabel("wavenumber $k$", fontsize=8)
fig.savefig(os.path.join(D, "fig1e_v3.png"))
fig.savefig(os.path.join(D, "fig1e_v3.pdf"))                      # vector axes + labels
plt.close(fig)
print("[fig1e] fig1e_v3.png written", flush=True)
