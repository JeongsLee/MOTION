"""Fig3d v6 = v5 typography (Times New Roman, row labels clear of the images) WITH the
per-case physics diagnostic column that v5 dropped when it overwrote v4's output file.

The sixth column is v4's, unchanged in substance -- it quantifies HOW each model's error
is physical or not, one diagnostic per mechanism:
  gravity waves   -> error rms vs radius from the wavefront centre (BCAT peaks AT the front)
  compressibility -> spectral phase coherence vs wavenumber (MOTION holds phase)
  turbulence      -> spectral phase coherence (BCAT: plausible texture, collapsed phase)
  advective wake  -> streamwise error profile (MOTION localizes in the wake)

Runs locally off refdata/fig3d_sel.npz (same selected samples as v4/v5); the diagnostics
need only the single displayed frame, so no cluster dump is required.
Writes figs/fig3d_contour_v6.png -- a NEW name, so nothing overwrites anything again.
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
    "font.size": 20, "axes.linewidth": 1.0, "savefig.dpi": 220,
    "xtick.direction": "in", "ytick.direction": "in",
})
C_OUR, C_BCAT = "#E4572E", "#2E5EAA"

ROWS = [("shallow_water", "Gravity waves"),
        ("com_ns", "Compressibility"),
        ("cfdbench", "Advective wake"),
        ("pdearena_ns", "Turbulent transport")]
cols = ["Ground truth", "MOTION", "BCAT", r"$|$MOTION $-$ GT$|$", r"$|$BCAT $-$ GT$|$"]

D = np.load(os.path.join(HERE, "refdata", "fig3d_sel.npz"))


def coherence(p, g, nb=16):
    """spectral phase coherence per wavenumber band: |<G* P>| / <|G||P|>."""
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


fig, axs = plt.subplots(4, 6, figsize=(18.6, 3.15 * 4),
                        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.12],
                                     "wspace": 0.05, "hspace": 0.24,
                                     "left": 0.055, "right": 0.955,
                                     "top": 0.955, "bottom": 0.055})
STATS = {}
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

    # ---------------- 6th column: per-case physics diagnostic ----------------
    ax = axs[r, 5]
    ea, eb = a - gtc, b - gtc
    if fa == "shallow_water":
        # error rms vs radius from the wavefront centre (signal-energy centroid)
        w = (gtc - c00) ** 2
        iy, ix = np.indices(gtc.shape)
        cy, cx = (w * iy).sum() / w.sum(), (w * ix).sum() / w.sum()
        rr = np.sqrt((iy - cy) ** 2 + (ix - cx) ** 2)
        rb = np.linspace(0, gtc.shape[0] * 0.55, 22)
        rmid = 0.5 * (rb[1:] + rb[:-1])
        prof = lambda e: np.array([np.sqrt(np.mean(e[(rr >= rb[i]) & (rr < rb[i + 1])] ** 2))
                                   for i in range(len(rb) - 1)])
        sig = prof(gtc - c00)
        pa_, pb_ = prof(ea), prof(eb)
        en = pb_.max() + 1e-12                                 # error curves on their own scale
        h_our, = ax.plot(rmid, pa_ / en, color=C_OUR, lw=2.2, label="MOTION")
        h_bc, = ax.plot(rmid, pb_ / en, color=C_BCAT, ls="--", lw=2.2, label="BCAT")
        h_gt, = ax.plot(rmid, sig / sig.max(), color="k", lw=1.3, alpha=0.35,
                        label="GT signal")
        # Regime boundaries are read off the GROUND-TRUTH height profile h(r), not the
        # predictions.  The radial dam break has four regimes: an undisturbed column
        # centre, the rarefaction that eats into it, the plateau the collapse leaves
        # behind, and the bore front that separates that plateau from the still far field.
        # Candidate boundaries are the two ends of each steep descent, located where
        # |dh/dr| falls back below a tenth of its local peak.  Only the OUTER end of each
        # descent is drawn: those are the two sharp features (the fan's outer edge, where
        # the solution corners onto the plateau, and the bore's leading edge, beyond which
        # nothing has moved yet).  The inner ends are smooth onsets, so a threshold on
        # |dh/dr| places them arbitrarily and they are not marked.
        _hb = np.arange(0, gtc.shape[0] * 0.55, 1.5)
        _hm = 0.5 * (_hb[1:] + _hb[:-1])
        _h = np.array([gtc[(rr >= _hb[i]) & (rr < _hb[i + 1])].mean()
                       for i in range(len(_hb) - 1)])
        _d = np.abs(np.gradient(_h, _hm))
        _act = _d > 0.1 * _d.max()                      # where the profile is actually moving
        _edges, _in = [], False
        for _i, _a in enumerate(_act):
            if _a and not _in: _edges.append(_hm[_i]); _in = True
            elif not _a and _in: _edges.append(_hm[_i]); _in = False
        for _r in _edges[1:4:2]:
            ax.axvline(_r, color="0.45", ls=":", lw=1.4, zorder=0)
        ax.set_xlim(0, rmid[-1])
        ax.set_xlabel("wavefront radius (cells)", fontsize=19)
        ax.set_ylabel("error", fontsize=19)
        # MOTION/BCAT name the two curves in EVERY row, so the key sits above the box on the
        # first row, level with the column titles; GT signal appears in this panel only and
        # is keyed inside it.
        _top = ax.legend(handles=[h_our, h_bc], fontsize=19, frameon=False, ncol=2,
                         loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2,
                         columnspacing=1.4, handlelength=1.8)
        ax.add_artist(_top)
        ax.legend(handles=[h_gt], fontsize=15, frameon=False, loc="center right")
        # where does each model's error peak, relative to the signal's own peak radius?
        STATS[fa] = dict(r_sig=float(rmid[sig.argmax()]), r_our=float(rmid[pa_.argmax()]),
                         r_bcat=float(rmid[pb_.argmax()]),
                         flat_our=float(pa_.max() / (pa_.mean() + 1e-12)),
                         flat_bcat=float(pb_.max() / (pb_.mean() + 1e-12)))
    elif fa == "cfdbench":
        # streamwise error profile (spanwise-averaged), flow left -> right
        x = np.arange(gtc.shape[1])
        pa = np.sqrt((ea ** 2).mean(0)); pb = np.sqrt((eb ** 2).mean(0))
        ax.plot(x, pa, color=C_OUR, lw=2.2, label="MOTION")
        ax.plot(x, pb, color=C_BCAT, ls="--", lw=2.2, label="BCAT")
        ax.set_xlim(0, x[-1])
        ax.set_xlabel("streamwise position (cells)", fontsize=19)
        ax.set_ylabel("error", fontsize=19)
        n = len(x); h = n // 2
        STATS[fa] = dict(x_our=float(x[pa.argmax()]), x_bcat=float(x[pb.argmax()]),
                         downstream_our=float(pa[h:].sum() / (pa.sum() + 1e-12)),
                         downstream_bcat=float(pb[h:].sum() / (pb.sum() + 1e-12)))
    else:
        ks, ca_ = coherence(a, gtc)
        _, cb_ = coherence(b, gtc)
        ax.plot(ks, ca_, color=C_OUR, lw=2.2, label="MOTION")
        ax.plot(ks, cb_, color=C_BCAT, ls="--", lw=2.2, label="BCAT")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.set_xticks([3, 5, 10, 20, 40, 64])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlim(3, ks[-1])                      # after the ticks, so it is not undone
        ax.set_xlabel("wavenumber $k$", fontsize=19)
        ax.set_ylabel("phase coherence", fontsize=19)
        STATS[fa] = dict(coh_our_min=float(ca_.min()), coh_our_max=float(ca_.max()),
                         coh_bcat_min=float(cb_.min()), coh_bcat_max=float(cb_.max()),
                         coh_our_lowk=float(ca_[:4].mean()), coh_bcat_lowk=float(cb_[:4].mean()))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.2, lw=0.5)

fig.savefig(f"{OUT}/fig3d_contour_v6.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig3d_contour_v6.pdf", bbox_inches="tight")
for k, v in STATS.items():
    print(k, {kk: round(vv, 3) for kk, vv in v.items()}, flush=True)
print("FIG3D6_DONE")
