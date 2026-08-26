"""Fig3d v4: the 4-case mechanism contour EXTENDED by a per-case physics diagnostic
(6th column) that quantifies HOW each model's error is physical or not:
  gravity waves   -> error vs radius from the wavefront center (BCAT peaks AT the front)
  compressibility -> spectral phase coherence vs wavenumber (MOTION holds phase)
  turbulence      -> spectral phase coherence (BCAT: plausible texture, collapsed phase)
  advective wake  -> streamwise error profile (MOTION localizes in the wake)
Output -> /code-vol/motion_fv2/figs_out/fig3d_contour_v4.png
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "/code-vol/motion_fv2/figs_out"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 14, "axes.linewidth": 1.0, "savefig.dpi": 220,
    "xtick.direction": "in", "ytick.direction": "in",
})
C_OUR, C_BCAT = "#E4572E", "#2E5EAA"

ROWS = [("shallow_water", "shallow_water", [5], 0, 10, "Gravity waves", "maxerr"),
        ("com_ns", "com_ns", [0, 1, 3, 4], 3, 10, "Compressibility", "maxerr"),
        ("pdearena_ns", "incom_ns_arena", [0, 1, 2], 2, 10, "Turbulent transport", "maxstd"),
        ("cfdbench", "cfdbench", [0, 1, 2], 0, 10, "Advective wake", "maxerr")]
cols = ["ground truth", "MOTION", "BCAT", r"$|$MOTION $-$ GT$|$", r"$|$BCAT $-$ GT$|$"]


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


fig, axs = plt.subplots(4, 6, figsize=(17.4, 2.9 * 4),
                        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.12],
                                     "wspace": 0.05, "hspace": 0.10,
                                     "left": 0.035, "right": 0.985,
                                     "top": 0.955, "bottom": 0.055})
for r, (fa, fb, slots, pc, F, lab, rule) in enumerate(ROWS):
    A = np.load(f"/eu/_bfa_{fa}.npz")
    aP = A["pred"][..., slots][:, :F]; aG = A["gt"][..., slots][:, :F]
    Bz = np.load(f"/eu/_bcatf_{fb}.npz"); nc = len(slots)
    bP = Bz["pred"][..., :nc][:, :F]
    ns = min(len(aP), len(bP)); fr = F - 1

    def rl2(p, i):
        o = p[i, fr]; t = aG[i, fr]
        return 100 * np.sqrt(((o - t) ** 2).sum()) / np.sqrt((t ** 2).sum())
    if rule == "maxstd":
        s = int(np.argmax([aG[i, fr, ..., pc].std() for i in range(ns)]))
    else:
        s = int(np.argmax([rl2(aP, i) + rl2(bP, i) for i in range(ns)]))
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
    axs[r, 0].set_ylabel(lab, fontsize=14, rotation=90, va="center")
    bb = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    axs[r, 3].text(0.5, 0.06, f"{rl2(aP, s):.1f}%", color="k", fontsize=12.5, ha="center",
                   transform=axs[r, 3].transAxes, bbox=bb)
    axs[r, 4].text(0.5, 0.06, f"{rl2(bP, s):.1f}%", color="k", fontsize=12.5, ha="center",
                   transform=axs[r, 4].transAxes, bbox=bb)

    # ---------------- 6th column: per-case physics diagnostic ----------------
    ax = axs[r, 5]
    ea, eb = a - gtc, b - gtc
    if fa == "shallow_water":
        # error rms vs radius from the wavefront center (signal-energy centroid)
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
        ax.plot(rmid, pa_ / en, color=C_OUR, lw=2.0, label="MOTION")
        ax.plot(rmid, pb_ / en, color=C_BCAT, ls="--", lw=2.0, label="BCAT")
        ax.plot(rmid, sig / sig.max(), color="k", lw=1.2, alpha=0.35, label="GT signal (a.u.)")
        ax.text(0.97, 0.965, "error vs. wavefront radius", transform=ax.transAxes, fontsize=11.5,
                ha="right", va="top")
        ax.legend(fontsize=9.5, frameon=False, loc="center right")
    elif fa == "cfdbench":
        # streamwise error profile (spanwise-averaged), flow left -> right
        x = np.arange(gtc.shape[1])
        pa = np.sqrt((ea ** 2).mean(0)); pb = np.sqrt((eb ** 2).mean(0))
        ax.plot(x, pa, color=C_OUR, lw=2.0, label="MOTION")
        ax.plot(x, pb, color=C_BCAT, ls="--", lw=2.0, label="BCAT")
        ax.text(0.97, 0.965, "streamwise error profile", transform=ax.transAxes, fontsize=11.5,
                ha="right", va="top")
        ax.set_xlabel("streamwise position (cells)", fontsize=11)
        ax.legend(fontsize=9.5, frameon=False, loc=(0.03, 0.42))
    else:
        ks, ca_ = coherence(a, gtc)
        _, cb_ = coherence(b, gtc)
        ax.plot(ks, ca_, color=C_OUR, lw=2.0, label="MOTION")
        ax.plot(ks, cb_, color=C_BCAT, ls="--", lw=2.0, label="BCAT")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 40, 64])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
        ax.text(0.97, 0.5, "phase coherence", transform=ax.transAxes, fontsize=11.5,
                ha="right", va="center")
        ax.legend(fontsize=9.5, frameon=False, loc="lower left")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(labelsize=9.5)
    ax.grid(alpha=0.2, lw=0.5)
    print(f"{fa}: sample={s} diag done", flush=True)

fig.savefig(f"{OUT}/fig3d_contour_v4.png", bbox_inches="tight")
print("FIG3D4_DONE", flush=True)
