"""Render Fig-1d candidates from fig1d_ko.npz (head-knockout spectra, one model/one ckpt).

  python render_fig1d_ko.py /code-vol/motion_fv2/figs_m1s2b

Two layouts per family so the panel choice can be made by eye:
  fig1dko_<fam>_panels.png  -- 2 x (1+3) grid: columns = full model + KO_SHOW knockouts,
                               row 1 energy spectrum vs GT, row 2 phase coherence vs full.
  fig1dko_<fam>_overlay_<ch>.png -- single 2-stack with every dumped knockout overlaid.
Env: KO_SHOW=spectral,shock,diffusion  KO_CHAN=vel|last|all
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/code-vol/motion_fv2/figs_m1s2b"
Z = np.load(os.path.join(D, "fig1d_ko.npz"))
M = json.load(open(os.path.join(D, "fig1d_ko_meta.json")))
HEADS, FAMS = M["heads"], M["families"]

plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.titlesize": 9.5,
                     "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

MLAB = {"spectral": "spectral nonlocal", "shock": "upwind transport",
        "compressible": "density coupling", "gradvec": "directional-gradient bank",
        "reaction_nl": "reaction", "vortex": "vortex stretching",
        "diffusion": "diffusion", "buoyancy": "gravity gradient", "advection": "advection",
        "wave": "wave"}
FLAB = {"pdearena_ns": "buoyant turbulence", "incom_ns": "incompressible NS"}


def chans(A, mode):
    if mode == "last":
        return A[..., -1:]
    if mode == "vel" and A.shape[-1] >= 2:
        return A[..., :2]
    return A


def radial_bins(n):
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    return np.sqrt(KX ** 2 + KY ** 2).astype(int)


def spec_coh(gt, pr):
    """gt, pr: (N, H, W, C) physical fields -> radially binned E_gt, E_pr, coherence."""
    n = gt.shape[1]
    kr = radial_bins(n)
    G = np.fft.fft2(gt, axes=(1, 2))
    P = np.fft.fft2(pr, axes=(1, 2))
    Pg = (np.abs(G) ** 2).sum((0, -1))
    Pp = (np.abs(P) ** 2).sum((0, -1))
    cr = (P * np.conj(G)).sum((0, -1))
    Eg = np.zeros(n // 2); Ep = np.zeros(n // 2); C = np.zeros(n // 2)
    for k in range(1, n // 2):
        m = kr == k
        if not m.any():
            continue
        Eg[k], Ep[k] = Pg[m].mean(), Pp[m].mean()
        C[k] = np.abs(cr[m].sum()) / (np.sqrt(Pg[m].sum() * Pp[m].sum()) + 1e-12)
    return Eg, Ep, C


C_GT, C_FULL, C_KO = "k", "#2a78d6", "#eb6834"
PALETTE = ["#eb6834", "#3f9b57", "#8a5cd6", "#c2417f", "#8c8c8c", "#b8860b", "#2ca8a8", "#7f7f7f"]
SHOW = [h for h in os.environ.get("KO_SHOW", "spectral,shock,diffusion").split(",") if h in HEADS]
CHM = os.environ.get("KO_CHAN", "vel")

for fam in FAMS:
    gt = Z[f"{fam}__gt"]
    band = slice(15, 36) if fam.startswith("pdearena") else slice(8, 25)
    res = {}
    for kname in ["full"] + HEADS:
        g_, p_ = chans(gt, CHM), chans(Z[f"{fam}__{kname}"], CHM)
        Eg, Ep, C = spec_coh(g_, p_)
        rel = np.linalg.norm(p_ - g_) / (np.linalg.norm(g_) + 1e-12)
        res[kname] = (Eg, Ep, C, rel)
        print(f"[render] {fam} {kname}: rel {100*rel:.2f}%  "
              f"coh_band {C[band].mean():.3f}", flush=True)
    Eg = res["full"][0]
    k = np.arange(1, len(Eg))

    # -------- candidate 1: user 4-panel layout (no GT panel, no cuts) --------
    cols = ["full"] + SHOW
    fig, axes = plt.subplots(2, len(cols), figsize=(2.05 * len(cols), 3.3),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.3})
    for j, kname in enumerate(cols):
        _, Ep, C, rel = res[kname]
        lab = "full model" if kname == "full" else f"$-$ {MLAB.get(kname, kname)}"
        a0, a1 = axes[0, j], axes[1, j]
        a0.loglog(k, Eg[1:], "-", color=C_GT, lw=1.0, label="GT")
        a0.loglog(k, Ep[1:], "--" if kname != "full" else "-",
                  color=C_FULL if kname == "full" else C_KO, lw=1.2)
        a0.set_title(f"{lab}\nrel-$L_2$ {100*rel:.1f}%", fontsize=8.5)
        a0.axvspan(band.start, band.stop - 1, color="#dfe3e8", alpha=0.65, zorder=0)
        a1.axvspan(band.start, band.stop - 1, color="#dfe3e8", alpha=0.65, zorder=0)
        if kname != "full":
            a1.semilogx(k, res["full"][2][1:], "-", color=C_FULL, lw=1.0, alpha=0.55)
        a1.semilogx(k, C[1:], "--" if kname != "full" else "-",
                    color=C_FULL if kname == "full" else C_KO, lw=1.2)
        a1.set_ylim(-0.05, 1.05)
        a1.set_xlabel("wavenumber $k$", fontsize=8)
        if j == 0:
            a0.set_ylabel("$E(k)$", fontsize=8.5)
            a1.set_ylabel("phase coh. $\\gamma(k)$", fontsize=8.5)
        else:
            a0.set_yticklabels([]); a1.set_yticklabels([])
        ylims = axes[0, 0].get_ylim()
        a0.set_ylim(*ylims)
        for a in (a0, a1):
            a.tick_params(labelsize=7)
    tag = "-".join(SHOW)
    fig.savefig(os.path.join(D, f"fig1dko_{fam}_panels_{CHM}_{tag}.png")); plt.close(fig)

    # -------- candidate 2: overlay 2-stack with every knockout --------
    fig, ax = plt.subplots(2, 1, figsize=(2.9, 3.7), gridspec_kw={"hspace": 0.5})
    ax[0].loglog(k, Eg[1:], "-", color=C_GT, lw=1.3, label="ground truth")
    ax[0].loglog(k, res["full"][1][1:], "-", color=C_FULL, lw=1.2, label="full model")
    ax[1].semilogx(k, res["full"][2][1:], "-", color=C_FULL, lw=1.2)
    # KO_OVER restricts the overlay to the knockouts that carry the argument.  Drawing all
    # eight makes the legend unreadable at panel size and buries the two that matter: one
    # head guards the spectrum, another guards the phase.
    _over = [h for h in os.environ.get("KO_OVER", ",".join(HEADS)).split(",") if h in HEADS]
    for i, m in enumerate(_over):
        _, Ep, C, _ = res[m]
        ax[0].loglog(k, Ep[1:], "--", color=PALETTE[i % len(PALETTE)], lw=1.1,
                     label=f"w/o {MLAB.get(m, m)}")
        ax[1].semilogx(k, C[1:], "--", color=PALETTE[i % len(PALETTE)], lw=1.1)
    for a in ax:
        a.axvspan(band.start, band.stop - 1, color="#dfe3e8", alpha=0.65, zorder=0)
    ax[0].set_ylabel("energy $E(k)$")
    ax[0].legend(fontsize=6.2, ncol=1, loc="lower left", frameon=False,
                 handlelength=1.6, labelspacing=0.25, borderpad=0.1)
    ax[1].set_ylabel("phase coherence"); ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_xlabel("wavenumber $k$")
    # no in-panel title: it collides with the panel letter and the caption names the family
    fig.savefig(os.path.join(D, f"fig1dko_{fam}_overlay_{CHM}.png"))
    fig.savefig(os.path.join(D, f"fig1dko_{fam}_overlay_{CHM}.pdf"))  # vector line art
    plt.close(fig)

# -------- FINAL Fig-1d (user-selected 08-11): GT / full / -pressure proj / -upwind adv,
# velocity channel, vertical 2-stack (drop-in replacement for the fig_sepcollapse_v slot).
# Physics: no pressure projection -> energy AND phase collapse (incompressibility lost);
# no upwind advection -> energy nearly intact, phase alone collapses in the filament band.
if os.environ.get("KO_FINAL"):
    fam = "pdearena_ns"
    band = slice(15, 36)
    g_ = chans(Z[f"{fam}__gt"], "vel")
    sel = [("full", "#2a78d6", "-", "full model"),
           ("spectral", "#eb6834", "--", "$-$ spectral nonlocal"),
           ("shock", "#3f9b57", "--", "$-$ upwind transport")]
    curves = {m: spec_coh(g_, chans(Z[f"{fam}__{m}"], "vel")) for m, _, _, _ in sel}
    Eg = curves["full"][0]
    k = np.arange(1, len(Eg))
    fig, ax = plt.subplots(2, 1, figsize=(2.9, 3.8), gridspec_kw={"hspace": 0.52})
    ax[0].loglog(k, Eg[1:], "-", color="k", lw=1.4, label="ground truth")
    for m, c, ls, lab in sel:
        _, Ep, C = curves[m]
        ax[0].loglog(k, Ep[1:], ls, color=c, lw=1.2, label=lab)
        ax[1].semilogx(k, C[1:], ls, color=c, lw=1.2)
    for a in ax:
        a.axvspan(band.start, band.stop - 1, color="#dfe3e8", alpha=0.65, zorder=0)
    lo = min(np.min(v[1][1:]) for v in curves.values())
    hi = max(np.max(v[1][1:]) for v in curves.values())
    ax[0].set_ylim(lo * 0.5, hi * 2)                       # nothing clipped
    ax[0].set_ylabel("energy $E(k)$")
    ax[0].legend(fontsize=6.4, loc="lower left", frameon=True, borderpad=0.3)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_ylabel("phase coherence $\\gamma(k)$")
    ax[1].set_xlabel("wavenumber $k$")
    fig.savefig(os.path.join(D, "fig1d_v2.png")); plt.close(fig)
    print("[final] fig1d_v2.png written", flush=True)

print("rendered:", [f for f in os.listdir(D) if f.startswith("fig1dko") or f.startswith("fig1d_")], flush=True)
