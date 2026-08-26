"""Render Figure-1 panels b-e from fig1_data.npz (matplotlib; run as a VESSL CPU job).

  python render_fig1.py /code-vol/motion_fv2/figs_out
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/code-vol/motion_fv2/figs_out"
Z = np.load(os.path.join(D, "fig1_data.npz"))
M = json.load(open(os.path.join(D, "fig1_meta.json")))
FAMS, GN, PB = M["families"], M["mechanisms"], M["pb_fams"]
PB3D = M.get("pb3d", [])

plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.titlesize": 9.5,
                     "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.005})

SHORT = {"ACE": "Allen–Cahn", "CE-RM": "CE R–M shock", "pdearena_ns": "buoyant NS",
         "incom_ns": "incompressible NS",
         "Wave-Layer": "wave (layered)",
         "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": "3D CNS M0.1",
         "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": "3D CNS M1.0",
         "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": "3D CNS turbulent"}
# every tick carries the same formal style: the registry keys (incom_ns, com_ns, ...) are
# internal names and should not reach the figure (2026-08-18).
FSHORT = {"shallow_water": "SWE", "pdearena_ns": "PDEArena (cond)",
          "pdearena_uncond": "PDEArena (uncond)",
          "diff_react": "Diff.-React.", "incom_ns": "Incomp. NS", "com_ns": "Comp. NS",
          "Poisson-Gauss": "Poisson", "Wave-Layer": "Wave",
          "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": "3D-M0.1",
          "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": "3D-M1.0",
          "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": "3D-Turb"}

# ------------------------------------------------------------------ panel b (unified 2D+3D)
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def iso(ax, f, ref):
    lev = float(np.mean(ref) + 0.7 * np.std(ref))
    try:
        v, fc, _, _ = marching_cubes(np.clip(f, None, np.percentile(ref, 99.5)), lev)
        tri = Poly3DCollection(v[fc], alpha=0.9, linewidth=0.04)
        zc = v[fc][:, :, 2].mean(1); zc = (zc - zc.min()) / (zc.ptp() + 1e-9)
        tri.set_facecolor(plt.cm.viridis(zc)); tri.set_edgecolor((0, 0, 0, 0.05))
        ax.add_collection3d(tri)
    except Exception:
        pass
    n = ref.shape[0]
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1)); ax.axis("off")

# BEST-PER-FAMILY OVERRIDE (08-08): the buoyant-NS strip is drawn from the six-family
# benchmark weights (bylfa, the Fig-3 model) whose final-weight dump sits on /eu —
# the best prediction we have for that family until m1b/m2 converge. Interim only;
# the panel reverts to the 19-family checkpoint at convergence.
# HARD vortical case (08-08): the UNCONDITIONED buoyant-NS dump, sample chosen by the
# largest high-wavenumber energy fraction of the smoke field (most vortices), drawn
# in the paper-1 RdBu palette. Valid horizon of uncond is 4 frames -> frame 3.
BYLFA_PDEA = os.environ.get("BYLFA_PDEA", "/eu/_bfa_pdearena_uncond.npz")
_bg = _bp = None
if os.path.exists(BYLFA_PDEA):
    _bz = np.load(BYLFA_PDEA)
    _g, _p = _bz["gt"], _bz["pred"]                      # (100,10,128,128,6)

    def _hik(f):
        f = f - f.mean()
        F2 = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
        n = f.shape[0]
        ky, kx = np.indices(F2.shape)
        kr = np.sqrt((kx - n / 2) ** 2 + (ky - n / 2) ** 2)
        return F2[kr > 12].sum() / F2.sum()
    _s = int(np.argmax([_hik(_g[i, 3, ..., 2]) for i in range(len(_g))]))
    _bg, _bp = _g[_s, 3, ..., 2], _p[_s, 3, ..., 2]
    print(f"[fig1b] buoyant strip from bylfa UNCOND dump (sample {_s})", flush=True)

COLS = PB + PB3D
fig = plt.figure(figsize=(9.6, 3.0))
gs = fig.add_gridspec(2, len(COLS), hspace=0.04, wspace=0.05,
                      left=0.045, right=0.995, top=0.90, bottom=0.075)
for j, fam in enumerate(COLS):
    is3d = fam in PB3D
    gt = Z[f"b3d_{fam}_gt"] if is3d else Z[f"b_{fam}_gt"]
    pf = Z[f"b3d_{fam}_pred"] if is3d else Z[f"b_{fam}_pred"]
    force_rdbu = False
    if fam == "pdearena_ns" and _bg is not None:
        gt, pf = _bg, _bp
        force_rdbu = True                                # paper-1 palette for the smoke
    for i, f in enumerate((gt, pf)):
        if is3d:
            ax = fig.add_subplot(gs[i, j], projection="3d")
            iso(ax, f, gt)
            if j == len(PB):
                ax.text2D(-0.12, 0.5, ["ground truth", "MOTION"][i], fontsize=12,
                          rotation=90, va="center", transform=ax.transAxes)
        else:
            ax = fig.add_subplot(gs[i, j])
            v = np.percentile(np.abs(gt), 99)
            if force_rdbu:
                c00 = gt.mean(); ampv = np.abs(gt - c00).max() * 0.92
                ax.imshow(f.T, origin="lower", cmap="RdBu_r", vmin=c00 - ampv, vmax=c00 + ampv)
            elif gt.min() < 0:
                ax.imshow(f.T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            else:
                ax.imshow(f.T, origin="lower", cmap="viridis", vmin=np.percentile(gt, 1), vmax=v)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(["ground truth", "MOTION"][i], fontsize=12)
        if i == 0:
            ax.set_title(SHORT.get(fam, fam), fontsize=11.5)
    rel = np.linalg.norm(pf - gt) / (np.linalg.norm(gt) + 1e-12)
    fig.text(gs[1, j].get_position(fig).x0 + gs[1, j].get_position(fig).width / 2, 0.012,
             f"rel-$L_2$ {100*rel:.1f}%", ha="center", fontsize=11)
fig.savefig(os.path.join(D, "fig1b.png")); plt.close(fig)

# ------------------------------------------------------------------ panel c
KO, CN, BASE = Z["c_knockout"], Z["c_contrib"], Z["c_base"]
# persistence normalization (PHYS, refdata table); score = dErr / (persistence - base), clip [0,1]
PERS = {"shallow_water": 3.33, "diff_react": 13.64, "com_ns": 4.36, "incom_ns": 6.93,
        "pdearena_ns": 67.55, "pdearena_uncond": 44.90, "CE-RP": 35.38, "CE-CRP": 44.89,
        "CE-KH": 6.67, "CE-Gauss": 16.70, "CE-RM": 34.62, "NS-Sines": 86.96, "NS-Gauss": 55.70,
        "ACE": 24.61, "Wave-Layer": 154.31, "Poisson-Gauss": 100.0,
        "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": 9.78,
        "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": 21.52,
        "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": 49.90}
head = np.array([max(PERS[f] / 100.0 - BASE[j], 1e-3) for j, f in enumerate(FAMS)])
S = np.clip(KO / head[None, :], 0, 1)
# family-SPECIFIC contribution (08-07): heads such as compressibility carry a
# family-common component (density/persistence-like information) and so dominate the
# raw norm in every family. Report instead the log fold-ENRICHMENT of each head's
# share of the family's summed tendency over that head's median share across families
# (denominator floored at a 1% share so near-silent heads cannot explode the ratio),
# clipped at zero and normalized within each family -- each column then shows which
# mechanisms that family specifically draws on, with the common carrier cancelled.
_share = CN / (CN.sum(0, keepdims=True) + 1e-12)
_den = np.maximum(np.median(_share, axis=1, keepdims=True), 0.01)
_R = np.clip(np.log2(np.maximum(_share, 1e-9) / _den), 0, None)
CNn = _R / (_R.max(axis=0, keepdims=True) + 1e-12)
# Paper labels name the OPERATOR each head reads, not the physics claim (2026-08-13):
# the association with a mechanism is a result of the knockout map, not of the label.
MLAB = {"advection": "advection", "shock": "upwind transport", "diffusion": "diffusion",
        "wave": "wave-Laplacian", "buoyancy": "gravity gradient",
        "compressible": "density coupling", "vortex": "vortex stretching",
        "reaction_nl": "nonlinear reaction", "spectral": "spectral nonlocal",
        # the remaining ten rows of the SUPPLEMENTARY full map carried their registry
        # names, which name the development intent rather than the operator the head
        # reads (2026-08-18).  "geometry" was the worst of these: it consumes no geometry
        # input at all -- it is the mean curvature of the latent iso-surfaces, computable
        # on any field -- and it is the top knockout of 3D CNS M0.1, where a reader would
        # otherwise ask which geometry the volumes are supposed to carry.  All ten now use
        # the operator names the Methods section already lists.
        "geometry": "mean curvature", "gradvec": "directional gradient",
        "elliptic": "domain-mean coupling", "boundary_layer": "near-wall coupling",
        "dilatation": "dilatational coupling", "bernoulli": "dynamic pressure",
        "coefficient": "coefficient-weighted diffusion",
        "kinematic": "velocity-gradient invariants", "shear": "first-axis gradient",
        "reaction": "linear reaction"}

# Row/column DESIGN (08-07): specialist mechanisms on top, ordered so their family
# blocks descend left->right (diagonal); the two broad "universal operator" rows
# (compressibility, pressure projection) sit at the bottom as a horizontal band --
# the two-tier structure (specialist vs universal) becomes visible at a glance.
# mean curvature and the directional-gradient bank join the main panel (2026-08-18): the
# Results text now reports both as the leading knockouts of the three-dimensional families,
# so the panel has to show the rows those claims are read from.
SEL = ["advection", "reaction_nl", "wave", "shock", "compressible",
       "buoyancy", "diffusion", "vortex", "spectral", "geometry", "gradvec"]
_FBLOCKS = [["incom_ns", "NS-Sines", "NS-Gauss"],            # advection / incompressible transport
            ["ACE", "diff_react"],                            # reaction(+diffusion) 2D
            ["shallow_water", "Wave-Layer", "Poisson-Gauss"],  # wave / steady
            ["CE-RP", "CE-RM"],                               # shock
            ["com_ns", "CE-CRP", "CE-Gauss", "CE-KH"],        # compressible group
            ["pdearena_ns", "pdearena_uncond"],               # buoyancy
            ["pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08",   # 3D: diffusion/vortex
             "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08",
             "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08"]]
# The x axis is already ordered by system block; the Results text refers to those blocks by
# name ("incompressible flow", "reaction systems", ...), so the blocks are drawn as spans
# under the tick labels rather than left for the reader to infer (2026-08-18).
_BLAB = ["incompressible", "reaction", "wave / steady", "shock",
         "compressible Euler", "buoyant", "3D compressible"]
_BSPAN, _c0 = [], 0
for _blk, _lab in zip(_FBLOCKS, _BLAB):
    _BSPAN.append((_c0, _c0 + len(_blk) - 1, _lab)); _c0 += len(_blk)
_ford_names = [f for blk in _FBLOCKS for f in blk]
assert sorted(_ford_names) == sorted(FAMS), set(_ford_names) ^ set(FAMS)
FORD = [FAMS.index(f) for f in _ford_names]
DIV3D = len(FAMS) - 3 - 0.5

def heat_pair(rows, fname, w, hh, ybar=-0.62):
    """Stacked WIDE pair: knockout (Reds, causal) on top, contribution (Blues, activity) below."""
    global _YBAR; _YBAR = ybar
    idx = [GN.index(m) for m in rows]
    fig, axes = plt.subplots(2, 1, figsize=(w, hh * 1.12), sharex=True,
                             gridspec_kw={"hspace": 0.22})
    for ax, Mx, cm_, title in [(axes[0], S[idx][:, FORD], "Reds", "knockout damage"),
                               (axes[1], CNn[idx][:, FORD], "Blues",
                                "family-specific contribution")]:
        im = ax.imshow(Mx, aspect="auto", cmap=cm_, vmin=0, vmax=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([MLAB.get(m, m) for m in rows], fontsize=8)
        if DIV3D is not None:
            ax.axvline(DIV3D, color="k", lw=0.8)
        ax.set_title(title, fontsize=9, pad=0)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.008)
    axes[1].set_xticks(range(len(FAMS)))
    axes[1].set_xticklabels([FSHORT.get(FAMS[j], FAMS[j]) for j in FORD],
                            rotation=90, fontsize=7.5)
    tr = axes[1].get_xaxis_transform()          # x in data units, y in axes fraction
    for i0, i1, lab in _BSPAN:
        axes[1].plot([i0 - 0.38, i1 + 0.38], [_YBAR, _YBAR], transform=tr, clip_on=False,
                     color="0.25", lw=0.9, solid_capstyle="butt")
        for _e in (i0 - 0.38, i1 + 0.38):       # end ticks make the span read as a bracket
            axes[1].plot([_e, _e], [_YBAR, _YBAR + 0.012], transform=tr, clip_on=False,
                         color="0.25", lw=0.9)
        axes[1].text((i0 + i1) / 2.0, _YBAR - 0.018, lab, transform=tr, clip_on=False,
                     ha="center", va="top", fontsize=7.2, color="0.25")
    fig.savefig(os.path.join(D, fname))
    fig.savefig(os.path.join(D, fname.replace(".png", ".pdf")))   # vector: crisp labels
    plt.close(fig)

heat_pair(SEL, "fig1c.png", 7.2, 3.25, ybar=-0.92)
order = list(np.argsort(-S.sum(1)))
heat_pair([GN[i] for i in order], "fig1c_full.png", 7.2, 5.2, ybar=-0.42)   # supplementary: all mechanisms

# ------------------------------------------------------------------ panel d: spectra + coherence
def radial(F2):
    n = F2.shape[0]
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    kr = np.sqrt(KX**2 + KY**2).astype(int)
    out = np.zeros(n // 2)
    for k in range(1, n // 2):
        out[k] = F2[kr == k].mean() if (kr == k).any() else 0
    return out

def spec_coh(gt, pr):
    G, P = np.fft.fft2(gt), np.fft.fft2(pr)
    Eg, Ep = radial(np.abs(G) ** 2), radial(np.abs(P) ** 2)
    num, den = radial((P * np.conj(G)).real), radial(np.abs(P) * np.abs(G)) + 1e-12
    return Eg, Ep, num / den

ks = None; Eg = Eon = Eoff = Con = Coff = 0
for i in range(Z["d_gt"].shape[0]):
    eg, eon, con = spec_coh(Z["d_gt"][i], Z["d_on"][i])
    _, eoff, coff = spec_coh(Z["d_gt"][i], Z["d_off"][i])
    Eg, Eon, Eoff = Eg + eg, Eon + eon, Eoff + eoff
    Con, Coff = Con + con, Coff + coff
n = Z["d_gt"].shape[0]
k = np.arange(len(Eg))
fig, axes = plt.subplots(2, 1, figsize=(2.6, 3.45), gridspec_kw={"hspace": 0.5})
axes[0].loglog(k[1:], Eg[1:] / n, "k-", lw=1.2, label="ground truth")
axes[0].loglog(k[1:], Eon[1:] / n, "-", color="#2a78d6", lw=1.2, label="full model")
axes[0].loglog(k[1:], Eoff[1:] / n, "--", color="#eb6834", lw=1.2, label="transport off")
axes[0].set_xlabel("wavenumber $k$"); axes[0].set_ylabel("energy $E(k)$"); axes[0].legend(fontsize=7.5)
axes[1].semilogx(k[1:], Con[1:] / n, "-", color="#2a78d6", lw=1.2)
axes[1].semilogx(k[1:], Coff[1:] / n, "--", color="#eb6834", lw=1.2)
axes[1].set_xlabel("wavenumber $k$"); axes[1].set_ylabel("phase coherence"); axes[1].set_ylim(-0.1, 1.05)
fig.savefig(os.path.join(D, "fig1d.png")); plt.close(fig)

# ------------------------------------------------------------------ panel e: vortex head field
fig, axes = plt.subplots(1, 2, figsize=(2.3, 1.35))
if "e_vortex3d" in Z:
    v3 = Z["e_vortex3d"]; vm = np.percentile(v3, 99) + 1e-12
    axes[0].imshow(v3.T, origin="lower", cmap="magma", vmin=0, vmax=vm)
    axes[0].set_title(f"3D turb.\nmax {v3.max():.2e}", fontsize=7.5)
if "e_vortex2d" in Z:
    v2 = Z["e_vortex2d"]
    axes[1].imshow(v2.T, origin="lower", cmap="magma", vmin=0, vmax=(np.percentile(Z["e_vortex3d"], 99) if "e_vortex3d" in Z else 1))
    axes[1].set_title(f"2D flow\nmax {v2.max():.1e} ($\\equiv0$)", fontsize=7.5)
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("vortex-stretching head $|(\\omega\\!\\cdot\\!\\nabla)u|$", fontsize=8.5, y=1.08)
fig.savefig(os.path.join(D, "fig1e.png")); plt.close(fig)

# ------------------------------------------------------------------ panels f, g (fig1_extra.npz)
ep = os.path.join(D, "fig1_extra.npz")
if os.path.exists(ep):
    E = np.load(ep)
    if "f_c_true" in E:
        fig, axes = plt.subplots(1, 2, figsize=(2.5, 1.35))
        print("f shapes:", E["f_c_true"].shape, E["f_c_learned"].shape)
        axes[0].imshow(E["f_c_true"].T, origin="lower", cmap="cividis")
        axes[0].set_title("true $c(x)$", fontsize=8)
        axes[1].imshow(E["f_c_learned"].T, origin="lower", cmap="cividis")
        axes[1].set_title(f"coefficient head\n$|r|$={float(E['f_corr']):.2f}", fontsize=8)
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("hidden wave-speed recovery (Wave-Layer)", fontsize=8.5, y=1.06)
        fig.savefig(os.path.join(D, "fig1f.png")); plt.close(fig)
    def cut_panel(gt, fu, ns, row, labels, fname, cutlab):
        fig = plt.figure(figsize=(7.2, 1.75))
        v0, v1 = np.percentile(gt, 1), np.percentile(gt, 99)
        for i, (f, t) in enumerate([(gt, "ground truth"), (fu, "full model"), (ns, labels)]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(f.T, origin="lower", cmap="viridis", vmin=v0, vmax=v1)
            ax.axhline(row, color="w", lw=0.6, ls="--")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=8.5)
        ax = fig.add_subplot(1, 4, 4)
        x = np.arange(gt.shape[0])
        ax.plot(x, gt[:, row], "k-", lw=1.1, label="GT")
        ax.plot(x, fu[:, row], "-", color="#2a78d6", lw=1.1, label="full")
        ax.plot(x, ns[:, row], "--", color="#eb6834", lw=1.1, label=cutlab)
        ax.legend(fontsize=7); ax.set_title("density along cut", fontsize=8.5)
        ax.tick_params(labelsize=6)
        fig.subplots_adjust(wspace=0.12)
        fig.savefig(os.path.join(D, fname)); plt.close(fig)
    # combined two-row knockout vignette (e): shock (broad-mask learned selectivity) +
    # compressibility (never masked — fully emergent)
    if "g_gt" in E and "h_gt" in E:
        fig = plt.figure(figsize=(7.2, 3.4))
        rows = [("g", "shock head", "no shock", "CE Richtmyer–Meshkov"),
                ("h", "density-coupling head", "no density coupl.", "CE curved Riemann")]
        for r, (pre, rem, cl, fam) in enumerate(rows):
            gt, fu, ns, row = E[f"{pre}_gt"], E[f"{pre}_full"], E[f"{pre}_noshock" if pre=="g" else f"{pre}_nocmp"], int(E[f"{pre}_row"])
            v0, v1 = np.percentile(gt, 1), np.percentile(gt, 99)
            for i, (f, t) in enumerate([(gt, "ground truth"), (fu, "full model"), (ns, f"{rem} removed")]):
                ax = fig.add_subplot(2, 4, 4 * r + i + 1)
                ax.imshow(f.T, origin="lower", cmap="viridis", vmin=v0, vmax=v1)
                ax.axhline(row, color="w", lw=0.6, ls="--")
                if i == 0:
                    ax.text(2, row + 3, "cut", color="w", fontsize=8,
                            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(t, fontsize=8.5)
                if i == 0: ax.set_ylabel(fam, fontsize=8)
            ax = fig.add_subplot(2, 4, 4 * r + 4)
            x = np.arange(gt.shape[0])
            ax.plot(x, gt[:, row], "k-", lw=1.1, label="GT")
            ax.plot(x, fu[:, row], "-", color="#2a78d6", lw=1.1, label="full")
            ax.plot(x, ns[:, row], "--", color="#eb6834", lw=1.1, label=cl)
            ax.legend(fontsize=7)
            ax.set_title("density along cut", fontsize=8.5)
            ax.tick_params(labelsize=6)
        fig.subplots_adjust(wspace=0.14, hspace=0.28)
        fig.savefig(os.path.join(D, "fig1ef.png")); plt.close(fig)
    if "h_gt" in E:
        cut_panel(E["h_gt"], E["h_full"], E["h_nocmp"], int(E["h_row"]),
                  "density-coupling head removed", "fig1h.png", "no density coupl.")
    if "g_gt" in E:
        gt, fu, ns, row = E["g_gt"], E["g_full"], E["g_noshock"], int(E["g_row"])
        fig = plt.figure(figsize=(7.2, 1.75))
        v0, v1 = np.percentile(gt, 1), np.percentile(gt, 99)
        for i, (f, t) in enumerate([(gt, "ground truth"), (fu, "full model"), (ns, "shock head removed")]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(f.T, origin="lower", cmap="viridis", vmin=v0, vmax=v1)
            ax.axhline(row, color="w", lw=0.6, ls="--")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=8.5)
        ax = fig.add_subplot(1, 4, 4)
        x = np.arange(gt.shape[0])
        ax.plot(x, gt[:, row], "k-", lw=1.1, label="GT")
        ax.plot(x, fu[:, row], "-", color="#2a78d6", lw=1.1, label="full")
        ax.plot(x, ns[:, row], "--", color="#eb6834", lw=1.1, label="no shock")
        ax.legend(fontsize=7); ax.set_title("density along cut", fontsize=8.5)
        ax.tick_params(labelsize=6)
        fig.subplots_adjust(wspace=0.12)
        fig.savefig(os.path.join(D, "fig1g.png")); plt.close(fig)
print("rendered:", [f for f in os.listdir(D) if f.endswith(".png")])
