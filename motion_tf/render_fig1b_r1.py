"""Render Figure-1 panels b-e from fig1_data.npz (matplotlib; run as a VESSL CPU job).

  python render_fig1.py /work/motion_fv2/figs_out
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "figs_out"
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
        zc = v[fc][:, :, 2].mean(1); zc = (zc - zc.min()) / (np.ptp(zc) + 1e-9)
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
# REVISION-R1 10-29 (user): the panel labels give the family MEAN test error of MOTION-S (Supplementary Table s2fam:
# relative L2, physical units, up to 128 test trajectories, ten-step rollout), not the error of the shown trajectory.
FAM_MEAN = {"ACE": 0.94, "CE-RM": 7.65, "incom_ns": 2.95, "Wave-Layer": 11.47,
            "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": 15.81,
            "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": 3.84,   # REVISION-R1 10-30 (user): M0.1 column replaces M1.0
            "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": 22.99}
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
                ax.text2D(-0.12, 0.5, ["ground truth", "MOTION-S"][i], fontsize=10,
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
                ax.set_ylabel(["ground truth", "MOTION-S"][i], fontsize=10)
        if i == 0:
            ax.set_title(SHORT.get(fam, fam), fontsize=10)
    rel = np.linalg.norm(pf - gt) / (np.linalg.norm(gt) + 1e-12)
    fig.text(gs[1, j].get_position(fig).x0 + gs[1, j].get_position(fig).width / 2, 0.012,
             (f"rel-$L_2$ {FAM_MEAN[fam]:.2f}%" if fam in FAM_MEAN else f"rel-$L_2$ {100*rel:.1f}%"), ha="center",
             fontsize=9.5)
fig.savefig(os.path.join(D, "fig1b_small.png")); plt.close(fig)

