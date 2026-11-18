"""Supplementary figure, two-panel version (2026-11-17): panels a (command maps) and b (knockout map) of
render_figS_ns_control.py at a larger size; the calibration scatter and the correlation bars are dropped (their values
are in the command-summary table). Original docstring follows.

Supplementary figure (REVISION-R1, 10-22): latent commands on MOTION-NS (the joint-benchmark model, zsc arm, ckpt_160000),
in the format of main Fig. 2 (command -> commanded response -> reference; calibration; per-family bars).

  a  maps, step 1, held-out MTF test trajectory 1, amplitude 1 s.d. of u(h): Comp. NS shear and PDEArena vortex (vorticity of the
     velocity response vs the kinematic flow map of the command), Comp. NS expansion (density response vs the continuity
     reference -div(rho dU)).
  b  uniform command: one-step displacement on held-out trajectories against the calibrated command eta_c alpha (y = x).
     Incomp. NS: velocity and passive scalar shown separately.
  c  pattern correlation of the response with the reference (step 1 and step 10), per family and command; Incomp. NS
     velocity (open) and passive scalar (hatched, 2 s.d.).
Sources: motion_local/out_zsc_vs, out_zsc_ctl1, out_pda_ctl1, out_zsc_bigsv (latent_control_k.py / latent_control3.py,
MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1, MTF test units 5-8 for testing, 1-4 for calibration).
"""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
for _f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _p = f"/mnt/c/Windows/Fonts/{_f}"
    if os.path.exists(_p): fm.fontManager.addfont(_p)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
L = os.environ.get("MOTION_LOCAL", "motion_local")
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.linewidth": 0.6, "savefig.dpi": 400})
INK, MUTED = "#222222", "#8a8a8a"
COL = {"com_ns": "#1b9e77", "pdearena_ns": "#d95f02", "incom_ns": "#7570b3"}
LBL = {"com_ns": "Comp. NS", "pdearena_ns": "PDEArena", "incom_ns": "Incomp. NS"}


def vort(v):
    return np.gradient(v[..., 1], axis=0) - np.gradient(v[..., 0], axis=1)


def r_(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


fig = plt.figure(figsize=(7.2, 4.75))
G = GridSpec(1, 2, figure=fig, width_ratios=[1.75, 1.0], wspace=0.62, left=0.05, right=0.905, top=0.94, bottom=0.145)

# ---------------- a: maps
ga = fig.add_gridspec(3, 3, left=0.052, right=0.632, top=0.925, bottom=0.045, wspace=0.03, hspace=0.03)
rows = [("Comp. NS\nshear", np.load(f"{L}/out_zsc_vs/c3_curlshear_com_ns.npz")["com_ns|shear|vel"], "vel"),
        ("PDEArena\nvortex", np.load(f"{L}/out_zsc_vs/c3_curlshear_pdearena_ns.npz")["pdearena_ns|curl|vel"], "vel"),
        ("Comp. NS\nexpansion", np.load(f"{L}/out_zsc_ctl1/control3dv_zsc_com_ns.npz")["com_ns|div|rho"], "rho")]
Ucmd = [np.load(f"{L}/out_zsc_vs/c3_curlshear_com_ns.npz")["com_ns|shear|U"],
        np.load(f"{L}/out_zsc_vs/c3_curlshear_pdearena_ns.npz")["pdearena_ns|curl|U"],
        np.load(f"{L}/out_zsc_ctl1/control3dv_zsc_com_ns.npz")["com_ns|div|U"]]
titles = ["command", "response", "reference"]
J = {"a": {}}
for i, ((lab, A, kind), U) in enumerate(zip(rows, Ucmd)):
    if kind == "vel":
        m, ref = vort(A[0, 0]), vort(A[1, 0]); cmap = "RdBu_r"; refl = "flow map"
    else:
        m, ref = A[0], A[1]; cmap = "PuOr_r"; refl = "continuity"
    vm = np.percentile(np.abs(ref), 99) + 1e-12
    cmdf = vort(U) if i < 2 else (np.gradient(U[..., 0], axis=0) + np.gradient(U[..., 1], axis=1))
    for j, (F, c) in enumerate(((cmdf, "Greys"), (m, cmap), (ref, cmap))):
        ax = fig.add_subplot(ga[i, j])
        if j == 0:
            ax.imshow(F.T, origin="lower", cmap="PRGn", vmin=-np.abs(F).max(), vmax=np.abs(F).max())
        else:
            ax.imshow(F.T, origin="lower", cmap=c, vmin=-vm * (1 if j == 2 else np.percentile(np.abs(m), 99) / vm),
                      vmax=vm * (1 if j == 2 else np.percentile(np.abs(m), 99) / vm))
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(titles[j] if j < 2 else f"reference", fontsize=9)
        if j == 0: ax.set_ylabel(lab, fontsize=8.5, labelpad=3)
        if j == 2: ax.text(0.97, 0.05, refl, transform=ax.transAxes, ha="right", fontsize=7.5, color=INK,
                           bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
    rr = r_(A[0, 0], A[1, 0]) if kind == "vel" else r_(m, ref)   # velocity-response r (as the bars); density r for expansion
    J["a"][lab] = rr
    ax = fig.axes[-2]; ax.text(0.97, 0.05, f"$r$ = {rr:.2f}", transform=ax.transAxes, ha="right", fontsize=8,
                                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
fig.text(0.008, 0.955, "a", fontsize=11, fontweight="bold")

# ---------------- d (top right): knockout map of MOTION-NS (differentiation evidence) + D / AUC (computed in
# 05_code/figs: fx_common.criteria C2 over the always-on heads, align_metrics AUC; six families incl. CFDBench)
import sys as _sys
_sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from fx_common import load as _load, traj as _traj
_z = _load("NSS_zsc", "stats.npz")
FAMS = ["shallow_water", "com_ns", "incom_ns", "cfdbench", "pdearena_ns", "pdearena_uncond"]
FL = ["SWE", "Comp. NS", "Incomp. NS", "CFDBench", "PDEArena", "PDEArena (u)"]
HEADS = [("advection", "advection"), ("shock", "upwind transport"), ("diffusion", "diffusion"), ("dilatation", "dilatation"),
         ("buoyancy", "buoyancy"), ("compressible", "density coupling"), ("gradvec", "directional gradient"),
         ("spectral", "spectral nonlocal"), ("coefficient", "coeff. diffusion"), ("geometry", "geometry"),
         ("generic", "generic"), ("warp_off", "transport carrier")]
K = np.full((len(HEADS), len(FAMS)), np.nan)
for i, (h, _) in enumerate(HEADS):
    cfg = h if h == "warp_off" else "ko:" + h
    for j, fam in enumerate(FAMS):
        if f"{cfg}|{fam}|phys" in _z.files:
            K[i, j] = 100 * (_traj(_z, cfg, fam).mean() / _traj(_z, "base", fam).mean() - 1)
axd = fig.add_axes([0.805, 0.175, 0.19, 0.75])
from matplotlib.colors import LogNorm
im = axd.imshow(np.clip(K, 1, None), cmap="Blues", norm=LogNorm(1, 3000), aspect="auto")
for i in range(K.shape[0]):
    for j in range(K.shape[1]):
        if np.isfinite(K[i, j]):
            axd.text(j, i, f"{K[i, j]:.0f}", ha="center", va="center", fontsize=6.2,
                     color="white" if K[i, j] > 300 else INK)
axd.set_xticks(range(len(FAMS))); axd.set_xticklabels(FL, rotation=45, ha="right", fontsize=7.5)
axd.set_yticks(range(len(HEADS))); axd.set_yticklabels([l for _, l in HEADS], fontsize=7.5)
axd.set_title("knockout damage (%), MOTION-NS", fontsize=9, pad=4, loc="right")
fig.text(0.995, 0.012, "$D$ = 0.35 (physics ref. 0.38); AUC = 0.64 ($p$ = 0.018)", fontsize=7.5, ha="right", va="bottom")
fig.text(0.652, 0.955, "b", fontsize=11, fontweight="bold")
J["d"] = {h: dict(zip(FAMS, map(lambda v: None if not np.isfinite(v) else round(float(v), 1), K[i]))) for i, (h, _) in enumerate(HEADS)}


for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figS_ns_control_ab.{ext}"))
print("FIGS_NS_CONTROL_AB_DONE", {k: round(v, 2) for k, v in J["a"].items()})
