"""Fig. 1c (revision): knockout map of MOTION-S on the TEST split (128 trajectories per family, full 10-frame rollout,
physical units; ev20-m1f-stats) and family-specific contribution (ev20-m1f-contrib). Replaces the submitted panel,
which used MOTION on 2-4 validation trajectories at lead 5 (A02).
Knockout score = (KO error - full error) / (persistence error - full error), clipped to [0, 1], as in the submitted
panel. Cells whose paired 95% bootstrap interval of the error increase includes zero are drawn white; grey = head
closed by the metadata gate for that family.
  python fig1c_m1f.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = f"{R}/results/A14_factorial/data/m1f"; FIG = f"{R}/figures/out"
plt.rcParams.update({"font.family": "serif",
                     "font.size": 6, "pdf.fonttype": 42, "axes.linewidth": 0.4})
# REVISION-R1 11-01: contribution shares from 128 test trajectories per family (ev20-m1f-contrib128), as the knockout row
Z = np.load(f"{D}/stats.npz"); CJ = json.load(open(f"{D}_c128/contrib.json"))
RNG = np.random.default_rng(0)
PERS = {"shallow_water": 3.33, "diff_react": 13.64, "com_ns": 4.36, "incom_ns": 6.93, "pdearena_ns": 67.55,
        "pdearena_uncond": 44.90, "CE-RP": 35.38, "CE-CRP": 44.89, "CE-KH": 6.67, "CE-Gauss": 16.70, "CE-RM": 34.62,
        "NS-Sines": 86.96, "NS-Gauss": 55.70, "ACE": 24.61, "Wave-Layer": 154.31, "Poisson-Gauss": 100.0,
        "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": 9.78, "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": 21.52,
        "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": 49.90}
MLAB = {"advection": "advection", "shock": "upwind transport", "diffusion": "diffusion", "wave": "wave-Laplacian",
        "buoyancy": "gravity gradient", "compressible": "density coupling", "vortex": "vortex stretching",
        "reaction_nl": "nonlinear reaction", "spectral": "spectral nonlocal", "geometry": "mean curvature",
        "gradvec": "directional gradient", "elliptic": "domain-mean coupling", "boundary_layer": "near-wall coupling",
        "dilatation": "dilatational coupling", "bernoulli": "dynamic pressure", "coefficient": "coefficient-weighted diffusion",
        "kinematic": "velocity-gradient invariants", "shear": "first-axis gradient", "reaction": "linear reaction"}
SEL = ["advection", "reaction_nl", "wave", "shock", "compressible", "buoyancy", "diffusion", "vortex", "spectral",
       "geometry", "gradvec"]
# REVISION-R1 11-01 (user): FIG1C_FULL=1 renders the Supplementary full-vocabulary version (all nineteen gated heads,
# same data, ordering of the main panel first, normalization unchanged) as fig1c_full.{pdf,png}
FULL = os.environ.get("FIG1C_FULL") == "1"
if FULL:
    SEL = SEL + [h for h in MLAB if h not in SEL]
YB = 11 / len(SEL)                                    # bracket offsets scale with the panel height
BLOCKS = [(["incom_ns", "NS-Sines", "NS-Gauss"], "incompressible"), (["ACE", "diff_react"], "reaction"),
          (["shallow_water", "Wave-Layer", "Poisson-Gauss"], "wave / steady"), (["CE-RP", "CE-RM"], "shock"),
          (["com_ns", "CE-CRP", "CE-Gauss", "CE-KH"], "compressible Euler"), (["pdearena_ns", "pdearena_uncond"], "buoyant"),
          (["pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08", "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08",
            "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08"], "3D compressible")]
FS = {"shallow_water": "SWE", "diff_react": "Diff-React", "com_ns": "Comp. NS", "incom_ns": "Incomp. NS",
      "pdearena_ns": "PDEArena", "pdearena_uncond": "PDEArena-U", "Poisson-Gauss": "Poisson",
      "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08": "3D M0.1", "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08": "3D M1.0",
      "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08": "3D Turb"}
FAMS = [f for b, _ in BLOCKS for f in b]


def err(cfg, f):                                      # per-trajectory error (frame-weighted mean), fraction
    e = Z[f"{cfg}|{f}|phys"][:, 0]; w = Z[f"{cfg}|{f}|w"][:, 0]
    return (e * w).sum(1) / w.sum(1)


def scale_head(h):                                    # 'vortex' is stored as ko:vortex
    return h


S = np.full((len(SEL), len(FAMS)), np.nan); SIG = np.zeros_like(S, bool); CLOSED = np.zeros_like(S, bool)
base = {f: err("base", f) for f in FAMS}
for j, f in enumerate(FAMS):
    b = base[f]; head = max(PERS[f] / 100.0 - b.mean(), 1e-3)
    lab = CJ["labels"].get(f, {})
    for i, h in enumerate(SEL):
        if f"ko:{h}|{f}|phys" not in Z.files: continue
        d = err(f"ko:{h}", f) - b
        m = RNG.choice(d, (4000, len(d))).mean(1); lo, hi = np.percentile(m, [2.5, 97.5])
        S[i, j] = np.clip(d.mean() / head, 0, 1); SIG[i, j] = lo > 0
        CLOSED[i, j] = (h in lab and lab[h] == 0)
# family-specific contribution (log2 enrichment of a head's share over its median share, as in the submitted panel)
heads_all = sorted({h for f in FAMS for h in CJ["C"][f] if h != "H0"})
CN = np.array([[CJ["C"][f].get(h, 0.0) for f in FAMS] for h in heads_all])
share = CN / (CN.sum(0, keepdims=True) + 1e-12); den = np.maximum(np.median(share, 1, keepdims=True), 0.01)
Rr = np.clip(np.log2(np.maximum(share, 1e-9) / den), 0, None); CNn = Rr / (Rr.max(0, keepdims=True) + 1e-12)
CNs = np.array([CNn[heads_all.index(h)] if h in heads_all else np.zeros(len(FAMS)) for h in SEL])

# REVISION-R1 10-29 (user): larger type at the full-width placement (7.2x3.6 in at 6 pt -> 5.4x3.5 in, 6.4-7.2 pt);
# title without "(test, 128 trajectories)" (in the caption / Methods)
fig, axes = plt.subplots(2, 1, figsize=(5.4, 3.5 if not FULL else 5.9), sharex=True, gridspec_kw={"hspace": 0.22})
for ax, M, cm, title in ((axes[0], S, "Reds", "knockout damage"),
                         (axes[1], CNs, "Blues", "family-specific contribution")):
    A = np.where(np.isnan(M), 0, M)
    if ax is axes[0]: A = np.where(SIG, A, 0)
    im = ax.imshow(A, aspect="auto", cmap=cm, vmin=0, vmax=1)
    gy = np.ma.masked_where(~CLOSED, np.ones_like(A))
    # REVISION-R1 10-29 (user): submitted-panel grey (0.91; was #d0d0d0)
    ax.imshow(gy, aspect="auto", cmap=ListedColormap(["#e8e8e8"]), vmin=0, vmax=1)
    ax.set_yticks(range(len(SEL))); ax.set_yticklabels([MLAB[h] for h in SEL], fontsize=6.8)
    ax.axvline(len(FAMS) - 3.5, color="k", lw=0.7); ax.set_title(title, fontsize=7.2, pad=1.5)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.008)
axes[1].set_xticks(range(len(FAMS))); axes[1].set_xticklabels([FS.get(f, f) for f in FAMS], rotation=90, fontsize=6.6)
tr = axes[1].get_xaxis_transform(); c0 = 0
for blk, lab in BLOCKS:
    i0, i1 = c0, c0 + len(blk) - 1; c0 += len(blk)
    axes[1].plot([i0 - .38, i1 + .38], [-0.66 * YB, -0.66 * YB], transform=tr, clip_on=False, color="0.25", lw=0.7)
    WRAP = {"incompressible": "incom-\npressible", "compressible Euler": "compressible\nEuler", "3D compressible": "3D\ncompressible"}
    axes[1].text((i0 + i1) / 2, -0.69 * YB, WRAP.get(lab, lab), transform=tr, clip_on=False, ha="center", va="top", fontsize=6.4,
                 color="0.25", linespacing=1.0)
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/{'fig1c_full' if FULL else 'fig1c_m1f'}.{ext}", dpi=300, bbox_inches="tight")
if FULL:
    raise SystemExit(0)
summ = {"base_err_pct": {f: round(100 * float(base[f].mean()), 3) for f in FAMS},
        "n_sig_cells": int(SIG.sum()), "top_ko": {}}
for j, f in enumerate(FAMS):
    rows = [(float(err(f"ko:{h}", f).mean() / base[f].mean() - 1) * 100, h) for h in
            [k.split("|")[0][3:] for k in Z.files if k.startswith("ko:") and k.endswith(f"|{f}|phys") and "+" not in k]]
    rows.sort(reverse=True); summ["top_ko"][f] = [(h, round(v, 1)) for v, h in rows[:3]]
json.dump(summ, open(f"{FIG}/fig1c_m1f.json", "w"), indent=1); print(json.dumps(summ)[:1500])
