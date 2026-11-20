"""Fig. 4c (file name from development: fig4d), vertical layout (REVISION-R1 10-30, user): the equivalence test as one column so that panel d can sit
beside it at a larger width. Top row: the observed state u0 and its transformed equivalent Phi u0; both pass through
one operator; middle row: the two predictions; bottom: the equivariance defect |Phi^-1 G(Phi u0) - G(u0)|.
Same data and colour code as render_fig4a_v2.py (horizontal version, kept).
  python render_fig4d_vert.py [val_npz] [out_pdf]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm

for _f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _p = f"/mnt/c/Windows/Fonts/{_f}"
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SRC = sys.argv[1] if len(sys.argv) > 1 else "ens8_local/val_0.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures/out/fig4d_vert.pdf"
HERE = os.path.dirname(os.path.abspath(__file__))

f = np.load(SRC)["fields"]
zc = f.shape[3] // 2
_st = int(f.shape[1] // 32)
state = np.asarray(f[10, ::_st, ::_st, zc, 3], np.float32)[:32, :32]

_rd = os.path.join(HERE, "refdata")
_f3 = os.path.join(_rd, "fig4a_slices32_f3.npz")
_S = np.load(_f3 if os.path.exists(_f3) else os.path.join(_rd, "fig4a_slices32.npz"))
pred_id = np.asarray(_S["pred_id"], np.float32)
pred_rot_back = np.asarray(_S["pred_r18_back"], np.float32)

state = state - state.mean()
pred_id = pred_id - pred_id.mean()
pred_rot_back = pred_rot_back - pred_rot_back.mean()
HI = float(np.percentile(np.abs(state), 97))
HI_P = float(np.percentile(np.abs(pred_id), 97))
CMAP = "RdBu_r"


def T(a):
    return np.rot90(a)



plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Times New Roman", "font.size": 10.5, "mathtext.fontset": "stix"})
W, H = 1.85, 3.55
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, W); ax.set_ylim(0, H)
INK = "#22303f"; C_ID, C_TR = "#0b4f9c", "#e07b00"; C_ROT = "#1f7a5a"; PUR = "#7d3c98"
h = 0.27                                   # tile half-size (in)
XL, XR, XC = 0.60, 1.42, 1.01
Y0, Y1, YD = 2.95, 1.45, 0.52               # observed, predicted, defect rows
YB = 2.20                                   # operator box centre


def tile(cx, cy, img, rng, cmap=CMAP, ec=INK, vmin=None):
    ax.imshow(img, extent=(cx - h, cx + h, cy - h, cy + h), origin="lower", cmap=cmap,
              vmin=-rng if vmin is None else vmin, vmax=rng, interpolation="bicubic", zorder=2, aspect="auto")
    ax.add_patch(plt.Rectangle((cx - h, cy - h), 2 * h, 2 * h, fill=False, ec=ec, lw=0.7, zorder=3))


def path(pts, color, head=True, lw=1.0, ls="-"):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=3, solid_capstyle="butt")
    a, b = pts[-2], pts[-1]
    if head:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=lw, color=color, zorder=3,
                                     shrinkA=0, shrinkB=0, linestyle=ls))
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=3)


# row labels (left margin)
for y, s in ((Y0, "observed, $t=T_0$"), (Y1, "predicted, $t=T_1$")):
    ax.text(0.10, y, s, rotation=90, ha="center", va="center", color=INK, fontsize=10.5)

# observed: u0 and its transformed equivalent
tile(XL, Y0, state, HI); tile(XR, Y0, T(state), HI)
ax.add_patch(FancyArrowPatch((XL + h + 0.03, Y0), (XR - h - 0.03, Y0), arrowstyle="-|>", mutation_scale=6,
                             lw=1.2, color=C_ROT, zorder=3))
ax.text(XC, Y0 + 0.05, "$\\Phi$", color=C_ROT, fontsize=13, ha="center", va="bottom")
ax.text(XC, Y0 + h + 0.07, "flip $\\cdot$ rotation $\\cdot$ translation", color=INK, fontsize=8.6, ha="center",
        va="bottom")

# one operator
BW, BH = 0.52, 0.17
ax.add_patch(FancyBboxPatch((XC - BW, YB - BH), 2 * BW, 2 * BH, boxstyle="round,pad=0.01,rounding_size=0.04",
                            fc="#eef2f6", ec=INK, lw=0.9, zorder=4))
ax.text(XC, YB, "PDE\nfoundation model", ha="center", va="center", fontsize=9.4, color=INK, zorder=5,
        linespacing=1.0)
for x, col, xb in ((XL, C_ID, XC - 0.20), (XR, C_TR, XC + 0.20)):
    path([(x, Y0 - h - 0.02), (x, YB + BH + 0.10), (xb, YB + BH + 0.10), (xb, YB + BH + 0.015)], col)
    path([(xb, YB - BH - 0.015), (xb, YB - BH - 0.10), (x, YB - BH - 0.10), (x, Y1 + h + 0.02)], col)

# predictions: G(u0) and G(Phi u0) in the transformed frame
tile(XL, Y1, pred_id, HI_P); tile(XR, Y1, T(pred_rot_back), HI_P)

# equivariance defect: the transformed answer is mapped back with Phi^-1 and compared
dif = np.abs(pred_rot_back - pred_id)
tile(XC, YD, dif, float(np.percentile(dif, 99)), cmap="magma", ec=PUR, vmin=0.0)
path([(XL, Y1 - h - 0.02), (XL, YD), (XC - h - 0.025, YD)], PUR, ls=(0, (3, 2)))
path([(XR, Y1 - h - 0.02), (XR, YD), (XC + h + 0.025, YD)], PUR, ls=(0, (3, 2)))
ax.text(XR + 0.05, 0.5 * (Y1 - h + YD), "$\\Phi^{-1}$", color=C_ROT, fontsize=13, ha="left", va="center")
ax.text(XC, YD - h - 0.05, "equivariance defect", color=INK, fontsize=10.5, ha="center", va="top")

fig.savefig(OUT, bbox_inches="tight", dpi=600)
fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=600)
print("wrote", OUT)
