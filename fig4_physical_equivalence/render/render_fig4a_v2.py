"""Fig 4a: the test every PDE foundation model is put to here.

Two physically equivalent descriptions of one state -- the benchmark's own copy and a
rotated copy -- go into the same operator, and the two answers are compared after the
rotation is undone. The right column is measured, not assumed: both predictions come from
the fig4a-q32 run at the validated readout, and the map between them is their residual.

The columns sit close to the operator on purpose: the runs are short so the panel stays
nearly square and fills the height of the column it shares with panel b.

  python render_fig4a_v2.py [val_npz] [out_pdf]
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

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/jeongsulee/ens8_local/val_0.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figs/fig4a_equiv.pdf"
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


plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12.0,
                     "mathtext.fontset": "stix"})
FW, FH = 3.6, 4.0
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

XI, XO, XM = 0.170, 0.830, 0.500          # short runs, but long enough for the heads
XV1, XV2 = 0.352, 0.648
Y1, Y2, YM = 0.765, 0.250, 0.5075
S = 0.155
AR = FW / FH
SY = S * AR                               # square tiles
INK = "#22303f"
C_ID, C_TR = "#0b4f9c", "#e07b00"    # the two descriptions through the operator
C_ROT = "#1f7a5a"                     # the transform that relates them
PUR = "#7d3c98"
BW, BH = 0.104, 0.054
GAP = 0.012


def tile(cx, cy, img, rng=None):
    r = HI if rng is None else rng
    ax.imshow(img, extent=(cx - S, cx + S, cy - SY, cy + SY), origin="lower", cmap=CMAP,
              vmin=-r, vmax=r, interpolation="bicubic", zorder=2, aspect="auto")
    ax.add_patch(plt.Rectangle((cx - S, cy - SY), 2 * S, 2 * SY, fill=False, ec=INK,
                               lw=0.8, zorder=3))


def seg(p0, p1, head=False, color=INK, lw=1.2, ls="-"):
    if head:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=lw,
                                     color=color, zorder=3, shrinkA=0, shrinkB=0,
                                     linestyle=ls))
    else:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color=color, lw=lw, zorder=3,
                solid_capstyle="butt")


# ---- the two equivalent descriptions -----------------------------------------------
tile(XI, Y1, state)
tile(XI, Y2, T(state))

# ---- one operator --------------------------------------------------------------------
ax.add_patch(FancyBboxPatch((XM - BW, YM - BH), 2 * BW, 2 * BH,
                            boxstyle="round,pad=0.010,rounding_size=0.018",
                            fc="#eef2f6", ec=INK, lw=1.1, zorder=4))
ax.text(XM, YM + 0.015, "PDE", ha="center", va="center", fontsize=11.8, color=INK,
        zorder=5)
ax.text(XM, YM - 0.021, "foundation", ha="center", va="center", fontsize=11.8,
        color=INK, zorder=5)

DY = 0.026
for y_row, col, sgn in ((Y1, C_ID, +1), (Y2, C_TR, -1)):
    yb = YM + sgn * DY
    seg((XI + S, y_row), (XV1, y_row), color=col)
    seg((XV1, y_row), (XV1, yb), color=col)
    seg((XV1, yb), (XM - BW - GAP, yb), head=True, color=col)

# ---- the two measured answers ----------------------------------------------------------
tile(XO, Y1, pred_id, rng=HI_P)
tile(XO, Y2, T(pred_rot_back), rng=HI_P)
for y_row, col, sgn in ((Y1, C_ID, +1), (Y2, C_TR, -1)):
    yb = YM + sgn * DY
    seg((XM + BW + GAP, yb), (XV2, yb), color=col)
    seg((XV2, yb), (XV2, y_row), color=col)
    seg((XV2, y_row), (XO - S, y_row), head=True, color=col)

# ---- left leg: the transform, labelled on its left --------------------------------------
ax.add_patch(FancyArrowPatch((XI, Y1 - SY - 0.014), (XI, Y2 + SY + 0.014),
                             arrowstyle="-|>", mutation_scale=7, lw=1.5, color=C_ROT,
                             zorder=3))
ax.text(XI - 0.022, YM, "e.g. rotation", color=INK, fontsize=12.0, rotation=90,
        ha="right", va="center")

# ---- the residual, after the rotation is undone -------------------------------------------
dif = np.abs(pred_rot_back - pred_id)
dw = 0.052
ax.imshow(dif, extent=(XO - dw, XO + dw, YM - dw * AR, YM + dw * AR), origin="lower",
          cmap="magma", vmin=0.0, vmax=float(np.percentile(dif, 99)), zorder=4,
          interpolation="bicubic", aspect="auto")
ax.add_patch(plt.Rectangle((XO - dw, YM - dw * AR), 2 * dw, 2 * dw * AR, fill=False,
                           ec=PUR, lw=0.9, zorder=5))
ax.text(XO + dw + 0.024, YM, "residual", color=INK, fontsize=11.6, rotation=90,
        ha="left", va="center")
for _y0, _y1 in ((Y1 - SY - 0.016, YM + dw * AR + 0.018),
                 (Y2 + SY + 0.016, YM - dw * AR - 0.018)):
    seg((XO, _y0), (XO, _y1), head=True, color=PUR, lw=1.1, ls=(0, (3, 2)))

# ---- headings and captions -----------------------------------------------------------------
ax.text(XI, Y1 + SY + 0.024, "observed,  $t=T_0$", ha="center", fontsize=12.0, color=INK,
        va="bottom")
ax.text(XO, Y1 + SY + 0.024, "predicted,  $t=T_1$", ha="center", fontsize=12.0, color=INK,
        va="bottom")
YCAP = Y2 - SY - 0.042
ax.text(XI, YCAP - 0.026, "equivalent transform", color=INK, fontsize=12.0,
        ha="center", va="top")
ax.text(XI, YCAP - 0.086, "flip $\\cdot$ rotation $\\cdot$ translation", color=INK,
        fontsize=10.8, ha="center", va="top")
ax.text(XO, YCAP - 0.030, "equivalence under", color=INK, fontsize=12.0,
        ha="center", va="top")
ax.text(XO, YCAP - 0.082, "the inverse transform", color=INK, fontsize=12.0,
        ha="center", va="top")

fig.savefig(OUT, bbox_inches="tight", dpi=600)
fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=600)
print("wrote", OUT)
