"""Thumbnails for the Fig. 1a solution-sample cards.

The cards illustrate that the detached instance is queried across families, so they carry
GROUND TRUTH only -- no prediction, no model.  Families are chosen to not overlap panel b
(ACE, CE-RM, incom_ns, Wave-Layer, 3D M1.0 Rand, 3D Turb), so the schematic and the
qualitative panel do not show the same systems twice.

2D fields render as filled contours; the 3D field renders as a density isosurface at the
same viewing angle panel b uses, so the card set reads as "one operator, many families,
two and three dimensions".

usage: python3 make_fig1a_cards.py <gtcards.npz> <out_dir>
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/motion_gtcards/gtcards.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/e/motion_gtcards"
CMAP = {"CE-KH": "RdBu_r", "pdearena_ns": "magma", "shallow_water": "viridis"}
PX = 2.0            # inches; the card is ~11 mm wide in the figure, so 300 dpi is ample


def card2d(a, name, cmap):
    fig = plt.figure(figsize=(PX, PX))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    lo, hi = np.percentile(a, [1, 99])
    ax.contourf(a.T, levels=14, cmap=cmap, vmin=lo, vmax=hi)
    p = os.path.join(OUT, f"card_{name}.png")
    fig.savefig(p, dpi=300, transparent=True); plt.close(fig)
    return p


def card3d(v, name):
    from skimage import measure
    lo, hi = np.percentile(v, [2, 98])
    lev = lo + 0.55 * (hi - lo)
    verts, faces, _, _ = measure.marching_cubes(np.ascontiguousarray(v), level=lev)
    fig = plt.figure(figsize=(PX, PX))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off(); ax.set_position([0, 0, 1, 1])
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    color="#2E6B36", alpha=0.9, linewidth=0, antialiased=True)
    n = v.shape[0]
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_zlim(0, n)
    ax.view_init(elev=18, azim=-58)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    p = os.path.join(OUT, f"card_{name}.png")
    fig.savefig(p, dpi=300, transparent=True); plt.close(fig)
    return p


def main():
    Z = np.load(SRC)
    os.makedirs(OUT, exist_ok=True)
    for k in Z.files:
        a = np.asarray(Z[k], np.float64)
        short = "cns3d_M01" if k.startswith("pdebench3d") else k.replace("-", "")
        p = card3d(a, short) if a.ndim == 3 else card2d(a, short, CMAP.get(k, "viridis"))
        print(f"  {k:44s} {a.shape} -> {os.path.basename(p)}")


if __name__ == "__main__":
    main()
