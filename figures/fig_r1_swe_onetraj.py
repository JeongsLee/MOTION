"""REVISION-R1 11-03 (R1 comment 4): one shallow-water test trajectory, BCAT, PROSE-FD and MOTION-NS, last predicted frame.
Columns: ground truth h, prediction, then inverse-transformed prediction minus prediction for the row reflection,
the transposition and the 8-cell shift (corrected scalar inverse), and the reflection with the previous vector inverse
(h negated; BCAT and PROSE-FD only). Titles give the relative L2 error of each inverse-transformed prediction; the
difference panels share one colour scale across models.
  python3 -I fig_r1_swe_onetraj.py <swe_onetraj3.npz> <out_prefix>"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
Z = np.load(sys.argv[1]); OUT = sys.argv[2]
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "pdf.fonttype": 42})
gt = Z["gt"]; M = ["BCAT", "PROSE-FD", "MOTION-NS"]
fig, ax = plt.subplots(3, 6, figsize=(7.4, 4.5), constrained_layout=True)
vmin, vmax = float(gt.min()), float(gt.max())
dl = max(float(np.abs(Z[f"{m}|{g}"] - Z[f"{m}|id"]).max()) for m in M for g in ("fr", "d1", "sh8"))
for r, m in enumerate(M):
    p = Z[f"{m}|id"]; e = lambda k: 100 * float(Z[f"{m}|err|{k}"])
    d = [Z[f"{m}|fr"] - p, Z[f"{m}|d1"] - p, Z[f"{m}|sh8"] - p]
    panels = [(gt, "ground truth $h$", "viridis", vmin, vmax), (p, f"prediction\n{e('id'):.2f}%", "viridis", vmin, vmax),
              (d[0], f"reflection\n{e('fr'):.2f}%", "RdBu_r", -dl, dl),
              (d[1], f"transposition\n{e('d1'):.2f}%", "RdBu_r", -dl, dl),
              (d[2], f"8-cell shift\n{e('sh8'):.2f}%", "RdBu_r", -dl, dl),
              (Z[f"{m}|fr|buggy"], f"reflection, previous\ninverse {e('fr|buggy'):.0f}%", "viridis", -vmax, vmax)
              if f"{m}|fr|buggy" in Z.files else None]
    ims = []
    for c, pn in enumerate(panels):
        if pn is None:
            ax[r, c].axis("off"); continue
        a_, t, cm, lo, hi = pn
        ims.append(ax[r, c].imshow(a_, cmap=cm, vmin=lo, vmax=hi, origin="lower"))
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([]); ax[r, c].set_title(t, pad=2, fontsize=6.3)
    for cols, i in (((0, 1), 1), ((2, 3, 4), 4), ((5,), 5))[:len(ims) - 3]:
        im = ims[i]
        cb = fig.colorbar(im, ax=[ax[r, c] for c in cols], fraction=0.05, pad=0.015, shrink=0.9); cb.ax.tick_params(labelsize=5)
    ax[r, 0].set_ylabel(m, fontsize=7.5)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
print("traj", int(Z["traj"]), "frame", int(Z["frame"]))
