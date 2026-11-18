"""REVISION-R1 11-06: one CFDBench test trajectory, BCAT, PROSE-FD and MOTION-NS, last predicted frame, first flow
channel. Columns: ground truth, prediction, inverse-transformed prediction of the row mirror across the flow axis
(input mirrored, prediction mirrored back), and their difference (shared colour scale across models). Titles: relative
L2 error with respect to the ground truth (both flow channels). Trajectory: median BCAT identity error among the first
16 test trajectories (the MOTION-NS dump).
  python3 -I fig_cfd_onetraj.py <cfd_base.npz> <motion equiv.npz> <out_prefix>"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B, A, OUT = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
f = "cfdbench"
rel = lambda a, b: float(np.sqrt(((a - b) ** 2).sum()) / (np.sqrt((b ** 2).sum()) + 1e-12))
gt_m = A[f"{f}|dump_gt"][..., :2].astype(np.float32)
for m in ("BCAT", "PROSE-FD"):                                    # same trajectories in the same order
    d = [rel(gt_m[k], B[f"{m}|gt"][k]) for k in range(len(gt_m))]
    print(m, "GT mismatch vs MOTION dump (max rel):", round(max(d), 5))
    assert max(d) < 1e-2, "trajectory mismatch"
k = int(np.argsort(B["BCAT|err_id"])[len(B["BCAT|err_id"]) // 2])
D = {"BCAT": (B["BCAT|id"][k], B["BCAT|fr_ns"][k]), "PROSE-FD": (B["PROSE-FD|id"][k], B["PROSE-FD|fr_ns"][k]),
     "MOTION-NS": (A[f"{f}|dump_pred"][k, ..., :2].astype(np.float32), A[f"{f}|dumpm_fr_ns"][k, ..., :2].astype(np.float32))}
gt = B["BCAT|gt"][k]
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "pdf.fonttype": 42})
fig, ax = plt.subplots(3, 4, figsize=(6.2, 4.6), constrained_layout=True)
vmin, vmax = float(gt[..., 0].min()), float(gt[..., 0].max())
dl = max(float(np.abs(pm[..., 0] - p[..., 0]).max()) for p, pm in D.values())
for r, (m, (p, pm)) in enumerate(D.items()):
    panels = [(gt[..., 0], "ground truth", "viridis", vmin, vmax), (p[..., 0], f"prediction\n{100 * rel(p, gt):.2f}%", "viridis", vmin, vmax),
              (pm[..., 0], f"mirror, inverse-transformed\n{100 * rel(pm, gt):.2f}%", "viridis", vmin, vmax),
              (pm[..., 0] - p[..., 0], "mirrored minus\nprediction", "RdBu_r", -dl, dl)]
    ims = []
    for c, (a_, t, cm, lo, hi) in enumerate(panels):
        ims.append(ax[r, c].imshow(a_, cmap=cm, vmin=lo, vmax=hi, origin="lower"))
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([]); ax[r, c].set_title(t, pad=2, fontsize=6.3)
    for cols, i in (((0, 1, 2), 2), ((3,), 3)):
        cb = fig.colorbar(ims[i], ax=[ax[r, c] for c in cols], fraction=0.05, pad=0.015, shrink=0.9); cb.ax.tick_params(labelsize=5)
    ax[r, 0].set_ylabel(m, fontsize=7.5)
    print(m, "id %.3f%% mirror %.3f%% |mirror-pred|/|pred| %.3f" % (100 * rel(p, gt), 100 * rel(pm, gt), rel(pm, p)))
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
print("traj", k)
