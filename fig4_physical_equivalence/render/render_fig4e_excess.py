"""Fig 4b variant -- the defect RELATIVE TO THE OPERATOR'S OWN ERROR.

Same span form as render_fig4e_span.py, but the quantity is eps_equiv / (own error) instead
of the replica accuracy ratio.  The replica ratio divides the transformation's cost by the
model's error, and the numerical floor of the cost does NOT shrink with the error, so the
most accurate family produces the largest ratio -- accuracy is penalised.  eps and the error
share that floor, so eps/err does not blow up: 1.0 means the two physically identical
descriptions disagree by exactly as much as the operator is already wrong by, which is the
paper's interpretive rule stated as a number.
"""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
for _f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _p = f"/mnt/c/Windows/Fonts/{_f}"
    if os.path.exists(_p): fm.fontManager.addfont(_p)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import importlib.util
spec = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "render_fig4e_scatter.py"))
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "figs")
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 8.5, "axes.linewidth": 0.8, "savefig.dpi": 400})
# (transform, family, own error %, defect %) -- identical table to the scatter panel
exec(open(os.path.join(HERE, "render_fig4e_scatter.py")).read().split("COL = {")[0]
     .split("fig, ax")[0].replace("import matplotlib as mpl", "").replace('mpl.use("Agg")', ""), globals())
COL = {"MOTION": "#1a4f8a", "Poseidon-B": "#d97706", "BCAT": "#c0392b", "PROSE-FD": "#8f5b2a"}
MK = {"mirror": "o", "rotation": "s", "translation": "^"}
NT = {"SWE": 100, "Com": 100, "Incom": 100, "CFDbench": 100, "PDEarena": 100,
      "PDEarena-u": 100, "NS-PwC": 128, "ACE": 128, "Poisson": 128, "Wave": 128}
ORDER = ["MOTION", "Poseidon-B", "BCAT", "PROSE-FD"]
fig, ax = plt.subplots(figsize=(3.7, 3.05))
fig.subplots_adjust(left=0.235, right=0.985, top=0.90, bottom=0.16)
# A single reference at 1: the defect equals the operator's own prediction error.  It is a
# reading aid rather than a bound -- the triangle inequality allows a symmetry-carrying
# operator to reach 2 -- so the caption calls it a reference and not a threshold.
ax.axvline(1.0, color="0.35", lw=0.9, ls=(0, (4, 3)), zorder=1)
DY = {"mirror": 0.16, "rotation": 0.0, "translation": -0.16}
for i, m in enumerate(ORDER):
    y = len(ORDER) - 1 - i
    v = [eps / e for _, _, e, eps in D[m]]
    ax.plot([min(v), max(v)], [y, y], "-", color=COL[m], lw=2.6, alpha=0.30,
            solid_capstyle="round", zorder=2)
    for (t, fam, e, eps) in D[m]:
        _r = eps / e
        # faded below the reference (defect smaller than the operator's own error), solid
        # above it (the equivalent description costs more than the model is already wrong by)
        ax.plot(_r, y + DY[t], MK[t], ms=4.2, mfc=COL[m], mec="white", mew=0.55,
                alpha=1.0 if _r >= 1.0 else 0.35, zorder=4 if _r >= 1.0 else 3,
                clip_on=False)
    hi = max(v)
    ax.text(hi * 1.3, y, f"$\\times{hi:.2f}$" if hi < 10 else f"$\\times{hi:.0f}$",
            fontsize=7.2, color=COL[m], va="center")
    fams = sorted({fam for _, fam, _, _ in D[m]})
    ax.text(-0.012, y, f"{m}\n{len(fams)} families\n{sum(NT[f] for f in fams):,} trajectories",
            transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.6)
ax.set_xscale("log"); ax.set_xlim(0.09, 4000)
ax.set_xticks([0.1, 1, 10, 100, 1000]); ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
ax.minorticks_off(); ax.set_yticks([]); ax.set_ylim(-0.72, len(ORDER) - 0.18)
ax.set_xlabel("equivariance defect / prediction error")
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="x", length=3, width=0.8)
ax.legend(handles=[Line2D([], [], marker=MK[t], ls="none", ms=4.4, mfc="0.45", mec="white",
                          mew=0.55, label=l) for t, l in (("mirror", "mirrors"),
                          ("rotation", "$90^\\circ$ rotations"), ("translation", "translations"))],
          loc="upper right", bbox_to_anchor=(1.005, 0.97), frameon=False, fontsize=7.0,
          handletextpad=0.3, labelspacing=0.24)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig4e_excess.{ext}"), bbox_inches="tight")
print("wrote fig4e_excess.pdf")
for m in ORDER:
    v = [eps / e for _, _, e, eps in D[m]]
    print(f"  {m:11s} n={len(v):2d}  {min(v):.2f} - {max(v):7.2f}")
