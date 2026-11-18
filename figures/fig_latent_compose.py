"""Latent semantic composition with a numerical reference (revision R1-3; A19). Data: latent_compose.py.
  python fig_latent_compose.py NPZ TAG [AMP]
a  vorticity change caused by adding a vortex to a NS-Sines flow (held-out trajectory 0, step 1 and step 10):
   solver S(a+b) - S(a); model with the real composite input P(a+b) - P(a); anchor a+b with the latent of a only;
   anchor a+b with the composed latent h(a) + [h(c+b) - h(c)]; anchor a with the composed latent (pure insertion).
b  correlation with the solver's vortex effect (dots = trajectories; step 1 and 10).
c  latent additivity: cos between h(a+b) - h(a) and the component h(b) - h(0) (at rest) or h(c+b) - h(c) (other
   flow), whole latent / inside the vortex disk / component moved elsewhere.
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = f"{R}/figures/out"
NPZ, TAG = sys.argv[1], sys.argv[2]; AMP = sys.argv[3] if len(sys.argv) > 3 else "3"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                     "font.size": 6, "axes.titlesize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                     "legend.fontsize": 5.5, "pdf.fonttype": 42, "axes.linewidth": 0.5})
z = np.load(NPZ, allow_pickle=True); meta = json.loads(str(z["__meta__"])); rep = meta["report"][f"a{AMP}"]
K = int(meta["args"]["K"]); steps = sorted({1, K})
MAPS = [("solver", "solver\nS(a+b) − S(a)"), ("model", "model, real input\nP(a+b) − P(a)"),
        ("i_anchor_only", "anchor a+b,\nlatent of a only"), ("i_other", "anchor a+b,\ncomposed latent"),
        ("ii_other", "anchor a,\ncomposed latent")]
fig = plt.figure(figsize=(7.2, 3.0))
gm = fig.add_gridspec(2, len(MAPS), left=0.06, right=0.62, top=0.84, bottom=0.04, wspace=0.05, hspace=0.06)
for r_, si in enumerate((0, 1)):
    ref = z[f"a{AMP}|solver|w"][si]; vm = np.abs(ref).max()
    for j, (k, lab) in enumerate(MAPS):
        A = z[f"a{AMP}|{k}|w"][si] if k not in ("solver", "model") else z[f"a{AMP}|{k}|w"][si]
        ax = fig.add_subplot(gm[r_, j]); ax.imshow(A.T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.3)
        if r_ == 0: ax.set_title(lab, fontsize=5.6)
        if j == 0: ax.set_ylabel(f"step {steps[si] if si < len(steps) else K}", fontsize=6)
fig.text(0.06, 0.95, f"a  vorticity change from adding a vortex (peak {AMP}× rms u) to a NS-Sines flow", fontweight="bold", fontsize=6.5)

CONDS = [("model", "real input"), ("i_anchor_only", "anchor only"), ("i_rest", "+ comp. (rest)"), ("i_other", "+ comp. (other)"),
         ("i_random", "+ random"), ("ii_other", "insertion"), ("ii_flip", "insertion, flip"), ("ii_random", "insertion, random")]
ax = fig.add_axes([0.70, 0.60, 0.29, 0.28])
for q, (k, lab) in enumerate(CONDS):
    for s_, (st, col) in enumerate(((1, "#0072B2"), (K, "#D55E00"))):
        if k == "model":
            v = np.array([t["model_vs_solver"][str(st)]["r"] for t in rep])
        else:
            v = np.array([t["conds"][k][str(st)]["r_solver"] for t in rep])
        x = q + (s_ - 0.5) * 0.36
        ax.bar(x, v.mean(), 0.34, color=col, label=f"step {st}" if q == 0 else None)
        ax.scatter(np.full(len(v), x), v, s=2, color="k", lw=0, zorder=3)
ax.axhline(0, color="0.5", lw=0.4); ax.set_ylim(-0.3, 1); ax.set_xticks(range(len(CONDS)))
ax.set_xticklabels([c[1] for c in CONDS], rotation=55, ha="right"); ax.legend(frameon=False, loc="upper right")
ax.set_title("b  correlation with the solver's vortex effect", loc="left", fontweight="bold")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

ax = fig.add_axes([0.70, 0.02, 0.29, 0.20])
ADD = [("cos", "whole latent"), ("cos_disk", "vortex disk"), ("cos_misplaced", "component moved")]
for q, (k, lab) in enumerate(ADD):
    for s_, (cmp_, col) in enumerate((("rest", "#999999"), ("other", "#009E73"))):
        v = np.array([t["additivity"][cmp_][k] for t in rep]); x = q + (s_ - 0.5) * 0.36
        ax.bar(x, v.mean(), 0.34, color=col, label=("h(b) − h(0)" if cmp_ == "rest" else "h(c+b) − h(c)") if q == 0 else None)
        ax.scatter(np.full(len(v), x), v, s=2, color="k", lw=0, zorder=3)
ax.axhline(0, color="0.5", lw=0.4); ax.set_ylim(-0.3, 1); ax.set_xticks(range(len(ADD))); ax.set_xticklabels([a[1] for a in ADD])
ax.set_title("c  cos with h(a+b) − h(a)", loc="left", fontweight="bold"); ax.legend(frameon=False, loc="upper right")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/fig_latent_compose_{TAG}_a{AMP}.{ext}", dpi=300, bbox_inches="tight")
summ = {"n": len(rep), "model_vs_solver_r": {st: float(np.mean([t["model_vs_solver"][str(st)]["r"] for t in rep])) for st in (1, K)},
        "conds_r_solver": {k: {st: float(np.mean([t["conds"][k][str(st)]["r_solver"] for t in rep])) for st in (1, K)} for k, _ in CONDS if k != "model"},
        "latent_part_r": {k: {st: float(np.mean([t["conds"][k][str(st)].get("r_latent", np.nan) for t in rep])) for st in (1, K)} for k in ("i_rest", "i_other", "i_random")},
        "latent_share": float(np.mean([t["conds"]["i_anchor_only"]["1"]["latent_share"] for t in rep])),
        "additivity": {c: {k: float(np.mean([t["additivity"][c][k] for t in rep])) for k in ("cos", "cos_disk", "cos_misplaced", "frac_norm_disk_true")} for c in ("rest", "other")}}
json.dump(summ, open(f"{FIG}/fig_latent_compose_{TAG}_a{AMP}.json", "w"), indent=1); print(json.dumps(summ, indent=1))
