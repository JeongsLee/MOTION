"""Horizon / reduced-history test (revision R1; A11). Models trained with 10 observed frames + 10-step rollout are
evaluated with (t_obs, k) = (10,10), (5,15), (2,18) on the autoregressive families (single-shot-mode families
diff_react and Wave-Layer excluded for all models, since the learned-readout variant has no readout beyond k=10).
Inputs : 03_analyses/A11_decoder_readout/data/horizon/<TAG>/h<t>_k<k>.npz
Outputs: 04_response/figs/hz_curves.pdf, 04_response/tables/hz_summary.tex, hz_numbers.json
  python fig_horizon.py      (re-run as more runs finish; missing cells are marked pending)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
H = f"{R}/results/A11_decoder_readout/data/horizon"
FIG = f"{R}/figures/out"; TAB = f"{R}/figures/out"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
rng = np.random.default_rng(0)
MODELS = [("BR", "typed + ADA (B-R)", "#1f5f9e"), ("V1a", "generic heads (V1a)", "#d0672a"),
          ("V2", "no metadata (V2)", "#55a868"), ("V3", "learned decoder (V3)", "#7a7a7a"),
          ("BRW", "live warp (B-R-W)", "#8172b2")]
SETS = [(10, 10), (5, 15), (2, 18)]
SS = {"diff_react", "Wave-Layer"}
def load(t, h, k):
    p = f"{H}/{t}/h{h}_k{k}.npz"; return np.load(p) if os.path.exists(p) else None
def traj(Z, f):
    e, w = Z[f"base|{f}|phys"], Z[f"base|{f}|w"]; return ((e * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)).mean(1)
def lead(Z, f):
    e, w = Z[f"base|{f}|phys"], Z[f"base|{f}|w"]; return (e * w).sum((0, 1)) / np.maximum(w.sum((0, 1)), 1e-6), w.sum((0, 1)) > 0
NUM = {}
for h, k in SETS:
    ref = load("BR", h, k)
    if ref is None: continue
    fams = sorted({x.split("|")[1] for x in ref.files if x.startswith("base|")} - SS)
    cell = {}
    for t, _, _ in MODELS:
        Z = load(t, h, k)
        if Z is None or not all(f"base|{f}|phys" in Z.files for f in fams): continue
        ca = float(np.mean([100 * traj(Z, f).mean() for f in fams]))
        if t == "BR":
            cell[t] = {"ca": ca}; continue
        dd = [100 * (traj(Z, f) - traj(ref, f)) for f in fams]
        boots = np.mean([x[rng.integers(0, len(x), (2000, len(x)))].mean(1) for x in dd], 0)
        d = [100 * (traj(Z, f) - traj(ref, f)).mean() for f in fams]
        cell[t] = {"ca": ca, "d": float(np.mean(d)), "ci": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
                   "n_worse": int(sum(x > 0 for x in d)), "n": len(d)}
    NUM[f"{h}_{k}"] = {"fams": fams, "cells": cell}
# z0 (IC consistency) at (10,10)
z0 = {}
for t, _, _ in MODELS:
    Z = load(t, 10, 10)
    if Z is None or not any(x.startswith("z0|") for x in Z.files): continue
    fams = sorted({x.split("|")[1] for x in Z.files if x.startswith("z0|")} - SS)
    z0[t] = {f: float(100 * traj_z) for f in fams for traj_z in [np.mean(((Z[f"z0|{f}|phys"] * Z[f"z0|{f}|w"]).sum(-1) / np.maximum(Z[f"z0|{f}|w"].sum(-1), 1e-6)))]}
if "BR" in z0 and "V3" in z0:
    gap = {f: z0["V3"][f] - z0["BR"][f] for f in z0["BR"]}
    NUM["z0"] = {"max_gap": max(gap.values(), key=abs), "max_gap_fam": max(gap, key=lambda f: abs(gap[f])),
                 "mean_abs_gap": float(np.mean(np.abs(list(gap.values())))), "BR_mean": float(np.mean(list(z0["BR"].values())))}
# table
lab = dict((t, l) for t, l, _ in MODELS)
with open(f"{TAB}/hz_summary.tex", "w") as fh:
    fh.write("\\begin{tabular}{l" + "c" * len(SETS) + "}\n\\toprule\n")
    fh.write("Observed frames / rollout steps & " + " & ".join(f"{h} / {k}" + (" (training)" if h == 10 else "") for h, k in SETS) + "\\\\\n\\midrule\n")
    for t, l, _ in MODELS:
        row = [l]
        for h, k in SETS:
            c = NUM.get(f"{h}_{k}", {}).get("cells", {}).get(t)
            if c is None: row.append("\\pending{running}"); continue
            row.append(f"{c['ca']:.2f}" if t == "BR" else f"{c['ca']:.2f} ({c['d']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}])")
        fh.write(" & ".join(row) + "\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
# curves: class-average error per lead, each model relative to B-R, per setting
fig, axs = plt.subplots(1, len(SETS), figsize=(7.0, 2.3), sharey=True)
for ax, (h, k) in zip(axs, SETS):
    ref = load("BR", h, k); ax.set_title(f"{h} observed frames, {k} steps" + (" (training)" if h == 10 else ""))
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("lead (steps)")
    if ref is None or f"{h}_{k}" not in NUM: ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center"); continue
    fams = NUM[f"{h}_{k}"]["fams"]
    for t, l, c in MODELS[1:]:
        Z = load(t, h, k)
        if t not in NUM[f"{h}_{k}"]["cells"]: continue
        rel = []
        for f in fams:
            a, m = lead(ref, f); b, _ = lead(Z, f); r = np.full(k, np.nan); r[m] = 100 * (b[m] - a[m]); rel.append(r)
        ax.plot(np.arange(1, k + 1), np.nanmean(rel, 0), color=c, lw=1.3, label=l)
    ax.axvline(10.5, color="0.6", lw=0.6, ls=":")
axs[0].set_ylabel("error minus B-R\n(class avg., % points)"); axs[0].legend(frameon=False, loc="upper left", fontsize=6)
fig.tight_layout(); fig.savefig(f"{FIG}/hz_curves.pdf"); plt.close(fig)
json.dump(NUM, open(f"{TAB}/hz_numbers.json", "w"), indent=1, default=float)
for s, v in NUM.items():
    if s == "z0": print("z0", v); continue
    print(s, {t: {kk: (round(x, 2) if isinstance(x, float) else x) for kk, x in c.items()} for t, c in v["cells"].items()})
