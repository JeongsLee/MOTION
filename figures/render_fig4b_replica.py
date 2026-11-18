"""Fig. 4b (R1) (revision R1, 10-22; R1-5 / R2 / R3-3): trajectory-level replica error ratio
rho = mean_g err_g / err_id (1 = equivariant) per family, median with bootstrap 95% CI, for the released weights and
their baselines on the same held-out trajectories, from 03_analyses/equiv_full/stats3*.json (stats_equiv3.py).
Families of each comparison are grouped; the paired Holm-adjusted Wilcoxon p of MOTION vs each baseline is annotated.
Left: MOTION-S on its 14 equivalence-eligible families (the six Poseidon families scored on density/velocity/pressure, as
the Poseidon-T reference); right: MOTION-NS on the NS benchmark with BCAT / PROSE-FD as reference levels. No paired
marks (the baselines differ in capacity / corpus). Writes figs/fig4b_replica_r1.{pdf,png}.
  python render_fig4b_replica_draft.py [STATS_JSON]"""
import json
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
for _f in ("times.ttf", "timesbd.ttf", "timesi.ttf"):
    _p = f"/mnt/c/Windows/Fonts/{_f}"
    if os.path.exists(_p): fm.fontManager.addfont(_p)
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
ST = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "equiv_full", "stats3_r1.json")   # REVISION-R1 10-29: + MOTION-IVP trajectory-level (gn3w_traj)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 9.0, "axes.linewidth": 0.8, "savefig.dpi": 400})
COL = {"MOTION-S (pch)": "#1a4f8a", "MOTION-NS": "#1a4f8a", "Poseidon-T": "#d97706", "BCAT": "#c0392b",
       "PROSE-FD": "#8f5b2a"}
LAB = {"MOTION-S (pch)": "MOTION-S", "MOTION-NS": "MOTION-NS"}
FL = {"shallow_water": "SWE", "com_ns": "Comp. NS", "incom_ns": "Incomp. NS", "cfdbench": "CFDBench",
      "pdearena_ns": "PDEArena", "pdearena_uncond": "PDEArena (uncond)"}
# REVISION-R1 10-28 (user): one panel per training corpus -- MOTION-S and MOTION-L (nineteen families); MOTION-IVP with
# Poseidon-T / Poseidon-B (Poseidon pretraining families); MOTION-NS with BCAT / PROSE-FD (Navier--Stokes benchmark)
GROUPS = [("Nineteen-family corpus", ["MOTION-S*", "MOTION-L*"],
           ["shallow_water", "com_ns", "incom_ns", "diff_react", "ACE", "pdearena_ns", "pdearena_uncond",
            "CE-RP", "CE-CRP", "CE-KH", "CE-Gauss", "CE-RM", "NS-Sines", "NS-Gauss"]),
          ("Poseidon pretraining corpus", ["MOTION-IVP", "Poseidon-T", "Poseidon-B"],
           ["CE-RP", "CE-CRP", "CE-KH", "CE-Gauss", "NS-Sines", "NS-Gauss"]),
          ("Navier\u2013Stokes benchmark corpus", ["MOTION-NS", "BCAT", "PROSE-FD"],
           ["shallow_water", "com_ns", "incom_ns", "cfdbench", "pdearena_ns", "pdearena_uncond"])]
POS = {"CE-RP", "CE-CRP", "CE-KH", "CE-Gauss", "NS-Sines", "NS-Gauss"}
S = json.load(open(ST)); W = S["within"]; P = S["pairs"]
# REVISION-R1 11-02 (user): the Poseidon-corpus panel on the rows held out by MOTION-IVP and Poseidon (heldout_both:
# CE x4 91, NS-Sines 76, NS-Gauss 72); the hash test rows lie largely inside Poseidon's training data
_HO = os.path.join(HERE, "..", "results", "equiv_full", "stats3_heldout.json")
W.update({m: r for m, r in json.load(open(_HO))["within"].items() if m in ("MOTION-IVP", "Poseidon-T", "Poseidon-B")})
# MOTION-S*: Poseidon channels on the Poseidon families (as the Poseidon-T reference), all scored slots elsewhere
W["MOTION-S*"] = {f: (W["MOTION-S (pch)"][f] if f in POS else r) for f, r in W["MOTION-S"].items()}
COL["MOTION-S*"] = "#1a4f8a"; LAB["MOTION-S*"] = "MOTION-S"
W["MOTION-L*"] = {f: (W["MOTION (pch)"][f] if f in POS else r) for f, r in W["MOTION"].items()}
COL["MOTION-L*"] = "#6fa3d9"; LAB["MOTION-L*"] = "MOTION-L"
COL["MOTION-IVP"] = "#2a9d8f"; LAB["MOTION-IVP"] = "MOTION-IVP"
# (superseded 10-29 by the trajectory-level run) REVISION-R1 10-28: MOTION-IVP from the equivalence-run log only (family-mean error ratio over the seven D4 members;
# the per-trajectory npz was not written) -> open marker, no interval (03_analyses/ivps_equiv_gn3w/means_from_log.json)
# _IVP = json.load(open(os.path.join(HERE, "..", "results", "ivps_equiv_gn3w", "means_from_log.json")))
# W["MOTION-IVP"] = {f: {"rho_median": v["rho_D4_mean"], "rho_ci": [v["rho_D4_mean"]] * 2} for f, v in _IVP.items()}
COL["Poseidon-B"] = "#f0b25a"   # REVISION-R1 10-22 (user): Poseidon-B reference (capacity counterpart of MOTION, Fig. 5d)
FL.update({"diff_react": "Diffusion\u2013reaction", "ACE": "Allen\u2013Cahn"})


def star(p):
    return "" if p is None else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


# REVISION-R1 10-29 (user): main Fig. 5b = the two same-corpus comparisons (Poseidon corpus, NS benchmark), side by side;
# the nineteen-family panel (MOTION-S, MOTION-L) goes to the SI:  REPLICA_SET=main|si python render_fig4b_replica.py
import sys
SET = os.environ.get("REPLICA_SET", "main")
if SET == "si":
    GROUPS = GROUPS[:1]; fig = plt.figure(figsize=(2.9, 3.6))
    gs = fig.add_gridspec(1, 1, left=0.30, right=0.97, top=0.94, bottom=0.11); axes = [fig.add_subplot(gs[0, 0])]
elif SET == "main_old":
    GROUPS = [GROUPS[2], GROUPS[1]]; fig = plt.figure(figsize=(4.3, 3.3))      # MOTION-NS panel first, then MOTION-IVP
    gs = fig.add_gridspec(1, 2, left=0.25, right=0.99, top=0.84, bottom=0.13, wspace=0.75)
    plt.rcParams.update({"font.size": 10.5})
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
# REVISION-R1 10-30 (user): main Fig. 4e = per corpus, replica error ratio rho and the error change under symmetry
# averaging (median per-trajectory gain, %; negative = averaging reduces the error) side by side on shared family rows.
# The gain axis is symlog (linear within +-10 %) with a blue (error reduced) / red (error increased) band.
if SET == "main":
    GROUPS = [GROUPS[2], GROUPS[1]]
    plt.rcParams.update({"font.size": 10.5})
    fig = plt.figure(figsize=(4.9, 3.93))
    gs = fig.add_gridspec(1, 5, left=0.14, right=0.99, top=0.80, bottom=0.14, width_ratios=[1, 1.05, 0.78, 1, 1.05],
                          wspace=0.10)
    pairs = [(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])), (fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 4]))]
    MS = 5.2; GLO, GHI = -60, 700; CNEG, CPOS = "#2166ac", "#b2182b"
    for (axr, axg), (title, models, fams) in zip(pairs, GROUPS):
        off = np.linspace(0.2, -0.2, len(models))
        MK = {m: ("o" if m.startswith("MOTION") else "s") for m in models}
        axg.axvspan(GLO, 0, color=CNEG, alpha=0.07, lw=0, zorder=0)
        axg.axvspan(0, GHI, color=CPOS, alpha=0.07, lw=0, zorder=0)
        for ax in (axr, axg):
            for i in range(len(fams)): ax.axhline(len(fams) - 1 - i, color="0.88", lw=0.7, zorder=0)
        for i, fam in enumerate(fams):
            y = len(fams) - 1 - i
            for o, m in zip(off, models):
                r = W.get(m, {}).get(fam)
                if r is None: continue
                axr.errorbar(r["rho_median"], y + o, xerr=[[r["rho_median"] - r["rho_ci"][0]], [r["rho_ci"][1] - r["rho_median"]]],
                             fmt=MK[m], ms=MS, color=COL[m], mec="white", mew=0.5, elinewidth=1.3, capsize=2.2, zorder=3)
                g = r["gain_median"]
                axg.plot([0, g], [y + o, y + o], color=COL[m], lw=1.0, alpha=0.55, zorder=2)
                axg.plot(g, y + o, MK[m], ms=MS, color=COL[m], mec="white", mew=0.5, zorder=3)
        axr.axvline(1.0, color="0.35", lw=0.8, ls=(0, (4, 3))); axg.axvline(0.0, color="0.35", lw=0.8, ls=(0, (4, 3)))
        axr.set_yticks(range(len(fams)))
        axr.set_yticklabels([FL.get(f, f).replace("PDEArena (uncond)", "PDEArena\n(uncond)") for f in fams][::-1],
                            linespacing=0.95)
        axg.set_yticks(range(len(fams))); axg.set_yticklabels([]); axg.tick_params(axis="y", length=0)
        axr.set_xscale("log"); axr.set_xlim(0.9, 12.0); axr.set_xticks([1, 2, 5]); axr.set_xticklabels(["1", "2", "5"])
        axr.minorticks_off()
        axg.set_xscale("symlog", linthresh=10, linscale=1.0); axg.set_xlim(GLO, GHI)
        axg.set_xticks([-30, 0, 100]); axg.set_xticklabels(["$-30$", "0", "100"]); axg.minorticks_off()
        for lab, col in zip(axg.get_xticklabels(), (CNEG, "0.2", CPOS)): lab.set_color(col)
        for ax in (axr, axg): ax.set_ylim(-0.6, len(fams) - 0.4)                 # shared family rows
        _tr = axg.get_xaxis_transform()                                          # colour band on the gain axis
        axg.plot([GLO, 0], [0, 0], color=CNEG, lw=3.0, solid_capstyle="butt", clip_on=False, zorder=4, transform=_tr)
        axg.plot([0, GHI], [0, 0], color=CPOS, lw=3.0, solid_capstyle="butt", clip_on=False, zorder=4, transform=_tr)
        axg.spines["bottom"].set_visible(False)
        axr.set_xlabel("replica error\nratio $\\rho$", fontsize=9.8); axg.set_xlabel("$\\Delta$ error under\naveraging (%)", fontsize=9.8)
        for ax in (axr, axg):
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        axg.spines["left"].set_visible(False)
        axr.set_title({"Navier\u2013Stokes benchmark corpus": "Navier\u2013Stokes benchmark",
                       "Poseidon pretraining corpus": "Poseidon corpus"}[title], fontsize=10.8, pad=36, loc="left")
        for m in models: axr.plot([], [], MK[m], color=COL[m], ms=MS, label=LAB.get(m, m))
        axr.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=8.8, handletextpad=0.05,
                   columnspacing=0.4, borderaxespad=0.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4b_replica_r1.{ext}"), bbox_inches="tight")
    print("wrote fig4b_replica_r1 (rho + symmetry-averaging error change)"); sys.exit(0)
for ax, (title, models, fams) in zip(axes, GROUPS):
    off = np.linspace(0.2, -0.2, len(models)) if len(models) > 1 else [0.0]
    MS = 6.5 if SET == "main" else 3.6                    # REVISION-R1 10-29 (user): larger markers in the main panel
    # REVISION-R1 10-29 (user): MOTION = circle, the two baselines = square / triangle (colours unchanged)
    MK = {m: ("o" if m.startswith("MOTION") else "s") for m in models}   # baselines of one corpus share a symbol
    for i in range(len(fams)):                            # light grey guide line per family
        ax.axhline(len(fams) - 1 - i, color="0.88", lw=0.7, zorder=0)
    for i, fam in enumerate(fams):
        y = len(fams) - 1 - i
        for o, m in zip(off, models):
            r = W.get(m, {}).get(fam)
            if r is None: continue
            ax.errorbar(r["rho_median"], y + o, xerr=[[r["rho_median"] - r["rho_ci"][0]], [r["rho_ci"][1] - r["rho_median"]]],
                        fmt=MK[m], ms=MS, color=COL[m], mec="white", mew=0.5, elinewidth=1.4, capsize=2.5, capthick=1.2, zorder=3)
    ax.axvline(1.0, color="0.35", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(fams))); ax.set_yticklabels([FL.get(f, f) for f in fams][::-1])
    ax.set_xscale("log"); ax.set_xlim(0.9, 12.0)
    _t = [1, 2, 3, 5, 10] if SET == "main" else [1, 1.5, 2, 3, 5, 10]
    ax.set_xticks(_t); ax.set_xticklabels([f"{v:g}" for v in _t]); ax.minorticks_off()
    ax.set_title({"Navier\u2013Stokes benchmark corpus": "Navier\u2013Stokes benchmark", "Poseidon pretraining corpus": "Poseidon corpus"}.get(title, title)
                 if SET == "main" else title, fontsize=10.5 if SET == "main" else 9.4, pad=30 if SET == "main" else 4)
    ax.set_xlabel("replica error ratio $\\rho$")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for m in models:
        ax.plot([], [], MK[m], color=COL[m], ms=MS, label=LAB.get(m, m))
    if SET == "main":                                                  # legend above the panel, two columns
        ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=8.4,
                  handletextpad=0.1, columnspacing=0.5, borderaxespad=0.0, labelspacing=0.25)
    elif title.startswith("Poseidon") or title.startswith("Navier"):   # legend above the panel
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1 if SET == "main" else 3, frameon=False, fontsize=9.0 if SET == "main" else 7.8,
                  handletextpad=0.1, columnspacing=0.6, borderaxespad=0.0)
    else:
        ax.legend(loc="lower right", frameon=False, fontsize=6.4, handletextpad=0.2, borderaxespad=0.1)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, ("fig4b_replica_r1" if SET == "main" else "figS_replica_19fam") + f".{ext}"), bbox_inches="tight")
print("wrote fig4b_replica_r1")

