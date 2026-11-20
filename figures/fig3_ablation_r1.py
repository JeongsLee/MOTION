"""Fig. 3 ablation (revision R1, 2026-10-23, user): 2x2 layout, D / AUC (with permutation p) / E only; no table, no OOD;
vocabulary moved to SI (separate output). Derived from an earlier figure script.
Bar design following the Nature figure guide: 183 mm wide, sans-serif (Arial-metric Liberation Sans)
5–7 pt, Okabe–Ito colour-blind-safe palette, bars from zero, no frames/boxes/grid, thin axes, direct x labels.
Notation: mechanism = prescribed operator vocabulary, integration = closed-form synthesis (the M and I of MOTION)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from fx_common import load, ko_matrix, criteria, traj, R, D as DDIR
import align_metrics as AM            # revision: AUC (operator selectivity) and the D reference lines

for f in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"):
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "custom", "mathtext.it": "Liberation Sans:italic", "mathtext.cal": "DejaVu Sans:italic",
                     "font.size": 7.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 0, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.titlesize": 7.5, "axes.titlepad": 3})
INK, MUTED = "#1a1a1a", "#7a7a7a"
# Okabe–Ito
VER, ORA, SKY, BLU, GRN, PUR, GRY, LGR = "#D55E00", "#E69F00", "#56B4E9", "#0072B2", "#009E73", "#CC79A7", "#7f7f7f", "#c7c7c7"
_cache = {}


# REVISION-R1 10-30: BRG's stats.npz did not reach the object volume (STATS_OK logged); D and E only need the family
# means of the per-trajectory errors, which stats.log prints -> stand-in with the npz interface used by traj() /
# ko_matrix() (one pseudo-trajectory per family; validated on BR_s1 / BR_s2 / HY8mM: D within 3e-4, E exact)
class LogStats:
    def __init__(self, path):
        import re
        self.d = {}
        for m in re.finditer(r"^\[(\S+)\]\s+(\S+)\s+n=\s*\d+ phys=\s*([\d.]+)", open(path).read(), re.M):
            self.d[f"{m.group(1)}|{m.group(2)}|phys"] = np.full((1, 1, 1), float(m.group(3)) / 100)
            self.d[f"{m.group(1)}|{m.group(2)}|w"] = np.ones((1, 1, 1))
        self.files = list(self.d)

    def __getitem__(self, k): return self.d[k]


def metrics(c):
    if c in _cache: return _cache[c]
    Z = load(c, "stats.npz")
    if Z is None and os.path.exists(f"{DDIR}/{c}/stats.log"): Z = LogStats(f"{DDIR}/{c}/stats.log")
    if Z is None: _cache[c] = None; return None
    H = sorted({k.split("|")[0][3:] for k in Z.files if k.startswith("ko:") and "+" not in k
                and not k.split("|")[0][3:].startswith("generic")})
    FA = sorted({k.split("|")[1] for k in Z.files if k.startswith("base|")})
    q = criteria(ko_matrix(Z, H, FA), H, FA)
    five = c in ("VT5", "VB5", "VR5a", "VR5b")                     # 5-head vocabulary runs: the always-open heads are off,
    am = None if five else AM.metrics(c, nperm=1000)                  # relative-reliance basis (contrib.json)
    m = {"D": None if five else float(q["C2"].mean()),                # so D/AUC are undefined
         "A": (am["AUC"] if am else None), "Ap": (am["AUC_p"] if am else None), "Dphys": (am["D_phys"] if am else None),
         "S": float(q["C6"].mean()) if len(q["C6"]) and not five else None,
         "E": float(np.mean([100 * traj(Z, "base", f).mean() for f in FA])), "OOD": None}
    zh, bh = load(c, "horizon/h2_k18.npz"), load("BR", "horizon/h2_k18.npz")
    if zh is not None:
        ff = sorted({k.split("|")[1] for k in bh.files if k.startswith("base|")} - {"diff_react", "Wave-Layer"})
        ff = [f for f in ff if f"base|{f}|phys" in zh.files]
        m["OOD"] = float(np.mean([100 * traj(zh, "base", f).mean() for f in ff]))
    _cache[c] = m; return m


MT = {"D": "Differentiation $D$ ↑  (- - physics ref.)", "A": "Operator selectivity AUC ↑  (- - chance)", "S": "Same-form separation $S$ ↑",
      "E": "Error $E$ (%) ↓", "OOD": "Error, OOD (%) ↓"}
ROWS = [("c", "Operator design", ["D", "A", "E", "OOD"],
         [("BR", "MOTION", VER), ("V3", "w/o integration", ORA), ("V1a", "w/o mechanism", BLU),
          ("V1aV3", "w/o mechanism\n& integration", SKY)]),
        ("d", "Equation metadata", ["D", "A", "S", "E"],
         [("BR", "MOTION", VER), ("V2", "w/o metadata", "#f2a96b"), ("V1aN", "w/o mechanism\n& metadata", "#5b9fd0"),
          ("V1aV3N", "w/o mechanism, integration\n& metadata", "#a8d3f0")]),
        ("e", "Lagrangian transport", ["D", "A", "E", "OOD"],
         [("BR", "MOTION", VER), ("BRW", "+ $\mathcal{L}_t$", GRN), ("V1a", "w/o mechanism", BLU),
          ("V1aW", "w/o mechanism\n+ $\mathcal{L}_t$", "#66c2a5")]),
        ("f", "Head partition (all + $\mathcal{L}_t$)", ["D", "A", "E", "OOD"],
         [("BRW", "19 heads", GRN), ("VBL", "1 head\nlinear", "#2a9d8f"), ("VBN", "1 head\nnonlinear", "#8fd3c7"),
          ("HY8m", "19 + 8\ngeneric", PUR), ("HY8mM", "19 + 1\nmerged", "#e3a6c8"), ("VBLG", "1 head\n+ 8 generic", "#f1cfe2")]),
        ("g", "Vocabulary (all + $\mathcal{L}_t$)", ["E", "OOD"],                      # 5-head runs: D/AUC undefined (always-open heads off)
         [("BRW", "19 mechanisms", GRN), ("VT5", "top-5", GRY), ("VB5", "bottom-5", "#a6a6a6"),
          ("VR5a", "random-5 a", LGR), ("VR5b", "random-5 b", "#e3e3e3")])]

FS = 1.3
VOCAB = os.environ.get("VOCAB", "0") == "1"           # 1: SI vocabulary panel only
MTS = {"D": "$D$ ↑", "A": "AUC ↑", "E": "$E$ (%) ↓"}


def pfmt(p):
    return "$p\\leq$0.001" if p <= 0.0011 else f"$p$ = {p:.3f}" if p < 0.1 else f"$p$ = {p:.2f}"


def panel(cell, letter, rname, cols, models, ncol=None, title_dx=-0.42, xlabels=False, title_y=1.42, wspace=0.42):
    sub = cell.subgridspec(1, len(cols), wspace=wspace)
    for j, k in enumerate(cols):
        ax = fig.add_subplot(sub[0, j]); n = len(models)
        vals = [(metrics(c) or {}).get(k) for c, _, _ in models]
        vmax = max([v for v in vals if v is not None] + [1e-6])
        ref = {"D": next(((metrics(c) or {}).get("Dphys") for c, _, _ in models if (metrics(c) or {}).get("Dphys")), None),
               "A": AM.AUC_CHANCE}.get(k)
        if ref: vmax = max(vmax, ref)
        for i, ((c, lab, col), v) in enumerate(zip(models, vals)):
            if v is None: continue
            ax.bar(i, v, color=col, width=0.74, lw=0)
            ax.text(i, v + vmax * 0.02, f"{v:.2f}" if v < 5 else f"{v:.1f}", ha="center", va="bottom", fontsize=(4.6 if n < 6 else 4.0) * FS,
                    color=INK, weight="bold" if i == 0 else None)
            if k == "A":                                   # permutation p of the AUC, inside the bar (user 10-23)
                p = (metrics(c) or {}).get("Ap")
                if p is not None:
                    ax.text(i, vmax * 0.04, pfmt(p), rotation=90, ha="center", va="bottom", fontsize=4.4 * FS,
                            color="white" if col not in ("#a8d3f0", "#f1cfe2", "#e3a6c8", "#8fd3c7", "#f2a96b", "#66c2a5") else INK)
        if ref: ax.axhline(ref, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.set_ylim(0, vmax * 1.2); ax.set_xlim(-0.6, n - 0.4); ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=5.2 * FS, pad=1); ax.locator_params(axis="y", nbins=3)
        ax.set_title(MTS[k], loc="left", fontsize=6.0 * FS)
        if xlabels:
            ax.set_xticks(range(n)); ax.set_xticklabels([l for _, l, _ in models], rotation=40, ha="right", fontsize=5.2 * FS)
        if j == 0:
            if letter: ax.text(title_dx, title_y, letter, fontsize=8 * FS, weight="bold", va="bottom", transform=ax.transAxes)
            ax.text(title_dx + (0.16 if letter else 0), title_y, rname, fontsize=6.3 * FS, weight="bold", va="bottom",
                    transform=ax.transAxes, color=INK)
            from matplotlib.patches import Patch
            if not xlabels: ax.legend(handles=[Patch(color=col, label=lab.replace("\n", " ")) for _, lab, col in models],
                      loc="lower left", bbox_to_anchor=(0.0, 1.13), ncol=ncol or len(models), frameon=False,
                      fontsize=5.2 * FS, handlelength=0.9, handleheight=0.8, handletextpad=0.35, columnspacing=0.9,
                      borderaxespad=0)


OPD = [("BR", "MOTION-S", VER), ("V3", "w/o integration", ORA), ("V1a", "w/o mechanism", BLU),
       ("V1aV3", "w/o mechanism & integration", SKY)]
META = [("BR", "MOTION-S", VER), ("V2", "w/o metadata", "#f2a96b"), ("V1aN", "w/o mechanism & metadata", "#5b9fd0"),
        ("V1aV3N", "w/o mechanism, integration & metadata", "#a8d3f0")]
TRN = [("BR", "MOTION-S", VER), ("BRW", "+ $\mathcal{L}_t$", GRN), ("V1a", "w/o mechanism", BLU), ("V1aW", "w/o mechanism + $\mathcal{L}_t$", "#66c2a5")]
# REVISION-R1 10-26 (user): Fig. 3d = relaxations of the strict MOTION-S (warp; generic pathway G = 8 generic branches
# merged behind one gate, as in MOTION-NS/IVP; 8 separate generic heads); the head-partition variants go to the SI
# (HEADPART=1) and the response table
# REVISION-R1 10-29 (user): Fig. 3d = three bars (MOTION-S, + L_t, + L_t + G); the 8 separately gated generic heads
# (HY8m) only in the SI head-partition panel (HPS)
# REVISION-R1 10-30 (user): Fig. 3d = 2x2 (MOTION-S, + L_t, + G, + L_t + G); + G alone = rev_BRG_s0
HP = [("BR", "MOTION-S", VER), ("BRW", "+ $\mathcal{L}_t$", GRN), ("BRG", "+ $G$", PUR), ("HY8mM", "+ $\mathcal{L}_t$ + $G$", "#e3a6c8")]
HPS = [("BRW", "19 heads", GRN), ("VBL", "1 head linear", "#2a9d8f"), ("VBN", "1 head nonlinear", "#8fd3c7"),
       ("HY8m", "19 + 8 generic", PUR), ("HY8mM", "19 + 1 merged", "#e3a6c8"), ("VBLG", "1 head + 8 generic", "#f1cfe2")]
VOC = [("BRW", "19 mechanisms", GRN), ("VT5", "top-5", GRY), ("VR5a", "random-5 a", LGR),
       ("VR5b", "random-5 b", "#e3e3e3"), ("VB5", "bottom-5", "#a6a6a6")]
HEADPART = os.environ.get("HEADPART", "0") == "1"     # 1: SI head-partition panel only
if HEADPART:
    fig = plt.figure(figsize=(120 / 25.4, 62 / 25.4)); fig.patch.set_facecolor("white")
    g = fig.add_gridspec(1, 1, left=0.10, right=0.99, top=0.80, bottom=0.30)
    panel(g[0], None, "Head partition (all + $\mathcal{L}_t$)", ["D", "A", "E"], HPS, xlabels=True, title_dx=-0.20, title_y=1.10)
    out = f"{R}/figures/out/figS_headpartition"
elif VOCAB:
    fig = plt.figure(figsize=(60 / 25.4, 62 / 25.4)); fig.patch.set_facecolor("white")
    g = fig.add_gridspec(1, 1, left=0.24, right=0.97, top=0.80, bottom=0.30)       # REVISION-R1 11-06: title above the axis title
    panel(g[0], None, "Mechanism vocabulary (all + $\mathcal{L}_t$)", ["E"], VOC, xlabels=True, title_dx=-0.25, title_y=1.13)
    out = f"{R}/figures/out/figS_vocabulary"
else:
    # REVISION-R1 10-30 (user): smaller gap between rows a,b and c,d (hspace 0.95 -> 0.70), figure height kept (128 mm)
    fig = plt.figure(figsize=(183 / 25.4, 128 / 25.4)); fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(2, 1, hspace=0.70, left=0.06, right=0.99, top=0.83, bottom=0.03)
    r1 = outer[0].subgridspec(1, 2, wspace=0.16); r2 = outer[1].subgridspec(1, 2, wspace=0.16)
    panel(r1[0, 0], "a", "Operator design", ["D", "A", "E"], OPD, ncol=2)
    panel(r1[0, 1], "b", "Equation metadata", ["D", "A", "E"], META, ncol=2)
    panel(r2[0, 0], "c", "Lagrangian transport", ["D", "A", "E"], TRN, ncol=2)
    panel(r2[0, 1], "d", "Relaxed structure", ["D", "A", "E"], HP, ncol=2)
    out = f"{R}/figures/out/fig3_ablation_r1"
fig.savefig(out + ".png", dpi=260); fig.savefig(out + ".pdf")
import json
json.dump({c: {k: (metrics(c) or {}).get(k) for k in ("D", "A", "Ap", "E", "Dphys")} for c, _, _ in OPD + META + TRN + HP + HPS + VOC},
          open(out + ".json", "w"), indent=1)
print("saved", out)
