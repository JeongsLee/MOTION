"""Response Fig. R1.3 — within-family regime tracking (R1-3). Reliance of each test trajectory on the advection head
(relative KO damage) against its Mach number (measured from the ground truth; never given to the model), in the three
compressible-Euler families with a usable Mach spread. Rows: paper MOTION-S (m1f), MOTION (B-R, 20k), w/o mechanism
(V1a: the generic head that occupies the advection slot, same gate and metadata mask). Spearman rho, 95% bootstrap CI,
permutation p. Same per-trajectory seeds and order as reliance_corr.py (A08)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import spearmanr
from fx_common import R
for f in ("Regular", "Bold", "Italic"):
    p = f"/usr/share/fonts/truetype/liberation/LiberationSans-{f}.ttf"
    if os.path.exists(p): fm.fontManager.addfont(p)
plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "custom", "mathtext.it": "Liberation Sans:italic",
                     "font.size": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.titlesize": 6})
A = f"{R}/results"
MODELS = [("m2s2", "MOTION (paper)", f"{A}/A08_reliance_vs_regime/data/m2s2", "#8B2500"),
          ("m1f", "MOTION-S (paper)", f"{A}/A08_reliance_vs_regime/data/m1f", "#D55E00"),
          ("BR", "MOTION (ablation base)", f"{A}/A14_factorial/data/BR", "#E69F00"),
          ("V1a", "w/o mechanism (generic, same slot)", f"{A}/A14_factorial/data/V1a", "#0072B2")]
FAMS = ["CE-RP", "CE-CRP", "CE-Gauss"]
HEAD, DIAG = sys.argv[1] if len(sys.argv) > 1 else "advection", sys.argv[2] if len(sys.argv) > 2 else "mach"
D = np.load(f"{A}/A08_reliance_vs_regime/data/diag_test128.npz")
rng = np.random.default_rng(0)


def traj(S, cfg, fam):
    e, w = S[f"{cfg}|{fam}|phys"], S[f"{cfg}|{fam}|w"]
    return ((e * w).sum(-1) / np.maximum(w.sum(-1), 1e-6)).mean(1)


fig, axs = plt.subplots(len(MODELS), len(FAMS), figsize=(120 / 25.4, 128 / 25.4))
out = {}
for i, (key, lab, path, col) in enumerate(MODELS):
    S = np.load(f"{path}/stats.npz")
    for j, fam in enumerate(FAMS):
        ax = axs[i, j]
        eb = traj(S, "base", fam); r = 100 * (traj(S, f"ko:{HEAD}", fam) - eb) / np.maximum(eb, 1e-12)
        x = D[f"{fam}|{DIAG}"][: len(r)]; ok = np.isfinite(x) & np.isfinite(r); x, r = x[ok], r[ok]
        rho = spearmanr(x, r).correlation
        bs = [spearmanr(x[b], r[b]).correlation for b in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        p = float((np.abs([spearmanr(rng.permutation(x), r).correlation for _ in range(2000)]) >= abs(rho)).mean())
        out[f"{key}|{fam}"] = dict(rho=float(rho), ci=[float(lo), float(hi)], p=p, n=int(len(x)), mean_rel=float(r.mean()))
        ax.scatter(x, r, s=3, color=col, alpha=0.55, lw=0)
        # binned medians (quintiles of the regime)
        q = np.quantile(x, np.linspace(0, 1, 6)); c = 0.5 * (q[1:] + q[:-1])
        med = [np.median(r[(x >= a) & (x <= b)]) for a, b in zip(q[:-1], q[1:])]
        ax.plot(c, med, color="#1a1a1a", lw=0.8, marker="o", ms=1.8)
        ax.axhline(0, color="#bdbdbd", lw=0.4, zorder=0)
        ax.text(0.03, 0.97, f"ρ = {rho:+.2f} [{lo:+.2f}, {hi:+.2f}]\np {'< 0.001' if p < 1e-3 else f'= {p:.3f}'}, n = {len(x)}",
                transform=ax.transAxes, va="top", fontsize=4.8)
        ax.tick_params(labelsize=5, pad=1); ax.locator_params(nbins=4)
        if i == 0: ax.set_title(fam, fontsize=6.3, weight="bold")
        if i == len(MODELS) - 1: ax.set_xlabel("Mach number (ground truth)" if DIAG == "mach" else DIAG, fontsize=5.5)
        if j == 0: ax.set_ylabel(f"{lab}\n{HEAD} KO damage (%)", fontsize=5.3)
fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.5)
o = f"{R}/figures/out/fig_regime_{HEAD}_{DIAG}"
fig.savefig(o + ".pdf"); fig.savefig(o + ".png", dpi=260)
json.dump(out, open(f"{A}/A08_reliance_vs_regime/data/regime_{HEAD}_{DIAG}.json", "w"), indent=1)
for k, v in out.items(): print(f"{k:16s} rho {v['rho']:+.2f} CI [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] p {v['p']:.3f} n {v['n']} mean KO {v['mean_rel']:+.1f}%")
