"""REVISION-R1 11-02 (R1 minor 4): unnormalised counterparts of Fig. 1c for the Supplementary Information, from the
same data, heads and families as the full-vocabulary Fig. 1c (fig1c_m1f.py with FIG1C_FULL=1):
  a  knockout: relative error increase (%) = 100 (KO error / full error - 1), log scale; blank where the paired 95%
     bootstrap interval of the error increase includes zero (the Fig. 1c test). Fig. 1c instead divides the error
     increase by the headroom (persistence error - full error) and clips to [0, 1].
  b  contribution share s_{k,f} (%) of head k in the summed gated tendency magnitude of family f.
  c  log2 enrichment of the share over the head's median share across families (floor 1%), clipped at zero,
     BEFORE the per-family max normalisation used in Fig. 1c.
Grey: head closed by the metadata gate. Writes 04_response/figs/fig1c_raw.{pdf,png} (copied to the manuscript figs).
  python fig1c_raw.py
"""
import os
import numpy as np

os.environ["FIG1C_FULL"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "fig1c_m1f.py")).read()
exec(_src[:_src.index("# REVISION-R1 10-29 (user): larger type")])          # data, S/SIG/CLOSED, CJ, share, Rr

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm

K = np.full_like(S, np.nan)                                                   # relative error increase, %
for j, f in enumerate(FAMS):
    b = err("base", f).mean()
    for i, h in enumerate(SEL):
        if f"ko:{h}|{f}|phys" in Z.files:
            K[i, j] = 100 * (err(f"ko:{h}", f).mean() / b - 1)
SH = np.array([100 * share[heads_all.index(h)] if h in heads_all else np.zeros(len(FAMS)) for h in SEL])
EN = np.array([Rr[heads_all.index(h)] if h in heads_all else np.zeros(len(FAMS)) for h in SEL])

fig, axes = plt.subplots(3, 1, figsize=(5.4, 8.4), sharex=True, gridspec_kw={"hspace": 0.16})
panels = [(np.where(SIG & ~CLOSED, K, np.nan), "Reds", LogNorm(1, 1000), "knockout: relative error increase (%)"),
          (np.where(CLOSED, np.nan, SH), "Blues", None, "contribution share (%)"),
          (np.where(CLOSED, np.nan, EN), "Blues", None, "log$_2$ enrichment over the median share (not normalised)")]
for ax, (M, cm, norm, title) in zip(axes, panels):
    cmap = plt.get_cmap(cm).copy(); cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, norm=norm,
                   vmin=None if norm else 0, vmax=None if norm else np.nanmax(M))
    ax.imshow(np.ma.masked_where(~CLOSED, np.ones_like(M)), aspect="auto", cmap=ListedColormap(["#e8e8e8"]), vmin=0, vmax=1)
    ax.set_yticks(range(len(SEL))); ax.set_yticklabels([MLAB[h] for h in SEL], fontsize=6.4)
    ax.axvline(len(FAMS) - 3.5, color="k", lw=0.7); ax.set_title(title, fontsize=7.2, pad=1.5)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.008)
axes[-1].set_xticks(range(len(FAMS))); axes[-1].set_xticklabels([FS.get(f, f) for f in FAMS], rotation=90, fontsize=6.6)
for k, ax in enumerate(axes):
    ax.text(-0.33, 1.02, "abc"[k], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig1c_raw.{ext}"), dpi=300, bbox_inches="tight")
# numbers quoted in the SI caption / response
mx = np.nanmax(np.where(SIG & ~CLOSED, K, np.nan), 0)
for j, f in enumerate(FAMS):
    o = ~CLOSED[:, j]
    print(f"{FS.get(f, f):12s} max KO {mx[j]:8.1f}%  top share {SH[o, j].max():5.1f}% ({MLAB[SEL[int(np.argmax(np.where(o, SH[:, j], -1)))]]})"
          f"  max enrich {EN[o, j].max():4.2f}")
print("wrote fig1c_raw")
