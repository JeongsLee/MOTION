"""Fig 4e (exploratory) -- the equivariance defect against the accuracy it sits on.

x: the operator's own error on the family (physical, %).  y: the defect eps_equiv (%).
The diagonal y = x is the reference the paper's interpretive rule names: two predictions
that are each accurate to e cannot disagree by much more than 2e, so a defect at the level
of the error means no preferred representative is left to detect.  Distance ABOVE the
diagonal is the whole claim, and family difficulty is absorbed by the diagonal -- which is
what the replica accuracy ratio could not do, since dividing by the error penalises the
operator that is most accurate.
Legal transforms only; sources as in render_fig4e_span.py.
"""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
for _f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _p = f"/mnt/c/Windows/Fonts/{_f}"
    if os.path.exists(_p): fm.fontManager.addfont(_p)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 8.5, "axes.linewidth": 0.8, "savefig.dpi": 400})

# model -> (transform, family, own error %, defect eps %)
D = {
 "MOTION": [("mirror","SWE",0.048,0.03),("rotation","SWE",0.048,0.03),("translation","SWE",0.048,0.08),   # REVISION-R1 10-22: MOTION-NS (zsc) rows,
            ("mirror","Com",2.068,0.85),("rotation","Com",2.068,0.89),("translation","Com",2.068,0.11),   # eval_equiv run4 / run3
            ("mirror","Incom",0.830,0.54),("rotation","Incom",0.830,0.56),("translation","Incom",0.830,0.55),
            ("mirror","PDEarena",8.034,7.74),("mirror","PDEarena-u",5.953,6.40),
            ("mirror","CFDbench",0.062,0.04),
            # REVISION-R1 10-22 (user): transfer rows (NS-PwC/ACE/Poisson/Wave, moved to SI with the IVP benchmark) replaced
            # by MOTION-S on the six Poseidon pretraining families (eval20k m1f, EQUIV_SCORE_SLOTS rho,u,v,p, test first 32)
            ("mirror","CE-RP",10.322,5.935),("rotation","CE-RP",10.322,6.25),("translation","CE-RP",10.322,2.636),
            ("mirror","CE-CRP",23.463,10.821),("rotation","CE-CRP",23.463,11.473),("translation","CE-CRP",23.463,6.088),
            ("mirror","CE-KH",8.761,14.742),("rotation","CE-KH",8.761,18.285),("translation","CE-KH",8.761,2.287),
            ("mirror","CE-Gauss",6.022,3.985),("rotation","CE-Gauss",6.022,4.152),("translation","CE-Gauss",6.022,1.388),
            ("mirror","NS-Sines",11.484,9.925),("rotation","NS-Sines",11.484,10.344),("translation","NS-Sines",11.484,2.932),
            ("mirror","NS-Gauss",11.422,7.718),("rotation","NS-Gauss",11.422,8.101),("translation","NS-Gauss",11.422,2.419)],
 # Poseidon-T on the same test trajectories (poseidon_equiv.py, Poseidon-T_equiv_test.json)
 "Poseidon-T": [("mirror","CE-RP",6.683,4.167),("rotation","CE-RP",6.683,4.351),("translation","CE-RP",6.683,5.686),
                ("mirror","CE-CRP",26.672,12.902),("rotation","CE-CRP",26.672,13.416),("translation","CE-CRP",26.672,9.232),
                ("mirror","CE-KH",5.769,13.158),("rotation","CE-KH",5.769,16.091),("translation","CE-KH",5.769,6.678),
                ("mirror","CE-Gauss",3.319,2.491),("rotation","CE-Gauss",3.319,2.562),("translation","CE-Gauss",3.319,1.68),
                ("mirror","NS-Sines",18.341,27.621),("rotation","NS-Sines",18.341,27.759),("translation","NS-Sines",18.341,20.778),
                ("mirror","NS-Gauss",5.406,5.887),("rotation","NS-Gauss",5.406,5.948),("translation","NS-Gauss",5.406,3.782)],
 # REVISION R1 (A01): BCAT/PROSE-FD SWE mirror/rotation corrected (channel-padding bug; were 200.0 / 141.4)
 "BCAT": [("mirror","SWE",0.208,0.21),("rotation","SWE",0.208,0.22),("translation","SWE",0.208,1.21),
          ("mirror","Com",3.893,1.23),("rotation","Com",3.893,1.24),("translation","Com",3.893,0.49),
          ("mirror","Incom",5.175,5.78),("rotation","Incom",5.175,5.79),("translation","Incom",5.175,5.46),
          ("mirror","PDEarena",7.510,8.01),("mirror","PDEarena-u",5.692,5.78),("mirror","CFDbench",0.453,1.218)],   # CFD: 2 flow channels (replica_all.json, 10-21)
 "PROSE-FD": [("mirror","SWE",0.203,0.10),("rotation","SWE",0.203,0.12),
              # from the surviving dumps (job pertraj-harvest); the isotropic families admit
              # every mirror, so the worst of {fr,fc,r180} is the legal worst case
              ("mirror","Com",3.770,0.731),("mirror","Incom",7.483,5.888),
              ("rotation","Com",3.770,0.740),("rotation","Incom",7.483,5.893),
              # job symk8-prose-cfd (scalar channels, no sign flip) and symk8-prose-arena
              ("mirror","CFDbench",0.237,2.099),   # 2 flow channels, as BCAT and MOTION (replica_all.json, 10-21)
              ("mirror","PDEarena",11.420,8.078),("mirror","PDEarena-u",9.255,6.746),
              # job symk8-prose-shift: worst of the one-cell and the patch-aligned roll
              ("translation","SWE",0.203,3.968),("translation","Com",3.770,0.605),
              ("translation","Incom",7.483,10.083)],
}
COL = {"MOTION": "#1a4f8a", "Poseidon-T": "#d97706", "BCAT": "#c0392b", "PROSE-FD": "#8f5b2a"}
MK = {"mirror": "o", "rotation": "s", "translation": "^"}

fig, ax = plt.subplots(figsize=(3.7, 3.05))
fig.subplots_adjust(left=0.155, right=0.985, top=0.965, bottom=0.155)
xx = np.array([0.03, 400.0])
ax.plot(xx, xx, "-", color="0.35", lw=0.9, zorder=1)
ax.plot(xx, 3 * xx, ls=(0, (4, 3)), color="0.65", lw=0.8, zorder=1)
ax.text(0.20, 0.145, "$\\epsilon=$ own error", rotation=39, fontsize=6.6, color="0.35")
ax.text(0.056, 0.30, "$3\\times$", rotation=39, fontsize=6.4, color="0.65")
for m, rows in D.items():
    for t, fam, e, eps in rows:
        ax.plot(e, eps, MK[t], ms=4.4, mfc=COL[m], mec="white", mew=0.55, zorder=4)
for lab, xy, off, col in (("BCAT / PROSE-FD\nrow mirror, SWE", (0.205, 200.0), (14, -4), "#c0392b"),
                          ("transpose, SWE", (0.205, 141.4), (14, -14), "#c0392b"),
                          ("Poseidon-B\nrotation, NS-PwC", (4.35, 11.34), (7, 9), "#d97706")):
    ax.annotate(lab, xy, textcoords="offset points", xytext=off, fontsize=6.4, color=col)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.035, 30); ax.set_ylim(0.035, 400)
ax.set_xlabel("operator's own error on the family (%)")
ax.set_ylabel("equivariance defect $\\epsilon_{\\rm equiv}$ (%)")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
h1 = [Line2D([], [], marker="o", ls="none", ms=4.4, mfc=COL[m], mec="white", mew=0.55, label=m)
      for m in D]
h2 = [Line2D([], [], marker=MK[t], ls="none", ms=4.4, mfc="0.45", mec="white", mew=0.55, label=l)
      for t, l in (("mirror", "mirrors"), ("rotation", "$90^\\circ$ rot."), ("translation", "transl."))]
ax.legend(handles=h1 + h2, loc="lower right", frameon=False, fontsize=6.8,
          handletextpad=0.3, labelspacing=0.26, ncol=2, columnspacing=0.9)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig4e_scatter.{ext}"), bbox_inches="tight")
print("wrote fig4e_scatter.pdf")
for m, rows in D.items():
    r = [eps / e for _, _, e, eps in rows]
    print(f"  {m:11s} n={len(rows):2d}  eps/err  {min(r):.2f} - {max(r):7.2f}")
