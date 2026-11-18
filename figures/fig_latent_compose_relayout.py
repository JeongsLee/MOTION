"""Re-layout of the latent-composition figure (Supplementary Fig. 'Latent composition with a numerical reference').

The npz written by fv2_opt/latent_compose.py (input of fig_latent_compose.py) is not available locally, so this
script rebuilds the figure from the existing vector PDF without changing any data:
  - panel a: the ten vorticity maps are the images embedded in the PDF (same pixels);
  - panels b, c: bar heights and per-trajectory dots are read from the PDF drawings and converted to data values
    with the axis box and the y-limits of the original script (-0.3, 1.0).
Layout: panel a on top (two rows of five maps), panels b and c side by side below, larger fonts.

  python fig_latent_compose_relayout.py [SRC.pdf] [OUT.pdf]
"""
import os
import sys

import fitz
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{R}/figures/out/fig_latent_compose_m1f_a3.pdf"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{R}/figures/out/fig_latent_compose_m1f_a3_v2.pdf"
YLIM = (-0.3, 1.0)

doc = fitz.open(SRC)
pg = doc[0]

# ---------------- panel a: embedded images, in reading order
ims = []
for im in pg.get_images(full=True):
    rect = pg.get_image_rects(im[0])[0]
    pm = fitz.Pixmap(doc, im[0])
    if pm.n > 3:
        pm = fitz.Pixmap(fitz.csRGB, pm)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3]
    # orientation check against the rendered page
    ren = pg.get_pixmap(clip=rect, dpi=300)
    r = np.frombuffer(ren.samples, np.uint8).reshape(ren.height, ren.width, ren.n)[:, :, :3].astype(float)
    def small(x):
        iy = np.linspace(0, x.shape[0] - 1, 48).astype(int); ix = np.linspace(0, x.shape[1] - 1, 48).astype(int)
        return x[np.ix_(iy, ix)].astype(float).mean(-1).ravel()
    c0 = np.corrcoef(small(a), small(r))[0, 1]; c1 = np.corrcoef(small(a[::-1]), small(r))[0, 1]
    if c1 > c0:
        a = a[::-1]
    ims.append((round(rect.y0), rect.x0, a))
ims.sort(key=lambda t: (t[0], t[1]))
assert len(ims) == 10
rows = [ims[:5], ims[5:]]

# ---------------- panels b, c: drawings
dr = [x for x in pg.get_drawings() if x["rect"].x0 > 330]
white = sorted([x["rect"] for x in dr if x.get("fill") == (1.0, 1.0, 1.0) and x["rect"].width > 100], key=lambda r: r.y0)
assert len(white) == 2
axb, axc = white
def val(y, ax):
    return YLIM[0] + (ax.y1 - y) / (ax.y1 - ax.y0) * (YLIM[1] - YLIM[0])
def close(c, ref):
    return c is not None and max(abs(a - b) for a, b in zip(c, ref)) < 0.03
COL = {"b1": (0.0, 0.447, 0.698), "b10": (0.835, 0.369, 0.0), "rest": (0.6, 0.6, 0.6), "other": (0.0, 0.62, 0.451)}
def bars(ax, colors):
    out = []
    zero = ax.y1 - (0 - YLIM[0]) / (YLIM[1] - YLIM[0]) * (ax.y1 - ax.y0)
    for x in dr:
        r = x["rect"]
        if x.get("type") == "f" and r.x0 >= ax.x0 - 1 and r.x1 <= ax.x1 + 1 and r.y0 >= ax.y0 - 1 and r.y1 <= ax.y1 + 1:
            for name in colors:
                if close(x.get("fill"), COL[name]) and r.width < 25 and (abs(r.y1 - zero) < 0.6 or abs(r.y0 - zero) < 0.6):
                    top = r.y0 if abs(r.y0 - zero) >= abs(r.y1 - zero) else r.y1   # edge farther from zero
                    out.append(((r.x0 + r.x1) / 2, name, val(top, ax), r.width))
    return sorted(out)
def dots(ax):
    out = []
    for x in dr:
        r = x["rect"]
        if x.get("type") == "f" and close(x.get("fill"), (0, 0, 0)) and 1.0 < r.width < 2.0 and 1.0 < r.height < 2.0:
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if ax.x0 - 2 <= cx <= ax.x1 + 2 and ax.y0 - 2 <= cy <= ax.y1 + 2:
                out.append((cx, val(cy, ax)))
    return out
bb, bc = bars(axb, ("b1", "b10")), bars(axc, ("rest", "other"))
db, dc = dots(axb), dots(axc)
assert len(bb) == 16 and len(bc) == 6 and len(db) == 64 and len(dc) == 24, (len(bb), len(bc), len(db), len(dc))
def attach(bs, ds):
    res = []
    for cx, name, h, w in bs:
        res.append((name, h, sorted(v for x, v in ds if abs(x - cx) < w / 2 + 0.3)))
    assert all(len(r[2]) == 4 for r in res), [len(r[2]) for r in res]
    return res
B, C = attach(bb, db), attach(bc, dc)
# exact bar heights from the summary json written by fig_latent_compose.py (the PDF read-out is used as a check)
import json
J = json.load(open(f"{R}/figures/out/fig_latent_compose_m1f_a3.json"))
jb = [J["model_vs_solver_r"]] + [J["conds_r_solver"][k] for k in ("i_anchor_only", "i_rest", "i_other", "i_random", "ii_other", "ii_flip", "ii_random")]
for q in range(8):
    for s_, st in enumerate(("1", "10")):
        nm, h, ds = B[2 * q + s_]; assert abs(h - jb[q][st]) < 0.006, (q, st, h, jb[q][st])
        assert abs(np.mean(ds) - jb[q][st]) < 0.006, (q, st, np.mean(ds), jb[q][st])
        B[2 * q + s_] = (nm, jb[q][st], ds)
for q, k in enumerate(("cos", "cos_disk", "cos_misplaced")):
    for s_, c_ in enumerate(("rest", "other")):
        nm, h, ds = C[2 * q + s_]; assert abs(h - J["additivity"][c_][k]) < 0.006 and abs(np.mean(ds) - J["additivity"][c_][k]) < 0.006
        C[2 * q + s_] = (nm, J["additivity"][c_][k], ds)

# ---------------- new figure
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "pdf.fonttype": 42, "axes.linewidth": 0.6})
MAPS = ["solver\nS(a+b) − S(a)", "model, real input\nP(a+b) − P(a)", "anchor a+b,\nlatent of a only",
        "anchor a+b,\ncomposed latent", "anchor a,\ncomposed latent"]
fig = plt.figure(figsize=(7.1, 5.55))
gm = fig.add_gridspec(2, 5, left=0.055, right=0.995, top=0.895, bottom=0.425, wspace=0.04, hspace=0.05)
for r_, (row, lab) in enumerate(zip(rows, ("step 1", "step 10"))):
    for j, (_, _, a) in enumerate(row):
        ax = fig.add_subplot(gm[r_, j]); ax.imshow(a, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.4)
        if r_ == 0: ax.set_title(MAPS[j], fontsize=7.5)
        if j == 0: ax.set_ylabel(lab, fontsize=8)
fig.text(0.012, 0.975, "a", fontweight="bold", fontsize=10)
fig.text(0.04, 0.975, "vorticity change from adding a vortex (peak 3× rms u) to a NS-Sines flow", fontsize=8.5)

CONDS = ["real input", "anchor only", "+ comp. (rest)", "+ comp. (other)", "+ random", "insertion", "insertion, flip",
         "insertion, random"]
ax = fig.add_axes([0.075, 0.115, 0.50, 0.235])
for q in range(8):
    for s_, (name, lab) in enumerate((("b1", "step 1"), ("b10", "step 10"))):
        nm, h, ds = B[2 * q + s_]; assert nm == name
        x = q + (s_ - 0.5) * 0.36
        ax.bar(x, h, 0.34, color=COL[name], label=lab if q == 0 else None)
        ax.scatter(np.full(4, x), ds, s=4, color="k", lw=0, zorder=3)
ax.axhline(0, color="0.5", lw=0.5); ax.set_ylim(*YLIM); ax.set_xticks(range(8))
ax.set_xticklabels(CONDS, rotation=35, ha="right"); ax.legend(frameon=False, loc="upper right")
ax.set_ylabel("correlation")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.text(0.012, 0.375, "b", fontweight="bold", fontsize=10)
fig.text(0.04, 0.375, "correlation with the solver's vortex effect", fontsize=8.5)

ax = fig.add_axes([0.675, 0.115, 0.315, 0.235])
ADD = ["whole latent", "vortex disk", "component\nmoved"]
for q in range(3):
    for s_, (name, lab) in enumerate((("rest", "h(b) − h(0)"), ("other", "h(c+b) − h(c)"))):
        nm, h, ds = C[2 * q + s_]; assert nm == name
        x = q + (s_ - 0.5) * 0.36
        ax.bar(x, h, 0.34, color=COL[name], label=lab if q == 0 else None)
        ax.scatter(np.full(4, x), ds, s=4, color="k", lw=0, zorder=3)
ax.axhline(0, color="0.5", lw=0.5); ax.set_ylim(*YLIM); ax.set_xticks(range(3)); ax.set_xticklabels(ADD)
ax.legend(frameon=False, loc="upper right"); ax.set_ylabel("cosine")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.text(0.615, 0.375, "c", fontweight="bold", fontsize=10)
fig.text(0.643, 0.375, "cos with h(a+b) − h(a)", fontsize=8.5)

fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print("written", OUT)
print("b means (step 1, step 10):", [(round(B[2 * q][1], 3), round(B[2 * q + 1][1], 3)) for q in range(8)])
print("c means (rest, other):", [(round(C[2 * q][1], 3), round(C[2 * q + 1][1], 3)) for q in range(3)])
