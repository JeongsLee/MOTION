"""NCS Figure 3 panels.
a) 6-family matched-budget convergence curves: MOTION (bylfa) vs BCAT/PROSE-FD matched retrains.
b) 4-family qualitative contour: GT | MOTION | BCAT | |MOTION-GT| | |BCAT-GT|
   (rows: shallow_water h, com_ns PRESSURE, pdearena_uncond smoke, cfdbench Vx).
Data: /eu train.logs (curves) + /eu/_bfa_*.npz / _bcatf_*.npz (final-weight dumps, verified 08-07).
Outputs -> /code-vol/motion_fv2/figs_out/fig3{a_curves,b_contour}.png
"""
import os, re
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

OUT = "/code-vol/motion_fv2/figs_out"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17, "legend.fontsize": 14,
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    "axes.linewidth": 1.1, "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.6, "grid.color": "#9aa0a6",
    "legend.frameon": False, "savefig.dpi": 220,
    "axes.spines.top": True, "axes.spines.right": True,
})
C_OUR, C_BCAT, C_PROSE = "#E4572E", "#2E5EAA", "#8A8A8A"

FAMS = ["shallow_water", "com_ns", "incom_ns", "pdearena_ns", "cfdbench", "pdearena_uncond"]
NICE = {"shallow_water": "Shallow Water", "com_ns": "Compressible NS",
        "incom_ns": "Incompressible NS", "pdearena_ns": "PDEArena NS (conditioned)",
        "cfdbench": "CFDBench", "pdearena_uncond": "PDEArena NS (unconditioned)"}
FAIRMAP = {"shallow_water": "shallow_water", "com_ns": "com_ns", "incom_ns": "incom_ns",
           "pdearena_ns": "incom_ns_arena", "cfdbench": "cfdbench",
           "pdearena_uncond": "incom_ns_arena_u"}
INV = {v: k for k, v in FAIRMAP.items()}
SAMPLES_PER_EPOCH = 16000
FINAL_X = 160000 * 8


def parse_ours(p):
    S = []
    for L in open(p, errors="ignore"):
        m = re.search(r"\[EVAL step (\d+)\].*overall=([\d.]+)%.*shallow_water=([\d.]+)%\s+"
                      r"com_ns=([\d.]+)%\s+incom_ns=([\d.]+)%\s+pdearena_ns=([\d.]+)%\s+"
                      r"cfdbench=([\d.]+)%\s+pdearena_uncond=([\d.]+)%", L)
        if m:
            v = [float(x) for x in m.groups()]
            S.append((int(v[0]) * 8, v[1], dict(zip(FAMS, v[2:]))))
    S.sort()
    return S


def parse_fair(p):
    ep, per = -1, {}
    for L in open(p, errors="ignore"):
        m = re.search(r"End of epoch (\d+)", L)
        if m:
            ep = int(m.group(1)); continue
        m = re.search(r"FULLTEST_RELNORM\s+(\S+)\s+n=100\s+mean=([\d.]+)%", L)
        if m and m.group(1) in INV:
            per.setdefault(ep, {}).setdefault(INV[m.group(1)], []).append(float(m.group(2)))
    S = []
    for e in sorted(per):
        d = per[e]
        if e < 0 or not all(f in d for f in FAMS):
            continue
        fd = {f: d[f][0] for f in FAMS}
        S.append(((e + 1) * SAMPLES_PER_EPOCH, float(np.mean(list(fd.values()))), fd))
    return S


ours = parse_ours("/eu/results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa/train.log")
bcat = parse_fair("/eu/results/p1_bcat_fair/v1/train.log")
prose = parse_fair("/eu/results/p1_prose_fair/v1/train.log")
for nm, S in (("ours", ours), ("bcat", bcat), ("prose", prose)):
    assert S and S[-1][0] == FINAL_X, (nm, S[-1][0] if S else None)
    print(f"{nm}: n={len(S)} overall_end={S[-1][1]:.3f}", flush=True)

# class-average crossings (quoted in the caption)
xo = np.array([s[0] for s in ours], float); yo = np.array([s[1] for s in ours], float)
for nm, S in (("BCAT", bcat), ("PROSE", prose)):
    tgt = S[-1][1]
    k = np.argmax(yo <= tgt)
    x_at = xo[k] if k == 0 or yo[k - 1] <= tgt else np.interp(
        np.log(tgt), np.log([yo[k], yo[k - 1]]), np.log([xo[k], xo[k - 1]]))
    x_at = float(np.exp(x_at)) if not np.isscalar(x_at) or x_at < 1e3 else float(x_at)
    print(f"crossing vs {nm} final {tgt:.3f}%: x={x_at:.3g} ({x_at/FINAL_X:.2f}x budget)", flush=True)

XT = [2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
xfmt = FuncFormatter(lambda v, _: (f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k"))
yfmt = FuncFormatter(lambda v, _: f"{v:g}")


def style(ax, xlab=False, ylab=False):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(XT)
    ax.xaxis.set_major_formatter(xfmt); ax.xaxis.set_minor_formatter(NullFormatter())
    lo, hi = ax.get_ylim()
    dec = np.log10(hi / lo)
    subs = (2, 3, 4, 5, 6, 7, 8, 9) if dec < 0.8 else (2, 3, 4, 6) if dec < 1.5 else (2, 3, 5)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(yfmt); ax.yaxis.set_minor_formatter(yfmt)
    ax.grid(True, which="both", alpha=0.18)
    if xlab: ax.set_xlabel("training samples seen")
    if ylab: ax.set_ylabel(r"rel-$L_2$ (%)")


SER = [(ours, C_OUR, "-", "o", "MOTION"),
       (bcat, C_BCAT, "--", "s", "BCAT (matched retrain)"),
       (prose, C_PROSE, "-.", "^", "PROSE-FD (matched retrain)")]

fig, axs = plt.subplots(2, 3, figsize=(15.0, 8.4))
for i, (ax, fam) in enumerate(zip(axs.ravel(), FAMS)):
    for S, c, ls, mk, lab in SER:
        x = [s[0] for s in S]; y = [s[2][fam] for s in S]
        ax.plot(x, y, ls, color=c, lw=2.2, label=lab, zorder=4 if c == C_OUR else 3)
        ax.plot([x[-1]], [y[-1]], marker=mk, color=c, ms=8, mec="white", mew=0.8, zorder=5)
    ax.set_title(NICE[fam], pad=8)
    style(ax, xlab=(i >= 3), ylab=(i % 3 == 0))
axs[0, 0].legend(loc="lower left", handlelength=2.4)
fig.tight_layout(pad=1.1, w_pad=1.6, h_pad=1.8)
fig.savefig(f"{OUT}/fig3a_curves.png", bbox_inches="tight")
plt.close(fig)
print("FIG3A_DONE", flush=True)

# ---------------- panel b: 4-family qualitative contour ----------------
# (ours_fam, bcat_fam, ours slots, plotted channel within slots, frames, row label)
ROWS = [("shallow_water", "shallow_water", [5], 0, 10, "Shallow Water\n($h$)"),
        ("com_ns", "com_ns", [0, 1, 3, 4], 3, 10, "Compressible NS\n(pressure)"),
        ("pdearena_uncond", "incom_ns_arena_u", [0, 1, 2], 2, 4, "PDEArena NS\n(uncond., smoke)"),
        ("cfdbench", "cfdbench", [0, 1, 2], 0, 10, "CFDBench\n($V_x$)")]
cols = ["ground truth", "MOTION", "BCAT", r"$|$MOTION $-$ GT$|$", r"$|$BCAT $-$ GT$|$"]

fig, axs = plt.subplots(4, 5, figsize=(14.4, 2.9 * 4))
for r, (fa, fb, slots, pc, F, lab) in enumerate(ROWS):
    A = np.load(f"/eu/_bfa_{fa}.npz")
    aP = A["pred"][..., slots][:, :F]; aG = A["gt"][..., slots][:, :F]
    Bz = np.load(f"/eu/_bcatf_{fb}.npz"); nc = len(slots)
    bP = Bz["pred"][..., :nc][:, :F]; bG = Bz["gt"][..., :nc][:, :F]
    ns = min(len(aP), len(bP)); aP, aG, bP, bG = aP[:ns], aG[:ns], bP[:ns], bG[:ns]
    gerr = np.abs(aG - bG).max()
    assert gerr < 1e-4, (fa, gerr)                     # GT alignment guard
    fr = F - 1
    s = int(np.argmax([aG[i, fr, ..., pc].std() for i in range(ns)]))
    gtc = aG[s, fr, ..., pc]; a = aP[s, fr, ..., pc]; b = bP[s, fr, ..., pc]
    c00 = gtc.mean(); amp = np.abs(gtc - c00).max() * 0.92
    emax = max(np.abs(a - gtc).max(), np.abs(b - gtc).max()) * 0.85 + 1e-9
    ims = [(gtc, "RdBu_r", c00 - amp, c00 + amp), (a, "RdBu_r", c00 - amp, c00 + amp),
           (b, "RdBu_r", c00 - amp, c00 + amp),
           (np.abs(a - gtc), "inferno", 0, emax), (np.abs(b - gtc), "inferno", 0, emax)]
    for c, (im, cm, lo, hi) in enumerate(ims):
        axs[r, c].imshow(im, cmap=cm, vmin=lo, vmax=hi)
        axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        for sp in axs[r, c].spines.values():
            sp.set_linewidth(1.0); sp.set_color("#555")
        if r == 0:
            axs[r, c].set_title(cols[c], fontsize=15, pad=8)

    def rl2(p, g):
        o = p[s, fr]; t = g[s, fr]
        return 100 * np.sqrt(((o - t) ** 2).sum()) / np.sqrt((t ** 2).sum())
    axs[r, 0].set_ylabel(lab, fontsize=13.5, rotation=90, va="center")
    bb = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    axs[r, 3].text(0.5, 0.06, f"{rl2(aP, aG):.1f}%", color="k", fontsize=12.5, ha="center",
                   transform=axs[r, 3].transAxes, bbox=bb)
    axs[r, 4].text(0.5, 0.06, f"{rl2(bP, bG):.1f}%", color="k", fontsize=12.5, ha="center",
                   transform=axs[r, 4].transAxes, bbox=bb)
    print(f"contour {fa}: MOTION {rl2(aP, aG):.2f}% BCAT {rl2(bP, bG):.2f}% (n={ns},sample={s})", flush=True)
fig.tight_layout(w_pad=0.4, h_pad=0.6)
fig.savefig(f"{OUT}/fig3b_contour.png", bbox_inches="tight")
print("FIG3B_DONE", flush=True)
