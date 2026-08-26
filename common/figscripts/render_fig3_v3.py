"""Fig3 v3 panels.
a) OVERALL class-averaged accuracy curve (3 models, log-log) with final-value guides
   and budget-crossing marks.
d) 4-case qualitative contour, rows labelled by MECHANISM:
   Gravity waves (SWE h) / Compressibility (com_ns pressure) /
   Turbulent transport (pdearena_ns conditioned smoke, MOST-TURBULENT sample) /
   Advective wake (cfdbench Vx). Non-turbulent rows keep the lowest-error sample.
e) matching 1x4 error-spectra strip (Supplementary/ED candidate).
Outputs -> /code-vol/motion_fv2/figs_out/{fig3a_overall,fig3d_contour_v3,fig3e_spectra_ed}.png
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
    "font.size": 17, "axes.labelsize": 19, "axes.titlesize": 18, "legend.fontsize": 14.5,
    "xtick.labelsize": 15.5, "ytick.labelsize": 15.5,
    "axes.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.6, "grid.color": "#9aa0a6",
    "legend.frameon": False, "savefig.dpi": 220,
    "axes.spines.top": True, "axes.spines.right": True,
})
C_OUR, C_BCAT, C_PROSE = "#E4572E", "#2E5EAA", "#8A8A8A"
FAMS = ["shallow_water", "com_ns", "incom_ns", "pdearena_ns", "cfdbench", "pdearena_uncond"]
INV = {"shallow_water": "shallow_water", "com_ns": "com_ns", "incom_ns": "incom_ns",
       "incom_ns_arena": "pdearena_ns", "cfdbench": "cfdbench",
       "incom_ns_arena_u": "pdearena_uncond"}


def parse_ours(p):
    S = []
    for L in open(p, errors="ignore"):
        m = re.search(r"\[EVAL step (\d+)\].*overall=([\d.]+)%", L)
        if m:
            S.append((int(m.group(1)) * 8, float(m.group(2))))
    return sorted(S)


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
        S.append(((e + 1) * 16000, float(np.mean([d[f][0] for f in FAMS]))))
    return S


ours = parse_ours("/eu/results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa/train.log")
bcat = parse_fair("/eu/results/p1_bcat_fair/v1/train.log")
prose = parse_fair("/eu/results/p1_prose_fair/v1/train.log")
print("finals:", ours[-1], bcat[-1], prose[-1], flush=True)

xo = np.array([s[0] for s in ours], float); yo = np.array([s[1] for s in ours], float)
CROSS = {}
for nm, S in (("BCAT", bcat), ("PROSE", prose)):
    tgt = S[-1][1]
    k = int(np.argmax(yo <= tgt))
    xat = float(np.exp(np.interp(np.log(tgt), np.log([yo[k], yo[k - 1]]), np.log([xo[k], xo[k - 1]])))) \
        if k > 0 and yo[k - 1] > tgt else float(xo[k])
    CROSS[nm] = (xat, tgt)
    print(f"cross {nm}: {xat:.3g} @ {tgt}", flush=True)

fig, ax = plt.subplots(figsize=(6.4, 6.6))
for S, c, ls, mk, lab, z in [((ours), C_OUR, "-", "o", "MOTION", 5),
                             ((bcat), C_BCAT, "--", "s", "BCAT (matched retrain)", 4),
                             ((prose), C_PROSE, "-.", "^", "PROSE-FD (matched retrain)", 3)]:
    x = [s[0] for s in S]; y = [s[1] for s in S]
    ax.plot(x, y, ls, color=c, lw=2.3, label=lab, zorder=z)
    ax.plot([x[-1]], [y[-1]], marker=mk, color=c, ms=8.5, mec="white", mew=0.8, zorder=6)
for nm, c in (("BCAT", C_BCAT), ("PROSE", C_PROSE)):
    xat, tgt = CROSS[nm]
    ax.plot([xat, 1.28e6], [tgt, tgt], ":", color=c, lw=1.4, zorder=2)
    ax.plot([xat], [tgt], marker="|", color=c, ms=13, mew=2.2, zorder=6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([2e4, 5e4, 1e5, 2e5, 5e5, 1e6])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if abs(v - round(v)) < 1e-9 and v < 10 else ""))
ax.set_xlabel("training samples seen")
ax.set_ylabel(r"class-averaged rel-$L_2$ (%)")
ax.legend(loc="upper right", handlelength=2.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3a_overall.png", bbox_inches="tight")
print("FIG3A_OVERALL_DONE", flush=True)
plt.close(fig)

# ---------------- panel d: mechanism-labelled contour ----------------
# (ours_fam, bcat_fam, slots, ch, frames, mech label, selection rule)
ROWS = [("shallow_water", "shallow_water", [5], 0, 10, "Gravity waves", "maxerr"),
        ("com_ns", "com_ns", [0, 1, 3, 4], 3, 10, "Compressibility", "maxerr"),
        ("pdearena_ns", "incom_ns_arena", [0, 1, 2], 2, 10, "Turbulent transport", "maxstd"),
        ("cfdbench", "cfdbench", [0, 1, 2], 0, 10, "Advective wake", "maxerr")]
cols = ["ground truth", "MOTION", "BCAT", r"$|$MOTION $-$ GT$|$", r"$|$BCAT $-$ GT$|$"]

CHOSEN = []
fig, axs = plt.subplots(4, 5, figsize=(14.4, 2.9 * 4))
for r, (fa, fb, slots, pc, F, lab, rule) in enumerate(ROWS):
    A = np.load(f"/eu/_bfa_{fa}.npz")
    aP = A["pred"][..., slots][:, :F]; aG = A["gt"][..., slots][:, :F]
    Bz = np.load(f"/eu/_bcatf_{fb}.npz"); nc = len(slots)
    bP = Bz["pred"][..., :nc][:, :F]; bG = Bz["gt"][..., :nc][:, :F]
    ns = min(len(aP), len(bP)); aP, aG, bP = aP[:ns], aG[:ns], bP[:ns]
    assert np.abs(aG - bG[:ns, :F, ..., :nc]).max() < 1e-4
    fr = F - 1

    def rl2(p, i):
        o = p[i, fr]; t = aG[i, fr]
        return 100 * np.sqrt(((o - t) ** 2).sum()) / np.sqrt((t ** 2).sum())
    for i in range(ns):
        print(f"    {fa} sample{i}: std={aG[i, fr, ..., pc].std():.4g} "
              f"MOTION {rl2(aP, i):.2f}% BCAT {rl2(bP, i):.2f}%", flush=True)
    if rule == "maxstd":
        s = int(np.argmax([aG[i, fr, ..., pc].std() for i in range(ns)]))
    elif rule == "maxerr":
        s = int(np.argmax([rl2(aP, i) + rl2(bP, i) for i in range(ns)]))
    else:
        s = int(np.argmin([rl2(aP, i) + rl2(bP, i) for i in range(ns)]))
    CHOSEN.append((fa, slots, pc, F, fr, s, lab))
    print(f"{fa} [{rule}]: sample={s} MOTION {rl2(aP, s):.2f}% BCAT {rl2(bP, s):.2f}%", flush=True)
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
    axs[r, 0].set_ylabel(lab, fontsize=14, rotation=90, va="center")
    bb = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    axs[r, 3].text(0.5, 0.06, f"{rl2(aP, s):.1f}%", color="k", fontsize=12.5, ha="center",
                   transform=axs[r, 3].transAxes, bbox=bb)
    axs[r, 4].text(0.5, 0.06, f"{rl2(bP, s):.1f}%", color="k", fontsize=12.5, ha="center",
                   transform=axs[r, 4].transAxes, bbox=bb)
fig.tight_layout(w_pad=0.4, h_pad=0.6)
fig.savefig(f"{OUT}/fig3d_contour_v3.png", bbox_inches="tight")
print("FIG3D_DONE", flush=True)
plt.close(fig)

# ---------------- Supp/ED: matching 1x4 error-spectra strip ----------------
def spec1d(f):
    f = f - f.mean()
    Fh = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    n = f.shape[0]
    ky, kx = np.indices(Fh.shape)
    kr = np.sqrt((kx - n / 2) ** 2 + (ky - n / 2) ** 2).astype(int)
    kmax = n // 2
    E = np.bincount(kr.ravel(), Fh.ravel(), minlength=kmax + 1)[:kmax + 1]
    cnt = np.bincount(kr.ravel(), minlength=kmax + 1)[:kmax + 1]
    return np.arange(kmax + 1), E / np.maximum(cnt, 1)


fig, axs = plt.subplots(1, 4, figsize=(15.0, 3.7))
for ax, (fa, slots, pc, F, fr, s, lab) in zip(axs, CHOSEN):
    A = np.load(f"/eu/_bfa_{fa}.npz")
    aP = A["pred"][..., slots][:, :F]; aG = A["gt"][..., slots][:, :F]
    fbmap = {"shallow_water": "shallow_water", "com_ns": "com_ns",
             "pdearena_ns": "incom_ns_arena", "cfdbench": "cfdbench"}
    Bz = np.load(f"/eu/_bcatf_{fbmap[fa]}.npz")
    bP = Bz["pred"][..., :len(slots)][:, :F]
    gtc = aG[s, fr, ..., pc]; a = aP[s, fr, ..., pc]; b = bP[s, fr, ..., pc]
    k, Eg = spec1d(gtc); _, Ea = spec1d(a - gtc); _, Eb = spec1d(b - gtc)
    ax.loglog(k[1:], Eg[1:] + 1e-30, "-", color="k", lw=2.5, label="ground truth", zorder=3)
    ax.loglog(k[1:], Ea[1:] + 1e-30, "-", color=C_OUR, lw=1.9, label="MOTION error", zorder=5)
    ax.loglog(k[1:], Eb[1:] + 1e-30, "--", color=C_BCAT, lw=1.9, label="BCAT error", zorder=4)
    ax.set_title(lab, fontsize=14, pad=6)
    ax.set_xlabel("wavenumber $k$")
    ax.set_xlim(1, 64)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.15)
axs[0].set_ylabel("$E(k)$")
axs[0].legend(loc="lower left", handlelength=1.9, fontsize=11.5)
fig.tight_layout(w_pad=1.4)
fig.savefig(f"{OUT}/fig3e_spectra_ed.png", bbox_inches="tight")
print("FIG3E_DONE", flush=True)
