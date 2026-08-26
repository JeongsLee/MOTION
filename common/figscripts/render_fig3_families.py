"""Fig3 per-family panels v2 (08-11): each panel = 5-col qual row + numeric table block;
Wave (d) = TWO qual rows (official + repaired two-field IC) sharing one table.

Inputs (fig3_local/preds/): {task}_motion.npz (+_f32 copies for NS/ACE — numeric mode that
reproduces the official eval; see FIGURES.md 08-11), {task}_scot.npz (Poisson -> _scot2
aligned), Wave-Layer-rep_{motion,scot}.npz (repaired protocol, anchor 0).
Annotated numbers = official evaluation records (job logs), NOT re-inference medians.
Run: python3 render_fig3_families.py [--check]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
LOC = "/mnt/d/working/2026/neuraladaf/v6/fig3_local"
OUT = os.path.join(HERE, "..", "figs")

import matplotlib.font_manager as _fm
for _f in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _fm.fontManager.addfont(f"/mnt/c/Windows/Fonts/{_f}")
plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": 24, "axes.titlesize": 27, "savefig.dpi": 220,
})
C_OUR, C_POS = "#E4572E", "#2E5EAA"

TASKS = {
    "NS-PwC":        dict(slot=0, sch=0, sample=int(os.environ.get("S_NS", 0)), ar=True),
    "ACE":           dict(slot=2, sch=0, sample=int(os.environ.get("S_ACE", 0)), ar=False),
    "Poisson-Gauss": dict(slot=2, sch=0, sample=int(os.environ.get("S_PG", 0)), ar=False),
    "Wave-Layer":    dict(slot=2, sch=0, sample=int(os.environ.get("S_WV", 0)), ar=False),
}
# per-panel table rows: (label, motion, poseidon, motion_wins)
TABLES = {
    "NS-PwC": [("single-shot", "5.51", "5.29", False),
               ("autoregressive", "3.30", "4.35", True)],
    "ACE": [(r"budget $\times$1", "1.39", "1.26", False),
            (r"budget $\times$4", "0.59", "0.70", True)],
    "Poisson-Gauss": [("steady solve", "6.25", "11.17", True)],
    "Wave-Layer": [("official, $u$", "31.27", "16.67", False),
                   (r"repaired, $u,v_0$", "11.89", "12.97", True)],
}

ENS = int(os.environ.get("ENS", "1"))                        # 1: MOTION column = symmetry-ensemble preds

def load(task):
    if ENS and os.path.exists(f"{LOC}/preds/{task}_motion_ens.npz"):
        mf = f"{LOC}/preds/{task}_motion_ens.npz"
    else:
        f32 = f"{LOC}/preds/{task}_motion_f32.npz"
        mf = f32 if task in ("NS-PwC", "ACE") and os.path.exists(f32) \
            else f"{LOC}/preds/{task}_motion.npz"
    dm = np.load(mf)
    if ENS and task == "Poisson-Gauss" and os.path.exists(f"{LOC}/preds/{task}_scot_ens2.npz"):
        sf = f"{LOC}/preds/{task}_scot_ens2.npz"
    elif ENS and os.path.exists(f"{LOC}/preds/{task}_scot_ens.npz"):
        sf = f"{LOC}/preds/{task}_scot_ens.npz"
    elif task == "Poisson-Gauss" and os.path.exists(f"{LOC}/preds/{task}_scot2.npz"):
        sf = f"{LOC}/preds/{task}_scot2.npz"
    else:
        sf = f"{LOC}/preds/{task}_scot.npz"
    return dm, np.load(sf)

def motion_phys(dm, arr16, slot):
    sc = dm["std"][:16, None, None, None, slot]
    mc = dm["mean"][:16, None, None, None, slot]
    return arr16[..., slot]*sc + mc

def check_alignment():
    for task, t in TASKS.items():
        try:
            dm, ds = load(task)
        except FileNotFoundError as e:
            print(task, "MISSING", e); continue
        g_m = motion_phys(dm, dm["gt16"], t["slot"])
        g_s = ds["ref16"][:, :, t["sch"]]
        Tn = g_s.shape[1]
        d = np.linalg.norm(g_m[:, -Tn:] - g_s)/np.linalg.norm(g_s)
        print(f"{task}: gt-align rel diff = {d:.2e}")
    for tag in ["motion", "scot"]:
        f = f"{LOC}/preds/Wave-Layer-rep_{tag}.npz"
        print("wave-rep", tag, "present" if os.path.exists(f) else "MISSING")

def draw_row(axs, gt, pm, ps, first=False, tag="", note="", emax=None):
    vmax = np.abs(gt).max()*0.9; vmin = -vmax
    if gt.min() >= 0:
        vmin, vmax = gt.min(), gt.max()
    em, ep = np.abs(pm-gt), np.abs(ps-gt)
    if emax is None:
        emax = max(em.max(), ep.max())*0.85
    ims = [(gt, "RdBu_r", vmin, vmax), (pm, "RdBu_r", vmin, vmax), (ps, "RdBu_r", vmin, vmax),
           (em, "magma", 0, emax), (ep, "magma", 0, emax)]
    titles = ["Ground truth", "MOTION", "Poseidon-B",
              r"$|$MOTION $-$ GT$|$", r"$|$Poseidon-B $-$ GT$|$"]
    for j, (arr, cmapn, v0, v1) in enumerate(ims):
        ax = axs[j]
        ax.imshow(arr, cmap=cmapn, vmin=v0, vmax=v1, origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        if first:
            ax.set_title(titles[j], fontsize=21.5, pad=5)
    if note:
        axs[0].text(0.03, 0.95, note, color="k", fontsize=21.275, va="top",
                    transform=axs[0].transAxes,
                    bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))
    if tag:
        axs[0].set_ylabel(tag, fontsize=24.975)

def draw_table(ax, rows, title="median rel. $L_1$ (%)", y0=0.70, dy=0.26):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    T = ax.transAxes
    ax.text(0.04, 0.95, title, fontsize=23.125, va="top", color="#333", transform=T)
    ax.text(0.58, y0 + 0.10, "MOTION", fontsize=20.35, color=C_OUR, fontweight="bold",
            ha="center", transform=T)
    ax.text(0.90, y0 + 0.10, "Poseidon-B", fontsize=20.35, color=C_POS, fontweight="bold",
            ha="center", transform=T)
    ax.plot([0.04, 0.99], [y0 + 0.04, y0 + 0.04], color="#999", lw=0.8,
            transform=T, clip_on=False)
    for k, (lab, vm, vp, mwin) in enumerate(rows):
        y = y0 - (k + 0.45)*dy
        ax.text(0.02, y, lab, fontsize=21.275, va="center", color="#222", transform=T)
        ax.text(0.58, y, vm, fontsize=23.125, va="center", ha="center", color=C_OUR,
                fontweight="bold" if mwin else "normal", transform=T)
        ax.text(0.90, y, vp, fontsize=23.125, va="center", ha="center", color=C_POS,
                fontweight="bold" if not mwin else "normal", transform=T)

def get_fields(task):
    t = TASKS[task]; dm, ds = load(task); s = t["sample"]
    src = "predAR16" if t["ar"] else "pred16"
    pm = motion_phys(dm, dm[src], t["slot"])[s, -1]
    if t["ar"] and ENS and os.path.exists(f"{LOC}/preds/NS-PwC_scot_arens.npz"):
        ps = np.load(f"{LOC}/preds/NS-PwC_scot_arens.npz")["predAR16"][s, -1, t["sch"]]
    else:
        ps = ds["predAR16" if t["ar"] else "pred16"][s, -1, t["sch"]]
    gt = motion_phys(dm, dm["gt16"], t["slot"])[s, -1]
    return gt, pm, ps

def get_wave_repaired(sample):
    f = f"{LOC}/preds/Wave-Layer-rep_motion_ens.npz"
    dm = np.load(f if ENS and os.path.exists(f) else f"{LOC}/preds/Wave-Layer-rep_motion.npz")
    fs = f"{LOC}/preds/Wave-Layer-rep_scot_ens.npz"
    ds = np.load(fs if ENS and os.path.exists(fs) else f"{LOC}/preds/Wave-Layer-rep_scot.npz")
    pm = motion_phys(dm, dm["pred16"], 2)[sample, -1]
    ps = ds["pred16"][sample, -1, 0]
    gt = motion_phys(dm, dm["gt16"], 2)[sample, -1]
    return gt, pm, ps

def render():
    # FIXED margins (no tight crop) so every panel shares the same left edge (user:
    # Wave rows must left-align with the other contours). a keeps a title strip.
    W, LM, RM = 13.4, 0.052, 0.999
    for task, letter, tag, top in [("NS-PwC", "a", "NS-PwC\n(in-family flow)", 0.845),
                                   ("ACE", "b", "ACE\n(OOD reaction)", 0.995),
                                   ("Poisson-Gauss", "c", "Poisson-Gauss\n(OOD steady)", 0.995)]:
        h = 2.72 if letter != "a" else 2.72/0.845*0.995
        fig, axs = plt.subplots(1, 5, figsize=(W, h))
        fig.subplots_adjust(wspace=0.03, left=LM, right=RM, top=top, bottom=0.005)
        gt, pm, ps = get_fields(task)
        note = "matched AR, velocity" if task == "NS-PwC" else ""
        draw_row(axs, gt, pm, ps, first=(letter == "a"), tag=tag, note=note)
        f = f"{OUT}/fig3{letter}_{task.lower().replace('-','')}.png"
        fig.savefig(f, pad_inches=0); plt.close(fig)
        print("saved", f)
    fig, axs = plt.subplots(2, 5, figsize=(W, 5.5))
    fig.subplots_adjust(wspace=0.03, hspace=0.04, left=LM, right=RM, top=0.995, bottom=0.005)
    s_ = TASKS["Wave-Layer"]["sample"]
    gt, pm, ps = get_fields("Wave-Layer")
    gtr, pmr, psr = get_wave_repaired(s_)
    emax = max(np.abs(pm-gt).max(), np.abs(ps-gt).max(),
               np.abs(pmr-gtr).max(), np.abs(psr-gtr).max())*0.85
    draw_row(axs[0], gt, pm, ps, first=False, tag="", note="ill-posed ($u$ only)", emax=emax)
    draw_row(axs[1], gtr, pmr, psr, first=False, tag="", note="closed ($u,v_0$)", emax=emax)
    fig.text(LM * 0.42, 0.5, "Wave-Layer\n(OOD hyperbolic)", rotation=90,
             va="center", ha="center", fontsize=24.975)
    f = f"{OUT}/fig3d_wavelayer.png"
    fig.savefig(f, pad_inches=0); plt.close(fig)
    print("saved", f)

if __name__ == "__main__":
    if "--check" in sys.argv:
        check_alignment()
    else:
        check_alignment(); render()
