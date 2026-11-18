"""R1 #1: in-training validation trajectories of MOTION-S across the two stages, and the stage-2 continuation control.
Stage 1 (published, amplitude loss, lr 3e-4 const, 2 H100 x 4) to 134k; stage 2 branches from the stage-1 best (110k):
published stage 2, its replay (second-stage loss) and the control continuing the first-stage loss, all with the
second-stage schedule (lr 5e-5 after 1k warmup, 1 H100 x 4). Monitor = class-averaged physical rel-L2 on 8 validation
trajectories per family, every 2,000 steps.   python fig_staged_curves.py
"""
import os, re, json
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
for _f in ("times.ttf", "timesbd.ttf", "timesi.ttf"):
    _p = f"/mnt/c/Windows/Fonts/{_f}"
    if os.path.exists(_p): fm.fontManager.addfont(_p)
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..")
D = os.path.join(R, "results", "A14_factorial", "data"); OUT = os.path.join(R, "figures", "out")
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 8.5, "axes.linewidth": 0.8, "savefig.dpi": 400})


def curve(p):
    st, out = 0, []
    for ln in open(p, errors="ignore"):
        m = re.search(r"step (\d+) loss", ln)
        if m: st = int(m.group(1))
        m = re.search(r"\[EVAL-PHYS\] class-avg rel-L2 ([0-9.]+)%", ln)
        if m: out.append((int(round(st / 2000) * 2000), float(m.group(1))))
    return out


S1 = curve(f"{D}/stage_logs/motion_m1_stage1_train.log")
PUB = curve(f"{D}/stage_logs/motion_m1f_stage2_train.log")
REP = curve(f"{D}/staged_s2/train.log"); CTL = curve(f"{D}/staged_s1/train.log")
B = 110000; s1best = dict(S1)[B]
C = {"s1": "0.35", "pub": "#1a4f8a", "rep": "#5b8fc9", "ctl": "#c0392b"}

fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1.55, 1]})
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.9, wspace=0.22)
a.plot([s / 1e3 for s, _ in S1], [v for _, v in S1], color=C["s1"], lw=1.2, label="stage 1 (amplitude loss, lr $3\\times10^{-4}$)")
for k, cur, lab, ls in (("pub", PUB, "published stage 2", "-"), ("rep", REP, "replay, second-stage loss", "--"),
                        ("ctl", CTL, "control, first-stage loss", ":")):
    xs = [B / 1e3] + [(B + s) / 1e3 for s, _ in cur]; ys = [s1best] + [v for _, v in cur]
    a.plot(xs, ys, color=C[k], lw=1.2, ls=ls, label=lab)
    b.plot([0] + [s / 1e3 for s, _ in cur], ys, color=C[k], lw=1.4, ls=ls, marker="o", ms=2.4, label=lab)
a.axvline(B / 1e3, color="0.6", lw=0.6, ls=(0, (3, 2)))
a.text(B / 1e3 - 1.5, 25.5, "stage-2 restart\nfrom stage-1 best", ha="right", va="top", fontsize=7, color="0.35")
a.set_yscale("log"); a.set_yticks([8, 10, 12, 15, 20, 25]); a.set_yticklabels(["8", "10", "12", "15", "20", "25"])
a.minorticks_off(); a.set_xlim(0, 136); a.set_ylim(7.6, 27)
a.set_xlabel("optimizer step ($\\times10^3$)"); a.set_ylabel("validation error (%)")
a.legend(frameon=False, fontsize=6.8, loc="lower left", bbox_to_anchor=(0.06, 0.0))
b.axhline(s1best, color="0.6", lw=0.6, ls=(0, (3, 2))); b.text(19.5, s1best - 0.06, f"stage-1 best {s1best:.2f}%", ha="right", va="top", fontsize=7, color="0.35")
b.set_xlim(-0.5, 24.5); b.set_xlabel("steps after restart ($\\times10^3$)"); b.set_ylabel("validation error (%)")
for ax, t in ((a, "a"), (b, "b")):
    ax.set_title(t, loc="left", fontweight="bold", fontsize=10, x=-0.08)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_staged_curves.{ext}"), bbox_inches="tight")
json.dump({"stage1": S1, "stage2_published": PUB, "replay": REP, "control": CTL, "branch_step": B},
          open(os.path.join(OUT, "fig_staged_curves.json"), "w"))
print("stage1 best", s1best, "| at 20k: pub", dict(PUB)[20000], "rep", dict(REP)[20000], "ctl", dict(CTL)[20000])
