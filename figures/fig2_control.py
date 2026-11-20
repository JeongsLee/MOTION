"""Main Fig. 2 (revision R1 redraw, 2026-10-19): the latent transport velocity is a shared, calibrated physical variable.
  a  uniform command, five families (10-step rollout, D* = +4 cells along x): unsteered forecast (top) and commanded
     forecast (bottom); centre line at x = n/2 (top) and at n/2 + 4 (bottom); arrow = commanded shift.
  b  one-step response: commanded displacement alpha * eta_c (cells) vs achieved displacement (cells), all families and
     both axes; mean +- s.d. over held-out trajectories; one pooled line.
  c  calibration constants eta_x, eta_y (cells per s.d. of u(h) per step), fitted on calibration trajectories only.
  d  spatially varying command (shear, 4 cells peak over 10 steps): NS-Sines maps + pattern correlation per family.
  e  vortex command vs kinematic flow map and NS solver (NS-Sines): r over steps, vorticity response at step 10.
Data: motion_local/out_ctlk (a), out_ctl1g or out_ctl1 (b, c), out_ctl2 (d), out_v3 (e); MOTION-S (m1f) checkpoint.
  python fig2_control.py  -> 04_response/figs/fig2_control.{pdf,png,json}
"""
import os, json
FS = float(os.environ.get("FIG_FS", "1.0"))           # REVISION-R1 10-23 (user): font scale for the main figure
VS = float(os.environ.get("FIG_VS", "1.0"))           # vertical compaction: figure height and band spacing scale
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

for f in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"):
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "custom", "mathtext.it": "Liberation Sans:italic",
                     "mathtext.rm": "Liberation Sans", "font.size": 6 * FS, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "axes.titlesize": 6 * FS, "axes.titlepad": 3})
INK, MUTED = "#1a1a1a", "#7a7a7a"
VER, ORA, SKY, BLU, GRN, PUR = "#D55E00", "#E69F00", "#56B4E9", "#0072B2", "#009E73", "#CC79A7"
L = os.environ.get("MOTION_LOCAL", "motion_local")
SHIFT10 = os.environ.get("SHIFT10", "0") == "1"        # 1: panel a at 10 px (incomp. NS stays at 4 px: nonlinear at 10)
FIELDS = os.environ.get("FIELDS", "0") == "1"           # 1: c and e show the fields themselves, not the response
CDIFF = FIELDS and os.environ.get("CDIFF", "1") == "1"   # REVISION-R1 10-23 (user): no r in the field panels (c, f); c shows vorticity changes, commanded - base
#   and solver(command) - solver(no command), next to the base field; no r in c (velocity r stays in d). CDIFF=0: full fields (SI)
OUT = os.path.join(os.path.dirname(__file__), "out", "fig2_control" + ("_fields" if FIELDS else "") + ("_10px" if SHIFT10 else "") + ("_cdiff" if CDIFF else ""))
FAMS = ["NS-Sines", "incom_ns", "com_ns", "CE-RP", "CE-Gauss"]
LBL = {"NS-Sines": "NS-Sines", "incom_ns": "Incomp. NS", "com_ns": "Comp. NS", "CE-RP": "CE-RP", "CE-Gauss": "CE-Gauss"}
COL = {"NS-Sines": BLU, "incom_ns": SKY, "com_ns": GRN, "CE-RP": VER, "CE-Gauss": ORA}
SN = {0: "u_x", 1: "u_y", 3: "\\rho", 4: "p", 6: "c"}
J = {}
FLOW = []                                              # (base ax, command ax, commanded ax) -> "+" and arrow marks


def npz(path):
    z = np.load(path, allow_pickle=True); return z, json.loads(str(z["__meta__"]))


def tag(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8*FS, fontweight="bold", va="bottom", ha="right")


MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, (222 if os.environ.get("SHOW_FHN", "0") == "1" else 222 * 0.80 * VS) * MM if FIELDS else 190 * MM))
_FHN = os.environ.get("SHOW_FHN", "0") == "1"
G = (GridSpec(4 if _FHN else 3, 1, figure=fig, height_ratios=[0.95, 1.6, 0.965, 1.0][:4 if _FHN else 3], hspace=0.40 if _FHN else 0.40 * 3.515 / 4.515 * 4 / 3 * float(os.environ.get("FIG_HS", "1.0")), left=0.05, right=0.99, top=0.98, bottom=0.04)
     if FIELDS else
     GridSpec(3, 1, figure=fig, height_ratios=[1.25, 1.15, 0.95], hspace=0.40, left=0.05, right=0.99, top=0.975, bottom=0.045))
g0 = GridSpecFromSubplotSpec(1, 2, G[0], width_ratios=[1.75, 1.0], wspace=0.16)

# ---------------- a: uniform command, 10-step rollout, D* = +4 cells along x (full domain, contours; velocity u_x)
ga = GridSpecFromSubplotSpec(2, 5, g0[0], hspace=0.08, wspace=0.05)   # base / commanded (error row removed, user 10-20)
J["a"] = {}
for j, fam in enumerate(FAMS):
    DPX = 10.0 if (SHIFT10 and fam != "incom_ns") else 4.0; TD = "out_ctltune10" if DPX == 10 else "out_ctltune"
    TUNE = os.path.exists(f"{L}/{TD}/tune_m1f_{fam}.npz")                        # command tuned on the shown trajectory
    z, m = npz(f"{L}/{TD}/tune_m1f_{fam}.npz" if TUNE else f"{L}/out_ctlk/controlk_m1f_{fam}.npz"); R = m["report"][fam]; sl = R["slots"]
    TK = "tune" if TUNE else "fig"
    def _rb(kk):                                       # field shown: the one whose structure varies most along x
        B0 = z[f"{fam}|fig|base"][..., kk]; fq0 = np.fft.fftfreq(B0.shape[0])
        T0 = np.real(np.fft.ifft(np.fft.fft(B0, axis=0) * np.exp(-2j * np.pi * fq0 * 4.0)[:, None], axis=0))
        return float(np.corrcoef(B0.ravel(), T0.ravel())[0, 1])
    k = sl.index(0)                                    # velocity u_x in every family (as the original control figure)
    ach = float(z[f"{fam}|{TK}|achieved"][k])
    base, steer = z[f"{fam}|{TK}|base"][..., k], z[f"{fam}|{TK}|steer"][..., k]; n = base.shape[0]
    fq = np.fft.fftfreq(n)                                                       # numeric reference: base shifted +4 cells
    ref = np.real(np.fft.ifft(np.fft.fft(base, axis=0) * np.exp(-2j * np.pi * fq * DPX)[:, None], axis=0))
    mu = float(base.mean()); vm = np.percentile(np.abs(np.stack([base, steer]) - mu), 99)
    rr = float(np.corrcoef(steer.ravel(), ref.ravel())[0, 1]); rb = float(np.corrcoef(base.ravel(), ref.ravel())[0, 1])
    for r, (F, lab) in enumerate(((base, "base"), (steer, "commanded"))):
        c0 = 0.0 if r == 2 else mu                                                # smooth, no contours (first-draft style)
        ax = fig.add_subplot(ga[r, j]); ax.imshow(F.T, origin="lower", cmap="RdBu_r", vmin=c0 - vm, vmax=c0 + vm,
                                                 interpolation="bilinear", extent=(0, n, 0, n))
        if r < 2:
            ax.axvline(n / 2 + (DPX if r else 0.0), color=INK, lw=0.7, ls="--")
            if r: ax.axvline(n / 2, color=INK, lw=0.4, ls=":")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
        if r == 0: ax.set_title(f"{LBL[fam]} (${SN[sl[k]]}$)", fontsize=6*FS)
        if r == 1:                                                                # command arrow below the prediction
            AL = DPX                                                              # arrow to scale: commanded shift in cells (user 10-21)
            ax.annotate("", xy=(n / 2 + AL, -n * 0.07), xytext=(n / 2, -n * 0.07), annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=3, shrinkA=0, shrinkB=0))
            ax.set_xlim(0, n); ax.set_ylim(0, n)
            if False:                                                             # row label dropped (user 10-23: overlapped; caption says arrows)
                ax.text(0.0, -n * 0.19, "command", ha="left", va="center", fontsize=5.5*FS, clip_on=False)
            ax.text(n / 2 + AL / 2, -n * 0.19, f"+{DPX:g} px", ha="center", va="center", fontsize=5*FS, clip_on=False)
        if j == 0: ax.set_ylabel(lab, fontsize=5.5*FS)
        if j == 0 and r == 0: tag(ax, "a", -0.18, 1.04)
    J["a"][fam] = dict(field=SN[sl[k]], tuned=TUNE, alpha=R.get("fig_tune", {}).get("alpha"),
                       alpha_calibrated=R.get("fig_tune", {}).get("alpha_calibrated"), achieved=ach, r_steer_vs_ref=rr, r_base_vs_ref=rb, n=int(n))
# (grey footnote under panel a removed, user 10-20; content goes to the caption)

# ---------------- b: one-step linear response (calibration constants -> SI table, kept in the json)
src = "out_ctl1g" if all(os.path.exists(f"{L}/out_ctl1g/control1g_m1f_{f}.npz") for f in FAMS) else "out_ctl1"
ax = fig.add_subplot(g0[1]); tag(ax, "b", -0.12)
X, Y, rows = [], [], []
for fam in FAMS:
    z, m = npz(f"{L}/{src}/control1{'g' if src.endswith('g') else ''}_m1f_{fam}.npz"); R = m["report"][fam]
    rows.append((fam, R["eta"]))
    for c, mk in ((0, "o"), (1, "^")):
        xs, ys, es = [], [], []
        for v in R[f"axis{c}"].values():
            A = np.nanmean(np.array(v["achieved"], float), 1)
            xs.append(v["target"]); ys.append(A.mean()); es.append(A.std())
        o = np.argsort(xs); xs, ys, es = np.array(xs)[o], np.array(ys)[o], np.array(es)[o]
        ax.errorbar(xs, ys, yerr=es, fmt=mk, ms=2.4, mfc=COL[fam] if c == 0 else "white", mec=COL[fam], ecolor=COL[fam],
                    elinewidth=0.5, capsize=1, lw=0, mew=0.6, label=f"{LBL[fam]}" if c == 0 else None)
        X += list(xs); Y += list(ys)
X, Y = np.array(X), np.array(Y); sl_ = float((X * Y).sum() / (X * X).sum())
r2 = 1 - ((Y - X) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()           # against y = x, the calibrated prediction (user 10-21)
lo = 1.1 * np.abs(np.r_[X, Y]).max(); t = np.linspace(-lo, lo, 2)
ax.plot(t, t, color=MUTED, lw=0.8, ls="--", zorder=0)                # LS fit line removed: calibration uses eta_c only
ax.set_xlim(-lo, lo); ax.set_ylim(-lo, lo)
ax.set_xlabel("calibrated latent command $\\eta_c\\,\\alpha$ (cells/step)"); ax.set_ylabel("physical advection (cells/step)")
ax.text(0.03, 0.97, f"$R^2$ = {r2:.3f} (to $y = x$)",
        transform=ax.transAxes, va="top", fontsize=5.2*FS)
ax.legend(frameon=False, fontsize=5.2*FS, loc="upper left", bbox_to_anchor=(0.0, 0.92), handletextpad=0.2, borderaxespad=0.2)
ins = ax.inset_axes([0.64, 0.14, 0.33, 0.28])
for fam in ("incom_ns",):
    z, m = npz(f"{L}/{src}/control1{'g' if src.endswith('g') else ''}_m1f_{fam}.npz"); R = m["report"][fam]
    for c, mk in ((0, "o"), (1, "^")):
        v = R[f"axis{c}"]; xs = [w["target"] for w in v.values()]
        ys = [np.nanmean(np.array(w["achieved"], float)) for w in v.values()]
        ins.plot(xs, ys, mk, ms=1.8, mfc=COL[fam] if c == 0 else "white", mec=COL[fam], mew=0.5)
    e = 1.15 * max(np.abs(xs)); ins.plot([-e, e], [-e, e], color=MUTED, lw=0.5, ls="--"); ins.set_xlim(-e, e); ins.set_ylim(-e, e)
ins.tick_params(labelsize=4.3*FS, length=1.5); ins.set_title("Incomp. NS", fontsize=4.8*FS, pad=1)
J["b"] = dict(source=src, pooled_slope=sl_, R2_identity=float(r2))
J["SI_eta"] = {fam: dict(eta_x=eta[0], eta_y=eta[1]) for fam, eta in rows}

# ---------------- d: shear command, e: vortex command vs flow map and solver
gd = (GridSpecFromSubplotSpec(1, 2, G[1], width_ratios=[1.45, 1.0], wspace=0.16) if FIELDS else
      GridSpecFromSubplotSpec(1, 2, G[1], width_ratios=[1.0, 1.0], wspace=0.18))
g3 = GridSpecFromSubplotSpec(1, 2, G[2], width_ratios=[1.0, 1.0] if FIELDS else [1.25, 1.0], wspace=0.22)
gdd = gd if FIELDS else \
      GridSpecFromSubplotSpec(2, 3, gd[0], height_ratios=[1.0, 0.8], hspace=0.45, wspace=0.08)
if FIELDS:            # base | command | prediction | ground truth (NS solver with the command); shear (top), vortex (bottom)
    from scipy.ndimage import map_coordinates as _mc
    gm = GridSpecFromSubplotSpec(7, 4, gd[0], hspace=0.25, wspace=0.10, height_ratios=[1, 1, 1, 1, 0.22, 1, 1])  # row 4: spacer before diffusion   # 2 sub-rows per map; shear, vortex, diffusion
    w_ = lambda F: (np.roll(F[..., 1], -1, 0) - np.roll(F[..., 1], 1, 0)) / 2 - (np.roll(F[..., 0], -1, 1) - np.roll(F[..., 0], 1, 1)) / 2
    def _line(U, n, K=10, sub=8):                                                # material line x = n/2 carried by U
        p0, p1 = np.full(200, n / 2.0), np.linspace(0, n - 1, 200)
        for _ in range(K * sub):
            u0 = _mc(U[..., 0], [p0, p1], order=1, mode="grid-wrap"); u1 = _mc(U[..., 1], [p0, p1], order=1, mode="grid-wrap")
            p0, p1 = p0 + u0 / sub, p1 + u1 / sub
        return p0, p1
    J["c_fields"] = {}
    rows_ = []
    for kind, AMP, path, lab in (("shear", "3", "out_ctl3f/control3f", "shear"), ("curl", "3", os.environ.get("VORTEX_DIR", "out_ctl3v20") + "/control3v", "vortex")):
        z, m = npz(f"{L}/{path}_m1f_NS-Sines.npz"); rows_.append((kind, AMP, z, m, lab))
    Wb = w_(rows_[0][2][f"NS-Sines|shear3|fields"][0]); vmf = np.percentile(np.abs(Wb), 99); n = Wb.shape[0]
    axb = fig.add_subplot(gm[2:4, 0]); axb.imshow(Wb.T, origin="lower", cmap="RdBu_r", vmin=-vmf, vmax=vmf, interpolation="bilinear",
                                                extent=(0, n, 0, n), aspect="equal")
    if not CDIFF: axb.axvline(n / 2, color=INK, lw=0.6, ls="--", alpha=0.45)
    axb.set_xticks([]); axb.set_yticks([]); axb.set_title("base", fontsize=5.5*FS)
    for s_ in axb.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
    if CDIFF: tag(axb, "c", -0.08)       # SI full-field version (CDIFF=0): no panel letter
    for r, (kind, AMP, z, m, lab) in enumerate(rows_):
        Fl = z[f"NS-Sines|{kind}{AMP}|fields"]; U_ = z[f"NS-Sines|{kind}{AMP}|U"]
        Wp, Wg = w_(Fl[1]), w_(Fl[3]); lx, ly = _line(U_, n)
        if CDIFF:                                       # changes: model commanded - base; solver with - without the command
            Wp, Wg = w_(Fl[1]) - w_(Fl[0]), w_(Fl[3]) - w_(Fl[4])
            vmd = float(np.percentile(np.abs(np.concatenate([Wp.ravel(), Wg.ravel()])), 99.5))
        C3 = m["report"]["NS-Sines"]["conds"][f"{kind}{AMP}"]
        rd = [x["10"]["r_dyn"] for x in C3 if x][0]; rk = [x["10"]["r_kin"] for x in C3 if x][0]   # the shown trajectory (user)
        J["c_fields"][kind] = dict(amp_std=float(AMP), r_solver10=float(rd), r_flowmap10=float(rk))
        ax = fig.add_subplot(gm[2 * r:2 * r + 2, 1]); mag = np.hypot(U_[..., 0], U_[..., 1]); um = float(mag.max())
        _cmd = ax
        ax.imshow((U_[..., 0] if kind == "shear" else mag).T, origin="lower", cmap="PuOr_r" if kind == "shear" else "Purples",
                  vmin=-um if kind == "shear" else 0, vmax=um, extent=(0, n, 0, n))
        st_ = 12; gx, gy = np.meshgrid(np.arange(st_ // 2, n, st_), np.arange(st_ // 2, n, st_), indexing="ij")
        ax.quiver(gx, gy, U_[gx, gy, 0], U_[gx, gy, 1], color="w" if kind == "shear" else INK, scale=um * 12, width=0.012)
        if kind == "shear":
            prof = U_[0, :, 0] / um; yy = np.arange(n)
            ax.plot(n / 2 + 0.35 * n * prof, yy, color="w", lw=1.6); ax.plot(n / 2 + 0.35 * n * prof, yy, color=INK, lw=0.8)
        ax.set_ylabel(lab, fontsize=5.5*FS, labelpad=1); J["c_fields"][kind]["peak_cells_10steps"] = um * 10
        for c_, (F, t_) in enumerate(((Wp, "commanded $-$ base"), (Wg, "NS solver change")) if CDIFF else
                                     ((Wp, "commanded"), (Wg, "ground truth (NS solver)"))):
            ax2 = fig.add_subplot(gm[2 * r:2 * r + 2, 2 + c_])
            if c_ == 0: FLOW.append((axb if r == 1 else None, _cmd, ax2))
            _v = vmd if CDIFF else vmf
            ax2.imshow(F.T, origin="lower", cmap="RdBu_r", vmin=-_v, vmax=_v,
                                                            interpolation="bilinear", extent=(0, n, 0, n))
            if not CDIFF: ax2.plot(lx, ly, color=INK, lw=0.6, ls="--", alpha=0.45)   # guide: line carried by the command alone
            #   (measured lines dropped, user 10-20: tracking error ~2.6 cells for the vortex, see material_line_track2.py)
            ax2.set_xlim(0, n); ax2.set_ylim(0, n)
            if r == 0: ax2.set_title(t_, fontsize=5.5*FS)
            if False: ax2.text(0.97, 0.03, f"r {rd:.2f}", transform=ax2.transAxes, ha="right", fontsize=4.8*FS,
                                 bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.4))
        for a_ in (ax, ax2, fig.axes[-2]):
            a_.set_xticks([]); a_.set_yticks([])
            for s_ in a_.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
        if r == 0: ax.set_title("command $\\delta u$", fontsize=5.5*FS)
    # ---- third row: NS-Sines prescribed DIFFUSION command (REVISION-R1 10-20): c*Lap(h) added to the Laplacian feature of the
    # Laplacian head group (diffusion, coefficient, compressible), c = 0.217 calibrated on traj 0-3 for nu x2; traj 4, step 5,
    # unfiltered. Ground truth: solver with nu x2.01 (cal4 store; same traj, same observed frame, same model base).
    zk = np.load(f"{L}/out_knob/cal4_m1f_NS-Sines.npz", allow_pickle=True); SI = 1   # traj 4 (= rows 1-2), step 5
    zc = np.load(f"{L}/out_knob/lsteerG_t4c.npz", allow_pickle=True); KN = "diff1.217"
    Bk, Ak, Sk = zk["base"][SI], zc[f"{KN}|model"][SI], zk["diff2.01|solver"][SI]
    KC = None
    rk_ = float(np.corrcoef(Ak.ravel(), Sk.ravel())[0, 1]); ak_ = float((Ak * Sk).sum() / (Sk * Sk).sum())   # projection gain
    _m4 = json.loads(str(zc["__meta__"]))["report"]["traj"][0]["conds"][KN]["calib"]["mult"]
    # diffusion command illustration (user 10-20): a pulse spreading from the centre — iso-line of the diffused pulse for
    # alpha = 0.8 / 1 / 1.2 (one line style, darker = larger alpha) over the alpha = 1 pulse; the cross-section profiles hang
    # below the square in an extension of the panel frame (green palette, distinct from the purple velocity commands)
    ax = fig.add_subplot(gm[5:7, 1]); ax.set_box_aspect(1); _cmd = ax; n_ = 128
    X_, Y_ = np.meshgrid(np.arange(n_), np.arange(n_), indexing="ij"); r2_ = (X_ - n_ / 2) ** 2 + (Y_ - n_ / 2) ** 2
    _als = (0.8, 1.0, 1.2); _sws = [11.0 * np.sqrt(al_ ** 3.4) for al_ in _als]; _cm = matplotlib.cm.Greens
    _pulse = lambda sw: np.exp(-r2_ / (2 * sw ** 2)) * (_sws[0] / sw) ** 2
    ax.imshow(_pulse(_sws[1]).T, origin="lower", cmap="Greens", vmin=0, vmax=0.6, extent=(0, n_, 0, n_))
    for k_, sw_ in enumerate(_sws):
        ax.contour(X_, Y_, np.exp(-r2_ / (2 * sw_ ** 2)), levels=[0.25], colors=[_cm(0.45 + 0.27 * k_)], linewidths=0.8)
    ax.set_xlim(0, n_); ax.set_ylim(0, n_)
    ax.set_title("command $\\alpha\\,\\nabla^2 h$", fontsize=5.5*FS, pad=2); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
    fig.canvas.draw(); _pp = ax.get_position(); _hb = 0.42 * _pp.height
    axp = fig.add_axes([_pp.x0, _pp.y0 - _hb, _pp.width, _hb]); xx_ = np.arange(n_)
    for k_, sw_ in enumerate(_sws):                                         # equal-area profiles hanging from the square
        axp.plot(xx_, -np.exp(-(xx_ - n_ / 2) ** 2 / (2 * sw_ ** 2)) * (_sws[0] / sw_), color=_cm(0.45 + 0.27 * k_), lw=0.9)
    axp.set_facecolor("none"); axp.set_xlim(0, n_); axp.set_ylim(-1.12, 0.06); axp.set_xticks([]); axp.set_yticks([])   # box bottom edge stays visible
    for s_ in axp.spines.values(): s_.set_visible(False)                  # profiles hang below the box, unframed
    axp.text(0.80, 0.30, "$\\alpha$ ↑", transform=axp.transAxes, ha="left", fontsize=4.6*FS)
    ax.set_ylabel("diffusivity", fontsize=5.5*FS, labelpad=1)
    # full fields at step 10 like the shear / vortex rows (user 10-20): commanded = model base + commanded response;
    # ground truth = NS solver with nu x2.01 from the observed frame (out_knob/solver_full_t4.npz, same units)
    _sf = np.load(f"{L}/out_knob/solver_full_t4.npz")
    Wc = zk["base"][2] + zc[f"{KN}|model"][2]; Wg = _sf["nu2.01"][2]
    if CDIFF:                                           # changes: commanded response; solver nu x2.01 - solver nu x1
        Wc, Wg = zc[f"{KN}|model"][2], _sf["nu2.01"][2] - _sf["nu1"][2]
        vmd = float(np.percentile(np.abs(np.concatenate([Wc.ravel(), Wg.ravel()])), 99.5))
        J["c_fields"]["diffusion"] = dict(norm_ratio_step10=float(np.linalg.norm(Wc) / np.linalg.norm(Wg)))
    _uvm = np.load(f"{L}/out_knob/group_t4_uv.npz")[f"{KN}|model_uv"][2]; _uvs = (_sf["uv_nu2.01"] - _sf["uv_nu1"])[2]
    _rdf = float(np.corrcoef(_uvm.ravel(), _uvs.ravel())[0, 1])      # velocity response r at step 10, as for shear / vortex
    for c_, F in enumerate((Wc, Wg)):
        ax = fig.add_subplot(gm[5:7, 2 + c_])
        if c_ == 0: FLOW.append((None, _cmd, ax))
        _v = vmd if CDIFF else vmf
        ax.imshow(F.T, origin="lower", cmap="RdBu_r", vmin=-_v, vmax=_v, interpolation="bilinear")
        if False: ax.text(0.97, 0.03, f"r {_rdf:.2f}", transform=ax.transAxes, ha="right", fontsize=4.8*FS,
                            bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.4))
    for a_ in fig.axes[-2:]:
        a_.set_xticks([]); a_.set_yticks([])
        for s_ in a_.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
    J["c_fields"].setdefault("diffusion", {}); J["c_fields"]["diffusion"].update(traj=4, command="c*Lap(h) -> diffusion,coefficient,compressible", c=0.217, step=5, lowpass_k=KC, r=rk_, proj_gain=ak_, achieved=_m4)
    # (grey note under the base panel removed, user 10-20; content goes to the caption)
    gR = GridSpecFromSubplotSpec(2, 1, gd[1], height_ratios=[1.35, 1.0], hspace=0.55)
    # diffusion: (left) latent Laplacian factor s -> fitted viscosity factor (solver refit, start at no change, traj 0-4);
    # (right) step-1 response spectrum, commanded and solver, latent resolution limit k = 16 (user 10-20)
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    gS2 = GridSpecFromSubplotSpec(1, 2, gR[1], wspace=0.45)
    axc = fig.add_subplot(gS2[0]); J["c_calcurve"] = {}
    def _mults(path_, fac_):
        m_ = json.loads(str(np.load(path_, allow_pickle=True)["__meta__"]))["report"]["traj"]
        return np.array([t_["conds"][f"diff{fac_:g}"]["calib"]["mult"]["diff"] for t_ in m_ if f"diff{fac_:g}" in t_["conds"]])
    SER = [("Laplacian heads", VER, "s", [(f_, f"{L}/out_knob/x0one_group_curve.npz") for f_ in (0.75, 0.9, 1.1, 1.3)]
            + [(1.217, f"{L}/out_knob/x0one_group.npz")])]
    for lab_, col_, mk_, pts in SER:
        xs_, ms_, lo_, hi_ = [1.0], [1.0], [1.0], [1.0]
        for f_, path_ in pts:
            v_ = _mults(path_, f_); xs_.append(f_); ms_.append(float(np.exp(np.log(v_).mean()))); lo_.append(v_.min()); hi_.append(v_.max())
        o_ = np.argsort(xs_); xs_, ms_, lo_, hi_ = (np.array(a_)[o_] for a_ in (xs_, ms_, lo_, hi_))
        axc.errorbar(xs_, ms_, yerr=[ms_ - lo_, hi_ - ms_], fmt=mk_ + "-", color=col_, ms=2.2, lw=0.8, elinewidth=0.5, capsize=1, label=lab_)
        J["c_calcurve"][lab_] = dict(s=xs_.tolist(), nu=np.round(ms_, 3).tolist())
    _X, _Y = [], []                                                       # power-law fit through (1, 1) over all trajectories
    for f_, path_ in SER[0][3]:
        v_ = _mults(path_, f_); _X += [np.log(f_)] * len(v_); _Y += list(np.log(v_))
    _X, _Y = np.array(_X), np.array(_Y); _p = float((_X * _Y).sum() / (_X * _X).sum()); _r2 = 1 - ((_Y - _p * _X) ** 2).sum() / ((_Y - _Y.mean()) ** 2).sum()
    _aa = np.linspace(0.72, 1.33, 50); axc.plot(_aa, _aa ** _p, "--", color=MUTED, lw=0.6, zorder=0)
    axc.text(0.05, 0.93, f"$\\nu \\propto \\alpha^{{{_p:.1f}}}$ ($R^2$ = {_r2:.2f})", transform=axc.transAxes, fontsize=4.5*FS, va="top")
    J["c_calcurve"]["power"] = dict(p=_p, R2=float(_r2))
    _star = float(_mults(f"{L}/out_knob/x0one_group.npz", 1.217)[-1])     # held-out traj 5, unbiased-start fit (2.10; Supplementary Note 2.4)
    axc.plot([1.217], [_star], "*", color=VER, ms=5, mec=INK, mew=0.3, zorder=5); J["c_calcurve"]["star_heldout"] = _star
    axc.set_yscale("log"); axc.set_xscale("log"); axc.set_xticks([0.8, 1.0, 1.2]); axc.xaxis.set_major_formatter(FixedFormatter(["0.8", "1", "1.2"])); axc.xaxis.set_minor_formatter(NullFormatter()); axc.set_yticks([0.5, 1, 2]); axc.yaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2"]))
    axc.yaxis.set_minor_formatter(NullFormatter()); axc.set_ylim(0.3, 3.2); axc.set_xlim(0.7, 1.35)
    axc.set_xlabel("calibrated latent command $\\alpha$ (diffusivity)", fontsize=5*FS); axc.set_ylabel("fitted viscosity factor", fontsize=5*FS)
    axc.tick_params(labelsize=4.5*FS)
    tag(axc, "e", -0.32)
    axs = fig.add_subplot(gS2[1])
    def _rad(F):
        n_ = F.shape[0]; Fh = np.abs(np.fft.fft2(F)) ** 2; kk = np.fft.fftfreq(n_) * n_
        Kk = np.hypot(*np.meshgrid(kk, kk, indexing="ij")).astype(int)
        return np.array([Fh[Kk == i].mean() for i in range(1, n_ // 2)])
    J["c_spec"] = {}
    eb_ = _rad(zk["base"][0]); ks_ = np.arange(1, len(eb_) + 1)
    for src_, col_, lab_ in ((zc[f"{KN}|model"], VER, "commanded"), (zk["diff2.01|solver"], INK, "solver ($\\nu$ ×2)")):
        ra_ = np.sqrt(_rad(src_[0]) / eb_); axs.loglog(ks_[1:48], ra_[1:48], "-", color=col_, lw=0.8, label=lab_)
        m_ = (ks_ >= 2) & (ks_ <= 8); J["c_spec"][lab_ + "_slope_k2_8"] = float(np.polyfit(np.log(ks_[m_]), np.log(ra_[m_]), 1)[0])
    axs.set_xlabel("wavenumber $k$", fontsize=5*FS); axs.set_ylabel("$|\\delta\\hat\\omega| / |\\hat\\omega|$, step 1", fontsize=5*FS)
    axs.set_title("response spectrum", fontsize=5.2*FS); axs.legend(frameon=False, fontsize=4.3*FS, loc="upper left", handlelength=1.2)
    axs.tick_params(labelsize=4.5*FS)
    axs.xaxis.set_major_locator(FixedLocator([2, 4, 8, 16, 32])); axs.xaxis.set_major_formatter(FixedFormatter(["2", "4", "8", "16", "32"]))
    axs.xaxis.set_minor_formatter(NullFormatter())
    # NS-Sines, NS-solver reference only (user 10-20): pattern r of the commanded velocity change against the solver with the
    # command as an external advecting velocity, at steps 1, 5, 10; shear (out_ctl3f) and vortex (out_ctl3v20[,b], R = 20)
    J["c_bars"] = {}
    SRC = [("out_ctl3f/control3f", "shear"), ("out_ctl3v20/control3v", "curl"), ("out_ctl3v20b/control3v", "curl")]
    CURVES = {}
    for path_, kind_ in SRC:
        if not os.path.exists(f"{L}/{path_}_m1f_NS-Sines.npz"): continue
        _, m_ = npz(f"{L}/{path_}_m1f_NS-Sines.npz"); R_ = m_["report"]["NS-Sines"]; eb = float(np.mean(R_["eta"]))
        for key_, v_ in R_["conds"].items():
            if not key_.startswith(kind_): continue
            amp_ = float(key_[len(kind_):]); pk = amp_ * eb * 10
            lab_ = f"{pk:.0f} cells" if kind_ == "shear" else f"{np.degrees(pk / 20.0):.0f}°"
            CURVES[(kind_, amp_)] = (lab_, np.array([x["10"]["r_dyn"] for x in v_ if x]))
    gB = GridSpecFromSubplotSpec(1, 2, gR[0], wspace=0.25)
    for b_, (kind_, ttl_, cols_) in enumerate((("shear", "shear", [SKY, BLU, "#08306b"]), ("curl", "vortex", [ORA, VER, "#7f2704"]))):
        ax = fig.add_subplot(gB[b_]); items = sorted((k_[1], v_) for k_, v_ in CURVES.items() if k_[0] == kind_)
        for x_, (amp_, (lab_, rr_)) in enumerate(items):
            ax.bar(x_, rr_.mean(), 0.6, color=cols_[min(x_, 2)]); ax.plot(np.full(len(rr_), x_), rr_, ".", ms=1.8, color=INK)
            J["c_bars"][f"{ttl_} {lab_}"] = float(rr_.mean())
        ax.set_xticks(range(len(items))); ax.set_xticklabels([v_[0] for _, v_ in items], fontsize=4.8*FS)
        ax.set_ylim(0, 1.0); ax.tick_params(labelsize=4.8*FS); ax.set_title(ttl_, fontsize=5.5*FS)
        if b_ == 0: ax.set_ylabel("pattern r vs NS solver (step 10)", fontsize=5*FS); tag(ax, "d", -0.32)
        else: ax.set_yticklabels([])
else:
  z, m = npz(f"{L}/out_ctl2/control2_m1f_NS-Sines.npz"); B_, P_, Ref = z["NS-Sines|shear4|u"]
  n = B_.shape[0]; vm = np.percentile(np.abs(P_ - B_), 99)
  for j, (F, t_) in enumerate(((P_ - B_, "model response"), (Ref - B_, "flow map (reference)"))):
    ax = fig.add_subplot(gdd[0, j]); ax.imshow(F.T, origin="lower", cmap="PuOr_r", vmin=-vm, vmax=vm, extent=(0, n, 0, n))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t_, fontsize=5.5*FS)
    for s_ in ax.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
    if j == 0: tag(ax, "c", -0.08)
  ax = fig.add_subplot(gdd[0, 2]); yy = np.arange(n); ax.plot(4 * np.sin(2 * np.pi * yy / n), yy, color=INK, lw=0.8)
  ax.axvline(0, color=MUTED, lw=0.4); ax.set_ylim(0, n); ax.set_yticks([]); ax.set_xlabel("command $\\delta_x(y)$ (cells)", fontsize=5*FS)
  ax.set_title("shear, 4 cells peak", fontsize=5.5*FS)
if not FIELDS:
    ax = fig.add_subplot(gdd[1, :])
    rr, J["d"] = [], {}
    for fam in FAMS:
        _, m = npz(f"{L}/out_ctl2/control2_m1f_{fam}.npz"); res = [x for x in m["report"][fam]["shear4"] if x]
        v = [np.mean([x[f]["r_diff"] for f in x]) for x in res]; rr.append((np.mean(v), np.std(v))); J["d"][fam] = float(np.mean(v))
    ax.bar(range(5), [x[0] for x in rr], yerr=[x[1] for x in rr], color=[COL[f] for f in FAMS], width=0.6,
           error_kw=dict(lw=0.5, capsize=1.5))
    ax.set_xticks(range(5)); ax.set_xticklabels([LBL[f] for f in FAMS], fontsize=5.5*FS); ax.set_ylim(0, 1)
    ax.set_ylabel("pattern r vs flow map", fontsize=5.5*FS)

gee = GridSpecFromSubplotSpec(1, 4, g3[0], width_ratios=[1.3, 1, 1, 1], wspace=0.12)
if FIELDS: gee = None                      # vortex flow-map/solver panel folded into c
if not FIELDS:
    z, m = npz(f"{L}/out_v3/v3_m1f_NS-Sines.npz"); C = m["report"]["NS-Sines"]["conds"]
    ax = fig.add_subplot(gee[0]); tag(ax, "e", -0.12); J["f"] = {}
    for key, lab, col in (("r_kin", "flow map", MUTED), ("r_dyn", "NS solver", BLU)):
        ks = [1, 5, 10]; M = []
        for amp in ("0.25", "0.5", "1"):
            res = [x for x in C[f"curl{amp}"] if x]; M.append([np.mean([x[str(k)][key] for x in res]) for k in ks])
        M = np.array(M); ax.errorbar(ks, M.mean(0), yerr=M.std(0), color=col, marker="o", ms=2.5, lw=0.8, capsize=1, elinewidth=0.5,
                                     label=lab); J["f"][key] = M.mean(0).tolist()
    ax.set_xticks([1, 5, 10]); ax.set_xlabel("step"); ax.set_ylabel("pattern r with model response"); ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False, fontsize=5.5*FS, loc="lower left"); ax.set_title("vortex command (NS-Sines)", fontsize=5.5*FS)
    S, Kn, D = z["NS-Sines|curl1|steer"][1], z["NS-Sines|curl1|kin"][1], z["NS-Sines|curl1|dyn"][1]
    if FIELDS:                                                                    # fields = unsteered forecast + response
        Bw = z["NS-Sines|base_w"][1]; S, Kn, D = Bw + S, Bw + Kn, Bw + D
    vm = np.percentile(np.abs(S), 99)
    for j, (F, t_) in enumerate(((S, "model"), (Kn, "flow map"), (D, "NS solver"))):
        ax = fig.add_subplot(gee[j + 1]); ax.imshow(F.T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{t_}, step 10", fontsize=5.5*FS)
        for s_ in ax.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
        if FIELDS:
            n_ = F.shape[0]; R_ = float(m["args"]["rad"]); th = np.linspace(0, 2 * np.pi, 200)
            for rr_ in (R_, 1.6 * R_): ax.plot((n_ - 1) / 2 + rr_ * np.cos(th), (n_ - 1) / 2 + rr_ * np.sin(th), color=INK, lw=0.4, ls=":")
    ax.text(0.5, -0.12, ("vorticity field (unsteered forecast + response); dotted: command core R, 1.6R" if FIELDS else
                         "vorticity response (commanded minus unsteered)"), transform=ax.transAxes, ha="right", fontsize=5*FS, color=MUTED)

# ---------------- e: divergent command, compressible families: density response vs continuity and vs warp (step 1)
if FIELDS and os.path.exists(f"{L}/out_ctl3dv/control3dv_m1f_{os.environ.get('D_FAM', 'CE-RP')}.npz"):
    # base density | command (convergent / divergent, rad 20) | predicted d rho | continuity d rho (mass-conserving GT) |
    # image-warp d rho; below: pattern r and the total-mass budget sum(d rho)/sum|d rho| on the periodic domain
    DF = os.environ.get("D_FAM", "CE-RP"); DA = os.environ.get("D_AMP", "1")
    gx_ = GridSpecFromSubplotSpec(1, 2, G[2], width_ratios=[1.45, 1.0], wspace=0.16)   # same split as c (columns align)      # full-width band
    # layout as panel c (user 10-20): one base, source / sink rows (command -> commanded vs mass-conserving GT); image-warp
    # maps removed (the warp comparison stays in the bars); source and sink commands in different colours
    gm2 = GridSpecFromSubplotSpec(4, 4, gx_[0], hspace=0.12, wspace=0.10)   # panels the size of c (user)
    z, m = npz(f"{L}/out_ctl3dv/control3dv_m1f_{DF}.npz")
    rb0 = z[f"{DF}|div-{DA}|rhofields"][1]; n = rb0.shape[0]
    axb2 = fig.add_subplot(gm2[1:3, 0]); mu_ = rb0.mean(); s_d = np.percentile(np.abs(rb0 - mu_), 99)
    axb2.imshow(rb0.T, origin="lower", cmap="RdBu_r", vmin=mu_ - s_d, vmax=mu_ + s_d, interpolation="bilinear")
    axb2.set_title("base $\\rho$", fontsize=5.5*FS); tag(axb2, "f", -0.12); axb2.set_xticks([]); axb2.set_yticks([])
    for s_ in axb2.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
    for r, (kind, lab, cmap_) in enumerate((("div-", "expansion", "Reds"), ("div", "compression", "Blues"))):
        rin, rb, rp, rg, rw = z[f"{DF}|{kind}{DA}|rhofields"]; U_ = z[f"{DF}|{kind}{DA}|U"]
        dP, dG = rp - rb, rg - rb; vm = np.percentile(np.abs(dG), 99.5)
        net = float(dP.sum() / np.abs(dP).sum())
        ax = fig.add_subplot(gm2[2 * r:2 * r + 2, 1]); _cmd = ax; mag = np.hypot(U_[..., 0], U_[..., 1]); um = float(mag.max())
        ax.imshow(mag.T, origin="lower", cmap=cmap_, vmin=0, vmax=um * 1.15)
        st_ = 12; gx, gy = np.meshgrid(np.arange(st_ // 2, n, st_), np.arange(st_ // 2, n, st_), indexing="ij")
        ax.quiver(gx, gy, U_[gx, gy, 0], U_[gx, gy, 1], color=INK, scale=um * 12, width=0.012)
        ax.add_patch(matplotlib.patches.Circle(((n - 1) / 2, (n - 1) / 2), 20, fill=False, ec="0.45", lw=0.7, ls="--"))  # control volume, R = 20 cells
        if r == 0: ax.set_title("command $\\delta u$", fontsize=5.5*FS)
        ax.set_ylabel(lab, fontsize=5.5*FS, labelpad=1)
        for j, (F, t_) in enumerate(((dP, "commanded $-$ base"), (dG, "continuity reference\n$-\\nabla\\cdot(\\rho\\,\\delta u)$"))):
            ax = fig.add_subplot(gm2[2 * r:2 * r + 2, 2 + j])
            if j == 0: FLOW.append((axb2 if r == 0 else None, _cmd, ax))
            ax.imshow(F.T, origin="lower", cmap="PuOr_r", vmin=-vm, vmax=vm, interpolation="bilinear")
            if r == 0: ax.set_title(t_, fontsize=5.2*FS)
            if j == 0:                                                        # as panel c: r with the GT, bottom right (mean, 4 test traj)
                _rc = float(np.corrcoef(dP.ravel(), dG.ravel())[0, 1])       # r of the shown fields (user); mean +- sd in the bars
                if False: ax.text(0.97, 0.03, f"r {_rc:.2f}", transform=ax.transAxes, ha="right", fontsize=4.8*FS, bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.4))
        for a_ in fig.axes[-3:]:
            a_.set_xticks([]); a_.set_yticks([])
            for s_ in a_.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
        if r == 1:                                                           # what the two right columns show (user)
            fig.canvas.draw(); _p1, _p2 = fig.axes[-2].get_position(), fig.axes[-1].get_position()
            fig.text(0.5 * (_p1.x0 + _p2.x1), _p1.y0 - 0.006, "density change $\\delta\\rho$", ha="center", va="top", fontsize=5.2*FS)
    gbb = GridSpecFromSubplotSpec(2, 1, gx_[1], hspace=0.75); CF = ["CE-RP", "CE-Gauss", "com_ns"]; J["d"] = {}
    axr, axm = fig.add_subplot(gbb[0]), fig.add_subplot(gbb[1]); tag(axr, "g", -0.17); tag(axm, "h", -0.17)   # (user 10-20)
    for i, fam in enumerate(CF):
        if not os.path.exists(f"{L}/out_ctl3dv/control3dv_m1f_{fam}.npz"): continue
        z2, m2 = npz(f"{L}/out_ctl3dv/control3dv_m1f_{fam}.npz"); C = m2["report"][fam]["conds"]
        for key, off, col, lab in (("r_cont", 0.0, "#8c96a8", "vs mass-conserving GT"),):   # softer than black (user)        # image-warp bars removed (user 10-20)
            v = [x["1"][key] for kk in C for x in C[kk] if x and key in x["1"]]
            axr.bar(i + off, np.mean(v), 0.45, yerr=np.std(v), color=col, error_kw=dict(lw=0.5, capsize=1.5), label=lab if i == 0 else None)
            J["d"].setdefault(fam, {})[key] = [float(np.mean(v)), float(np.std(v))]
        # mass response of the centred control volume (R = 20 cells, the command radius): model mass change / the change the
        # continuity equation requires (0 = no density response, 1 = continuity); 4 test trajectories x 3 amplitudes (user 10-20)
        _cm = json.loads(str(np.load(f"{L}/out_ctl3dm/control3dv_m1f_{fam}.npz", allow_pickle=True)["__meta__"]))["report"][fam]["conds"]
        for off_, (sg_, col_, lab_) in zip((-0.18, 0.18), (("div-", "#d6604d", "expansion"), ("div", "#4393c3", "compression"))):
            v_ = [x["1"]["cv_model"][x["1"]["cv_R"].index(20)] / x["1"]["cv_req"][x["1"]["cv_R"].index(20)]
                  for a_ in ("1", "2", "4") for x in _cm.get(f"{sg_}{a_}", []) if x and "cv_req" in x["1"]]
            # REVISION-R1 11-02 (user): individual points (4 trajectories x 3 amplitudes) replaced by s.d. error bars, as in g
            axm.bar(i + off_, np.mean(v_), 0.34, yerr=np.std(v_), color=col_, alpha=0.85, error_kw=dict(lw=0.5, capsize=1.5),
                    label=lab_ if i == 0 else None)
            J["d"][fam][f"cv20_{lab_}"] = [float(np.mean(v_)), float(np.std(v_)), len(v_)]
    for a_, yl in ((axr, "pattern r of $\\delta\\rho$"), (axm, "|net mass change|\n$|\\Sigma\\delta\\rho| / \\Sigma|\\delta\\rho|$")):
        a_.set_xticks(range(3)); a_.set_xticklabels([LBL[f] for f in CF], fontsize=5*FS); a_.set_ylabel(yl, fontsize=5*FS)
    axr.set_ylim(0, 1.1); axr.set_ylabel("pattern r of $\\delta\\rho$", fontsize=5*FS)
    axm.set_ylim(-0.1, 1.6); axm.axhline(1.0, color="#7fbf7f", lw=0.8, ls="--", zorder=0)   # 1 = continuity (light green)
    axm.axhline(0, color="#e8a0a0", lw=0.8, ls="--", zorder=0)                                # 0 = no density response (light red)
    axm.set_ylabel("mass-change ratio", fontsize=5*FS)
    axm.legend(frameon=False, fontsize=4.4*FS, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=1, columnspacing=0.6)
else:
    gee2 = GridSpecFromSubplotSpec(2, 3, g3[0] if FIELDS else gd[1], height_ratios=[1.0, 0.8], hspace=0.45, wspace=0.08)
    z, m = npz(f"{L}/out_ctl3d/control3_m1f_CE-Gauss.npz"); dr, ct, kw = z["CE-Gauss|div|rho"]
    vm = np.percentile(np.abs(ct), 99)
    for j, (F, t_) in enumerate(((dr, "model $\\delta\\rho$"), (ct, "continuity $-\\nabla\\cdot(\\rho\\,\\delta u)$"),
                                 (kw, "image warp $-\\delta u\\cdot\\nabla\\rho$"))):
        ax = fig.add_subplot(gee2[0, j]); ax.imshow(F.T, origin="lower", cmap="PuOr_r", vmin=-vm, vmax=vm)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t_, fontsize=5.5*FS)
        for s_ in ax.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
        if j == 0:
            tag(ax, "d", -0.08); ax.set_ylabel("CE-Gauss, source", fontsize=5.5*FS)
    ax = fig.add_subplot(gee2[1, :]); CF = ["CE-RP", "CE-Gauss", "com_ns"]; J["e"] = {}
    for i, fam in enumerate(CF):
        _, m = npz(f"{L}/out_ctl3d/control3_m1f_{fam}.npz"); C = m["report"][fam]["conds"]
        for key, off, col, lab in (("r_cont", -0.18, INK, "vs continuity"), ("r_kinrho", 0.18, "#bdbdbd", "vs image warp")):
            v = [x["1"][key] for kk in ("div0.5", "div1", "div2", "div-0.5", "div-1", "div-2") for x in C[kk] if x and key in x["1"]]
            ax.bar(i + off, np.mean(v), 0.34, yerr=np.std(v), color=col, error_kw=dict(lw=0.5, capsize=1.5), label=lab if i == 0 else None)
            J["e"].setdefault(fam, {})[key] = [float(np.mean(v)), float(np.std(v))]
    ax.axhline(0, color=INK, lw=0.4); ax.set_xticks(range(3)); ax.set_xticklabels([LBL[f] for f in CF], fontsize=5.5*FS)
    ax.set_ylim(-0.1, 1.3); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("pattern r of $\\delta\\rho$ (step 1)", fontsize=5.5*FS)
    ax.legend(frameon=False, fontsize=5*FS, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("divergent command (source / sink, 3 amplitudes)", fontsize=5.5*FS)

SHOW_FHN = os.environ.get("SHOW_FHN", "0") == "1"   # FHN joint-command band (former i) dropped from Fig. 2 (user 10-20)
if SHOW_FHN:
    # ---------------- g: diffusion knob, calibrated as an equivalent-equation change (solver refit)
    K_ = json.load(open(os.path.join(os.path.dirname(__file__), "out", "fig_knob_calib.json")))
    if FIELDS:            # maps: base | knob | prediction (response) | ground truth (solver, diffusion term x2); right: multipliers
        gE = GridSpecFromSubplotSpec(1, 2, G[3], width_ratios=[1.75, 1.0], wspace=0.16)
        gme = GridSpecFromSubplotSpec(2, 4, gE[0], hspace=0.14, wspace=0.06)
        # CALIBRATED joint commands (user 10-20): targets on the equation multipliers, group factors from the 2x2 calibration
        # matrix C (type-1 regime, fitted on calibration trajectories), held-out trajectory 4; GT = solver with the targets
        zj = np.load(f"{L}/out_knob/joint1_m1f_diff_react.npz", allow_pickle=True); Bf = zj["base"]
        TJ = json.loads(str(zj["__meta__"]))["report"]["test"]
        J["e_maps"] = {}
        for r, (tg, lab, col) in enumerate((("0.7_1", "diffusion control", BLU), ("1_1.5", "reaction control", VER))):
            td, trt = [float(v) for v in tg.split("_")]
            A, Cg = zj[f"t{tg}|model"], zj[f"t{tg}|solver"]
            rr = float(np.corrcoef(A.ravel(), Cg.ravel())[0, 1]); ar = float(np.linalg.norm(A) / np.linalg.norm(Cg))
            rec = [t for t in TJ if t["target"] == [td, trt]]
            fD, fR = rec[0]["f"]; mD = np.mean([t["m"]["diff"] for t in rec]); mR = np.mean([t["m"]["react"] for t in rec])
            J["e_maps"][tg] = dict(r=rr, amp_ratio=ar, f=[fD, fR], achieved=[float(mD), float(mR)])
            mu_ = Bf.mean(); sb = np.percentile(np.abs(Bf - mu_), 99)
            ax = fig.add_subplot(gme[r, 0]); ax.imshow(Bf.T, origin="lower", cmap="RdBu_r", vmin=mu_ - sb, vmax=mu_ + sb, interpolation="bilinear")
            ax.set_ylabel(lab, fontsize=5.2*FS, labelpad=1)
            if r == 0: ax.set_title("base (FHN $u$, frame 10)", fontsize=5.2*FS); tag(ax, "i", -0.12)
            ax = fig.add_subplot(gme[r, 1]); ax.set_facecolor("#f4f1f8"); ax.set_box_aspect(1)
            ax.text(0.5, 0.80, f"target: $D$ ×{td:g}, $R$ ×{trt:g}", ha="center", va="center", fontsize=5.0*FS, color=col,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.50, f"Laplacian heads ×{fD:.2f}\nnon-operator heads ×{fR:.2f}", ha="center", va="center",
                    fontsize=4.6*FS, transform=ax.transAxes)
            ax.text(0.5, 0.18, f"achieved:\n$D$ ×{mD:.2f}, $R$ ×{mR:.2f}", ha="center", va="center", fontsize=4.4*FS,
                    color=MUTED, transform=ax.transAxes)
            if r == 0: ax.set_title("calibrated command", fontsize=5.2*FS)
            vm_ = np.percentile(np.abs(np.stack([A, Cg])), 99)                      # one scale per row (amplitudes comparable)
            for c_, (F, t_) in enumerate(((A, "commanded (response)"), (Cg, "ground truth: solver\nwith the target multipliers"))):
                ax = fig.add_subplot(gme[r, 2 + c_]); ax.imshow(F.T, origin="lower", cmap="RdBu_r", vmin=-vm_, vmax=vm_, interpolation="bilinear")
                if r == 0: ax.set_title(t_, fontsize=5.0*FS)
                if c_ == 0: ax.text(0.97, 0.03, f"r {rr:.2f}, amp {ar:.2f}", transform=ax.transAxes, ha="right", fontsize=4.6*FS,
                                    bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.4))
            for a_ in fig.axes[-4:]:
                a_.set_xticks([]); a_.set_yticks([])
                for s_ in a_.spines.values(): s_.set_visible(True); s_.set_linewidth(0.4)
        gg = GridSpecFromSubplotSpec(1, 3, gE[1], wspace=0.55)
        _gp = f"{L}/out_knob/calgrp_m1f_diff_react.npz"                      # FHN group knobs (user 10-20)
        if os.path.exists(_gp):
            _T = json.loads(str(np.load(_gp, allow_pickle=True)["__meta__"]))["report"]["traj"]
            GRP = {}
            for kn in ("diff", "react"):
                for f_ in ("0.5", "2"):
                    ms = [t["conds"][f"{kn}{f_}"]["calib"]["mult"] for t in _T if "calib" in t["conds"].get(f"{kn}{f_}", {})]
                    for term in ("diff", "react"):
                        GRP.setdefault(f"{kn}->{term}", {})[f_] = [float(np.mean([x[term] for x in ms])), float(np.std([x[term] for x in ms]))]
                for term in ("diff", "react"): GRP[f"{kn}->{term}"]["1"] = [1.0, 0.0]
            K_["diff_react_grp"] = GRP; J["e_groups"] = GRP
    else:
        gg = GridSpecFromSubplotSpec(1, 2, g3[1], wspace=0.35)
    _E = [("NS-Sines", "diff->diff", "diff->adv", "viscosity $\\nu$", "advection", "NS-Sines:\ndiffusion head", "diffusion-head factor")]
    if FIELDS and "diff_react_grp" in K_:
        _E += [("diff_react_grp", "diff->diff", "diff->react", "diffusion $D$", "reaction", "FHN: Laplacian-\nhead group", "group factor"),
               ("diff_react_grp", "react->react", "react->diff", "reaction", "diffusion $D$", "FHN: non-operator\nhead group", "group factor")]
    else:
        _E += [("diff_react", "diff->diff", "diff->react", "diffusion $D$", "reaction", "FitzHugh–Nagumo", "diffusion-head factor")]
    for j, (fam, own, other, lo_, lt_, ttl_, xl_) in enumerate(_E):
        ax = fig.add_subplot(gg[j]); fs = ["0.5", "1", "2"]; x = np.array([0.5, 1, 2])
        for key, col, lab in ((own, VER if own.startswith("react") else BLU, lo_), (other, MUTED, lt_)):
            mu = [K_[fam][key][f][0] for f in fs]; sd = [K_[fam][key][f][1] for f in fs]
            ax.errorbar(x, mu, yerr=sd, marker="o", ms=2.5, color=col, lw=0.8, capsize=1, elinewidth=0.5, label=lab)
        ax.plot([0.4, 2.2], [0.4, 2.2], color=MUTED, lw=0.5, ls="--", zorder=0)
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels(["½", "1", "2"])
        ax.set_yticks([0.5, 1, 2]); ax.set_yticklabels(["½", "1", "2"]); ax.set_xlim(0.4, 2.4); ax.set_ylim(0.4, 2.4)
        ax.set_xlabel(xl_, fontsize=5.2*FS)
        if j == 0:
            ax.set_ylabel("fitted equation multiplier", fontsize=5.5*FS)
            if not FIELDS: tag(ax, "f", -0.25)
        ax.set_title(ttl_, fontsize=4.8*FS)
        ax.legend(frameon=False, fontsize=4.6*FS, loc="lower right", handlelength=1.2)
    J["g"] = {f: {k: K_[f][k] for k in K_[f] if "->" in k} for f in ("NS-Sines", "diff_react")}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
# base + command -> commanded: "+" between base and command, arrow from command to commanded (user 10-20)
from matplotlib.patches import FancyArrowPatch
fig.canvas.draw()
for b_, c_, o_ in FLOW:
    pc, po = c_.get_position(), o_.get_position(); yc = 0.5 * (pc.y0 + pc.y1)
    fig.add_artist(FancyArrowPatch((pc.x1 + 0.002, yc), (po.x0 - 0.002, yc), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=6, lw=0.9, color=INK))
    if b_ is not None:
        pb = b_.get_position()
        fig.text(pb.x1 + 0.35 * (pc.x0 - pb.x1), 0.5 * (pb.y0 + pb.y1), "+", ha="center", va="center", fontsize=9*FS, color=INK)
print("SIZES c-base", [round(v, 4) for v in axb.get_position().bounds], "f-base", [round(v, 4) for v in axb2.get_position().bounds])
fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=300)
json.dump(J, open(OUT + ".json", "w"), indent=1)
print("saved", OUT, json.dumps(J["b"]))
