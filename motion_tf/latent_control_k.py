"""K-STEP-CALIBRATED latent transport control (revision R1, main Fig. 2 redraw, 2026-10-19).
Differs from latent_control.py only in the calibration horizon: the transport efficiency eta_c (cells per unit alpha per
step, alpha = command in std units of u(h)_c) is measured on calibration trajectories A as the displacement after the SAME
K steps used for scoring (persistent command alpha = +/-1 along latent component c, displacement along grid axis c), for
BOTH axes c = 0, 1. latent_control.py measured eta on step 1 only, which misses the compounding of the AR rollout
(10-17 diagnosis: CE per-step increments grow ~2x over steps 2-4) and left per-family slopes 0.65-1.88 at K = 10.
Commands on held-out trajectories B: alpha = D* / (K eta_c) along axis c, D* in TARGETS, c in (0, 1). One constant per
family and axis. Also stored for the figure: input field (last observed frame), unsteered and steered forecast at K for
D* = FIGD along axis 0 on B0.
--- original docstring ---
Latent transport CONTROL (revision R1-3): can the forecast's transport be commanded through the learned latent velocity?
For each family: trajectories A (calibration) and B (held-out test).
  calibration 1 (no intervention): gain of u(h) on the true velocity (latent grid, normalized units) and dt/dx from the data
                                   (one-frame transport of a transported field by the true velocity) -> "u calibration".
  calibration 2: transport efficiency eta = displacement per unit alpha per step, measured on A with alpha = +/-1.
Command: a target displacement D* (cells, along grid axis 0, after K steps). The latent velocity component that transports
along axis 0 is changed at every step by
  (i)  u-calibration only : the latent velocity change that corresponds to the physical velocity D*/(K dt/dx)
  (ii) u + eta calibration: alpha = D* / (K eta)
Achieved displacement of every physical field of the forecast at step K, measured on B by an exact sub-cell search
(periodic Fourier translation minimizing || steered - T_s(base) ||). Numeric reference: a uniform velocity added to a
periodic flow translates the solution exactly (Galilean), so T_D*(ground truth) and T_D*(base forecast) are the targets.
  python latent_control.py --ckpt C --out OUT.npz [--fams ...] [--K 10]
"""
import argparse, itertools, json, os, zlib
import numpy as np
import tensorflow as tf
tf.random.set_seed(0); tf.config.experimental.enable_op_determinism()  # effects are ~1e-4; GPU non-determinism is ~6e-5
try:                                                  # Keras 2.15 eager leak: Softmax keeps its logits on the output
    from keras.src.layers.activation import softmax as _smod   # tensor (~70 MB/step on 64x64 latents) -> OOM
    _sm_call = _smod.Softmax.call
    def _sm_nolog(self, inputs, mask=None):
        out = _sm_call(self, inputs, mask)
        try: del out._keras_logits
        except AttributeError: pass
        return out
    _smod.Softmax.call = _sm_nolog
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--fams", default="incom_ns,NS-Sines,com_ns,CE-RP,CE-Gauss")
ap.add_argument("--nA", type=int, default=4); ap.add_argument("--nB", type=int, default=4); ap.add_argument("--K", type=int, default=10)
ap.add_argument("--targets", default="-4,-2,-1,1,2,4")
ap.add_argument("--figD", type=float, default=4.0)
ap.add_argument("--calib_alphas", default="-1,1", help="calibration command grid on A; eta = least-squares slope of displacement on alpha*K")
ap.add_argument("--fig_tune", type=int, default=0, help="1: illustration only - on B0, tune the axis-0 command so the u_x field moves exactly figD cells after K steps (secant), store the fields, skip the scored grid")
ap.add_argument("--alphas", default="", help="commands in std units of u(h) (overrides --targets): D* = alpha K eta_c per family/axis")
ap.add_argument("--per_step", type=int, default=0, help="1: record the achieved displacement at every step (ueta mode)")
ap.add_argument("--hist_base", type=int, default=0, help="1: during steered rollouts the observed-history window is filled with the UNSTEERED forecast (anchor and command unchanged)")
a = ap.parse_args()

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_2D, _grid
if os.environ.get("LOCAL_UNITS"):                     # local run on units exported by export_units.py
    import pickle, types
    _LU = pickle.load(open(os.environ["LOCAL_UNITS"], "rb"))
    def _restore(u):
        u = dict(u); f = u.pop("fields"); full = np.zeros(f.shape[:-1] + (int(u.pop("_S")),), f.dtype)
        full[..., u.pop("_act")] = f; u["fields"] = full
        from data.symbolic import OPERATOR_VOCAB          # exported without MOTION_LEGACY_VOCAB: drop the appended token
        if "op_multihot" in u and len(u["op_multihot"]) == len(OPERATOR_VOCAB) + 1:
            assert float(np.asarray(u["op_multihot"])[-1]) == 0.0; u["op_multihot"] = np.asarray(u["op_multihot"])[:-1]
        return u
    loader = types.SimpleNamespace(units_for=lambda fam, split: (_restore(u) for u in _LU[fam]))

ARCH = tuple(int(os.environ.get(k, v)) for k, v in (("ARCH_D", 384), ("ARCH_DEPTH", 8), ("ARCH_DW", 16)))  # MOTION: 896, 12, 48
model = V3Model(S=NUM_SLOTS, d=ARCH[0], depth=ARCH[1], d_w=ARCH[2], n_p=int(os.environ.get("N_P", "16")), d_cond=128, dims2=(64, 64), dims3=(32, 32, 32),
                t_in=10, transport=True, n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS, gate_cond=1, dict_decode=0, fam_head=1)
# REVISION-R1 10-27: MOTION_GLOBAL_NORM models (MOTION-IVP) need the per-family statistics before examples are built
import v3.data_ar as _DA
if _DA.GLOBAL_NORM_ON:
    for _f in a.fams.split(","):
        if os.environ.get("CTL_GN_JSON"):          # REVISION-R1 10-29: the run's global_norm.json (local runs: no train units)
            _g = json.load(open(os.environ["CTL_GN_JSON"]))[_f]
            _DA.GLOBAL_NORM[_f] = (np.asarray(_g["mean"], np.float32), np.asarray(_g["std"], np.float32))
        else:
            _DA.GLOBAL_NORM[_f] = _DA.compute_global_norm(loader.units_for(_f, "train"))[:2]
        print(f"[ctl] global norm {_f}", flush=True)
model.warm_build(); V = model.trainable_variables
load_ckpt(V, a.ckpt, d_w=ARCH[2], names=[v.name for v in V])
# REVISION-R1 (10-21): command under a runtime knockout / warp_off (same switches as eval_stats.py), to locate the
# pathway through which the latent-velocity command acts. MOTION_CTL_KO="head1,head2"; MOTION_CTL_WARPOFF=1.
_KO = [h for h in os.environ.get("MOTION_CTL_KO", "").split(",") if h]
if _KO or os.environ.get("MOTION_CTL_WARPOFF", "0") == "1":
    _ALLH = list(model.banks.names) + list(getattr(model.banks, "_gen", [])) + ["bernoulli", "kinematic", "vortex",
            "reaction_nl", "spectral", "coefficient", "compressible"]
    model.banks.ko_index = {h: i for i, h in enumerate(_ALLH)}
    _m = np.ones(len(_ALLH), np.float32)
    for _h in _KO:
        assert _h in model.banks.ko_index, f"unknown head {_h}"; _m[model.banks.ko_index[_h]] = 0.0
    model.banks.ko_mask = tf.Variable(_m, trainable=False, name="ko_mask")
    model.warp_scale = tf.Variable(0.0 if os.environ.get("MOTION_CTL_WARPOFF", "0") == "1" else 1.0, trainable=False)
    print(f"[ctl] knockout {_KO} warp_off {os.environ.get('MOTION_CTL_WARPOFF', '0')}", flush=True)
CAP, DELTA, STEER = {}, {"d": None}, {"v": None}
_bank = model.banks.call
def _bank_w(h, *args, **kw):
    CAP["h"] = h; return _bank(h, *args, **kw)
model.banks.call = _bank_w
_core = model.core
model.core = lambda h, e, roles: _core(h, e, roles) + DELTA["d"] if DELTA["d"] is not None else _core(h, e, roles)
_vel = model.banks.vel.call
def _vel_w(x, *args, **kw):
    u = _vel(x, *args, **kw); CAP["u"] = u
    return u + tf.cast(STEER["v"], u.dtype) if STEER["v"] is not None else u
model.banks.vel.call = _vel_w
# REVISION-R1 (10-21, user): command the physical-space warp (Lagrangian carrier) as well. The warp displacement
# disp(h) (coordinate units; backward warp, content moves by +disp) gets alpha * MOTION_CTL_WARP_CPA cells along the
# commanded axis whenever the latent command is on; MOTION_CTL_LAT=0 keeps the latent velocity unsteered (warp-only).
_WCPA = float(os.environ.get("MOTION_CTL_WARP_CPA", "0")); _LAT = os.environ.get("MOTION_CTL_LAT", "1") == "1"
WSTEER = {"a": None}                                       # (alpha_axis0, alpha_axis1) of the current command
if not _LAT:
    def _vel_w2(x, *args, **kw):
        u = _vel(x, *args, **kw); CAP["u"] = u; return u
    model.banks.vel.call = _vel_w2
if _WCPA and getattr(model, "disp", None) is not None:
    _disp = model.disp.call
    def _disp_w(x, *args, **kw):
        d = _disp(x, *args, **kw)
        if STEER["v"] is not None and WSTEER["a"] is not None:
            n_ = float(GRID_N["n"])
            add = tf.constant([WSTEER["a"][0] * _WCPA / n_, WSTEER["a"][1] * _WCPA / n_], d.dtype)
            d = d + add
        return d
    model.disp.call = _disp_w
GRID_N = {"n": 128}
SAVE = (1, 5, 10)


HB = {"p": None}                                      # unsteered forecast (K, Nn, S) for --hist_base


# REVISION-R1 10-29: models trained in single-shot / receding-horizon mode (MOTION-IVP: MOTION_FORCE_MODE=ss +
# MOTION_RH_FAMS) are rolled out with their own hop-1 operator, model.step_ss(..., kf, leads=[1]), as at inference
# (score_ivp_l1.py HOP=1). model.step is the per-frame AR operator, which such a model never trained: its readout at
# t = 1 equals the single-shot readout at the END of the kf-frame horizon, so the earlier MOTION-IVP control runs
# (ivp-control-gn3w, 10-27) advanced ~kf frames per step. The commands act on the shared trunk / bank (DELTA on the
# core output, STEER on banks.vel), identical in both operators. CTL_SS defaults to on when MOTION_FORCE_MODE=ss.
CTL_SS = os.environ.get("CTL_SS", "1" if os.environ.get("MOTION_FORCE_MODE", "") == "ss" else "0") == "1"
CTL_KF = int(os.environ.get("CTL_KF", "20"))


def _rollout1(b, K, delta=None, steer=None, persistent=False):
    T = lambda k: tf.constant(b[k])
    cn, xw, fm, gn = T("coords_node"), T("x_win"), T("fmask"), T("geom_node")
    qall = tf.concat([cn, T("coords_q")], 1); gall = tf.concat([gn, T("geom_q")], 1)
    u_prev = tf.concat([xw[:, -1], T("ic_q")], 1)
    K_, roles = int(b["K"]), [int(r) for r in b["roles"]]; gd = tuple(int(x) for x in b["dims"])
    win, fmask = xw, fm; pg, pg2 = _grid(xw[:, -1], gd, model.S), _grid(xw[:, -2], gd, model.S)
    Nn = int(cn.shape[1]); preds = []; first = {}
    for k in range(K):
        on = k == 0 or persistent
        DELTA["d"] = tf.constant(delta, tf.float32) if delta is not None and on else None
        STEER["v"] = tf.constant(steer, tf.float32) if steer is not None and on else None
        if CTL_SS:                                    # REVISION-R1 10-29: single-shot model -> its hop-1 operator
            pred = model.step_ss(cn, win, fmask, gn, K_, roles, T("cond"), T("op_multihot"), qall, gall, u_prev,
                                 kf=CTL_KF, fam_id=T("fam_id"), sdf_box=T("sdf_box"),
                                 box_dims=tuple(int(v) for v in b["box_dims"]), prev_grid=pg, grid_dims=gd,
                                 leads=[1])[:, 0]
        else:
            pred = model.step(cn, win, fmask, gn, K_, roles, T("cond"), T("op_multihot"), qall, gall, u_prev, steady=False,
                              prev_grid=pg, grid_dims=gd, fam_id=T("fam_id"), sdf_box=T("sdf_box"),
                              box_dims=tuple(int(v) for v in b["box_dims"]), fbmask=T("fbmask"), surf=False, prev2_grid=pg2)
        DELTA["d"] = STEER["v"] = None
        pred = tf.cast(pred[0] if isinstance(pred, (tuple, list)) else pred, tf.float32)
        if k == 0: first = {"h": CAP["h"].numpy(), "u": CAP["u"].numpy()}
        preds.append(pred[:, :Nn].numpy()); u_prev = pred; pn = pred[:, :Nn]
        hist = tf.constant(HB["p"][k][None], tf.float32) if (HB["p"] is not None and steer is not None) else pn
        win = tf.concat([win[:, 1:], hist[:, None]], 1); fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        pg2, pg = (_grid(hist, gd, model.S) if HB["p"] is not None and steer is not None else pg), _grid(pn, gd, model.S)
    return np.stack(preds, 1), first



CHUNK = int(os.environ.get("ROLL_CHUNK", "1"))         # trajectories per forward pass (memory: 12 GB local GPU)
def _sub(b, sl):
    n = int(np.asarray(b["x_win"]).shape[0])
    return {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n and k not in ("dims", "box_dims") else v)
            for k, v in b.items()}


def rollout(b, K, *args, **kw):
    n = int(np.asarray(b["x_win"]).shape[0]); outs = []
    for i in range(0, n, CHUNK):
        sl = slice(i, min(i + CHUNK, n)); kw2 = dict(kw); a2 = list(args)
        for key in ("delta", "edit"):
            v = kw2.get(key)
            if isinstance(v, np.ndarray) and v.ndim == 4 and v.shape[0] == n: kw2[key] = v[sl]
        if a2 and isinstance(a2[0], np.ndarray) and a2[0].ndim == 4 and a2[0].shape[0] == n: a2[0] = a2[0][sl]
        try:
            outs.append(_rollout1(_sub(b, sl), K, *a2, **kw2))
        except (tf.errors.ResourceExhaustedError, tf.errors.InvalidArgumentError) as e:   # diverged rollout
            print(f"  [rollout diverged / failed: {type(e).__name__}] -> NaN", flush=True)
            bs = _sub(b, sl); m_ = int(np.asarray(bs["x_win"]).shape[0])
            outs.append((np.full((m_, K, int(np.asarray(bs["coords_node"]).shape[1]), int(np.asarray(bs["x_win"]).shape[-1])),
                                 np.nan, np.float32), None))
    first = None if any(o[1] is None for o in outs) else outs[0][1]
    if isinstance(first, dict):
        first = {k: np.concatenate([o[1][k] for o in outs]) for k in first}
    elif first is not None:
        first = np.concatenate([o[1] for o in outs])
    return np.concatenate([o[0] for o in outs]), first

def make(fam, units, idx, boost=None):
    bd = (int(os.environ.get("MOTION_BOX_UNIFORM", "0")) or int(BOX_TABLE_2D.get(fam, 64)),) * 2; exs = []   # REVISION-R1: NS-S uniform box
    for i, s in zip(idx, units):
        s = dict(s)
        if boost is not None:                                   # Galilean reference: uniform input velocity boost
            f = np.array(s["fields"], np.float32, copy=True); fl = f.reshape(f.shape[0], -1, f.shape[-1])
            fl[..., boost[0]] += boost[1] * float(fl[:10, :, boost[0]].std()); s["fields"] = fl.reshape(f.shape)
        exs.append(make_example_ar(s, 10, a.K, 4, 2048, 8192, np.random.default_rng(zlib.crc32(f"{fam}:{i}:0".encode())),
                                   box_dims=bd))
    return stack_batch(exs)


g = lambda q, ax: (np.roll(q, -1, ax) - np.roll(q, 1, ax)) / 2
SNAME = {0: "u", 1: "v", 3: "rho", 4: "p", 6: "c"}


def tshift(q, s, ax):                             # periodic sub-cell translation: content moves by +s cells along ax
    k = np.fft.fftfreq(q.shape[ax]); sh = [1] * q.ndim; sh[ax] = -1
    return np.real(np.fft.ifft(np.fft.fft(q, axis=ax) * np.exp(-2j * np.pi * k * s).reshape(sh), axis=ax))


def best_shift(steer, base, ax, lo=-15.0, hi=15.0):
    F = np.fft.fft(base, axis=ax); k = np.fft.fftfreq(base.shape[ax]); sh = [1] * base.ndim; sh[ax] = -1
    err = lambda s: float(((steer - np.real(np.fft.ifft(F * np.exp(-2j * np.pi * k * s).reshape(sh), axis=ax))) ** 2).sum())
    grid = np.arange(lo, hi + 1e-9, 0.1); e = [err(s) for s in grid]; s0 = grid[int(np.argmin(e))]
    a_, b_ = s0 - 0.1, s0 + 0.1
    for _ in range(30):                          # golden-section refinement
        c_, d_ = b_ - 0.618 * (b_ - a_), a_ + 0.618 * (b_ - a_)
        if err(c_) < err(d_): b_ = d_
        else: a_ = c_
    return 0.5 * (a_ + b_)


def one(b, i):
    return _sub(b, slice(i, i + 1))


store, report = {}, {}
TG = [float(t) for t in a.targets.split(",")]
for fam in a.fams.split(","):
    units = list(itertools.islice(loader.units_for(fam, "test"), a.nA + a.nB))
    if len(units) < a.nA + a.nB: print("skip (units)", fam); continue
    iA, iB = list(range(a.nA)), list(range(a.nA, a.nA + a.nB))
    bA = make(fam, units[:a.nA], iA); bB = make(fam, units[a.nA:], iB)
    if not bool(bA["dense2d"]): print("skip", fam); continue
    dims = tuple(int(x) for x in bA["dims"]); sl = [s for s in FAMILIES[fam].channels if s in SNAME]
    baseA, fA = rollout(bA, a.K); baseB, fB = rollout(bB, a.K)
    PA = baseA.reshape(a.nA, a.K, *dims, -1); PB = baseB.reshape(a.nB, a.K, *dims, -1)
    # ---- calibration 1: u(h) gain vs true velocity (A), and dt/dx from the raw data (A)
    uh = fA["u"]; Hb = uh.shape[1]; fpool = dims[0] // Hb
    xA = np.asarray(bA["x_win"])[:, -1].reshape(a.nA, *dims, -1); xbA = xA.reshape(a.nA, Hb, fpool, Hb, fpool, -1).mean((2, 4))
    cc = [[abs(np.corrcoef(uh[..., c].ravel(), xbA[..., s].ravel())[0, 1]) for s in (0, 1)] for c in (0, 1)]
    ax_slot = [0, 1] if cc[0][0] + cc[1][1] >= cc[0][1] + cc[1][0] else [1, 0]      # slot carrying velocity along axis 0 / 1
    gain = float(np.polyfit(xbA[..., ax_slot[0]].ravel(), uh[..., 0].ravel(), 1)[0])  # u(h)_0 per normalized velocity
    tq = [s for s in sl if s not in (0, 1)]
    # dt/dx: fit dq = -kappa * (transport term) on the data; candidates: passive scalar (c), continuity (rho, conservative
    # form; the non-conservative -u.grad(rho) fits compressible data with R^2 ~ 0.05 and a random sign), vorticity.
    # The candidate with the best median R^2 is used (CORRECTION 2026-10-17: CE used -u.grad(rho) -> sign error).
    cand = {}
    for u in units[:a.nA]:
        f = np.asarray(u["fields"], np.float64).reshape(-1, *dims, np.asarray(u["fields"]).shape[-1])
        V0, V1 = f[:11, ..., ax_slot[0]], f[:11, ..., ax_slot[1]]; m_ = lambda x: 0.5 * (x[1:] + x[:-1])
        terms = {"vorticity": (g(V1, 1) - g(V0, 2), None)}
        if 6 in tq: terms["scalar"] = (f[:11, ..., 6], None)
        if 3 in tq: terms["continuity"] = (f[:11, ..., 3], "cons")
        for nm, (q, form) in terms.items():
            dq, qm = q[1:] - q[:-1], m_(q)
            adv = g(qm * m_(V0), 1) + g(qm * m_(V1), 2) if form == "cons" else m_(V0) * g(qm, 1) + m_(V1) * g(qm, 2)
            k_ = -(dq * adv).sum() / (adv * adv).sum(); r2 = 1 - ((dq + k_ * adv) ** 2).sum() / (dq ** 2).sum()
            cand.setdefault(nm, []).append((k_, r2))
    best = max(cand, key=lambda k_: np.median([c[1] for c in cand[k_]]))
    kappa = float(np.median([c[0] for c in cand[best]])); kappa_r2 = float(np.median([c[1] for c in cand[best]]))
    scaleB = np.asarray(bB["scale"]).reshape(a.nB, -1)[:, ax_slot[0]]
    # ---- calibration 2 (K-step): eta_c on A, persistent alpha = +/-1 along latent component c, displacement after K
    sd = [float(uh[..., c].std()) for c in (0, 1)]; eta = []
    if os.environ.get("MOTION_CTL_SD"):                  # REVISION-R1 10-22: absolute command unit (collapsed u(h))
        sd = [float(os.environ["MOTION_CTL_SD"])] * 2; print(f"[{fam}] command unit overridden: sd = {sd[0]}", flush=True)
    for c in (0, 1):
        xs, ys = [], []
        for al in [float(x) for x in a.calib_alphas.split(",")]:
            st = np.zeros((1, 1, 1, 3), np.float32); st[..., c] = al * sd[c]; WSTEER["a"] = (al if c == 0 else 0.0, al if c == 1 else 0.0)
            for i in range(a.nA):
                p, _ = rollout(one(bA, i), a.K, steer=st, persistent=True)
                if p is None or not np.isfinite(p).all(): continue
                P1 = p.reshape(1, a.K, *dims, -1)[0, -1]
                ys.append(float(np.mean([best_shift(P1[..., s], PA[i, -1, ..., s], c) for s in sl]))); xs.append(a.K * al)
        xs, ys = np.asarray(xs), np.asarray(ys)
        eta.append(float((xs * ys).sum() / (xs * xs).sum()))       # LS slope through 0 (= mean ratio for +/-1)
    R = dict(ax_slot=ax_slot, gain=gain, kappa=kappa, kappa_term=best, kappa_r2=kappa_r2, eta=eta, eta_K=a.K, sd_uh=sd, slots=sl)
    print(f"[{fam}] K-step calib (K={a.K}): eta axis0 {eta[0]:.4f} axis1 {eta[1]:.4f} cells/alpha/step", flush=True)
    if a.fig_tune:                       # illustration (main Fig. 2a): command tuned on THIS trajectory to give figD cells
        ku = sl.index(0); bi = one(bB, 0)
        def shift_of(al):
            st = np.zeros((1, 1, 1, 3), np.float32); st[..., 0] = al * sd[0]
            p, _ = rollout(bi, a.K, steer=st, persistent=True); P1 = p.reshape(1, a.K, *dims, -1)[0, -1]
            return best_shift(P1[..., 0], PB[0, -1, ..., 0], 0), P1
        a0 = a.figD / (a.K * eta[0]); s0, P0 = shift_of(a0); a1 = a0 * a.figD / max(s0, 1e-3); s1, P1 = shift_of(a1); hist = [(a0, s0), (a1, s1)]
        for _ in range(4):
            if abs(s1 - a.figD) < 0.02: break
            a2 = a1 + (a.figD - s1) * (a1 - a0) / ((s1 - s0) if abs(s1 - s0) > 1e-6 else 1e-6)
            a0, s0 = a1, s1; a1 = a2; s1, P1 = shift_of(a1); hist.append((a1, s1))
        store[f"{fam}|tune|input"] = np.asarray(bB["x_win"])[0, -1].reshape(*dims, -1)[..., sl].astype(np.float32)
        store[f"{fam}|tune|base"] = PB[0, -1][..., sl].astype(np.float32)
        store[f"{fam}|tune|steer"] = P1[..., sl].astype(np.float32)
        store[f"{fam}|tune|achieved"] = np.array([best_shift(P1[..., s], PB[0, -1, ..., s], 0) for s in sl], np.float32)
        R["fig_tune"] = dict(alpha=a1, alpha_calibrated=a.figD / (a.K * eta[0]), u_shift=s1, hist=hist)
        print(f"[{fam}] fig_tune: alpha {a1:.3f} (calibrated {a.figD / (a.K * eta[0]):.3f}) -> u_x shift {s1:.3f}", flush=True)
        report[fam] = R; continue
    # ---- control on B, both axes
    for c in (0, 1):
        res = {}
        TGc = ([float(x) * a.K * eta[c] for x in a.alphas.split(",")] if a.alphas else TG)
        for D in TGc:
            ach = np.zeros((a.nB, len(sl))); lat = np.zeros((a.nB, len(sl))); al = D / (a.K * eta[c])
            st = np.zeros((1, 1, 1, 3), np.float32); st[..., c] = al * sd[c]; WSTEER["a"] = (al if c == 0 else 0.0, al if c == 1 else 0.0)
            for i in range(a.nB):
                p, _ = rollout(one(bB, i), a.K, steer=st, persistent=True)
                if p is None or not np.isfinite(p).all(): ach[i] = np.nan; continue
                P1 = p.reshape(1, a.K, *dims, -1)[0, -1]
                for j, s in enumerate(sl):
                    ach[i, j] = best_shift(P1[..., s], PB[i, -1, ..., s], c)
                    lat[i, j] = best_shift(P1[..., s], PB[i, -1, ..., s], 1 - c, -5, 5)
                if c == 0 and (abs(D - a.figD) < 1e-6 or (a.alphas and D == max(TGc))) and i == 0:
                    store[f"{fam}|fig|input"] = np.asarray(bB["x_win"])[0, -1].reshape(*dims, -1)[..., sl].astype(np.float32)
                    store[f"{fam}|fig|base"] = PB[0, -1][..., sl].astype(np.float32)
                    store[f"{fam}|fig|steer"] = P1[..., sl].astype(np.float32)
                    store[f"{fam}|fig|achieved"] = ach[0].astype(np.float32)
            res[f"{D:.6g}"] = dict(achieved=ach.tolist(), lateral=lat.tolist(), alpha=al, target=D)
            print(f"[{fam}] axis{c} D*={D:+.0f}: alpha {al:+.2f} | achieved per field " +
                  ", ".join(f"{SNAME[s]} {np.nanmean(ach[:, j]):+.2f}" for j, s in enumerate(sl)) +
                  f" | lateral {np.nanmean(np.abs(lat)):.2f}", flush=True)
        R[f"axis{c}"] = res
    report[fam] = R
    np.savez_compressed(a.out, **store, __meta__=json.dumps({"ckpt": a.ckpt, "args": vars(a), "report": report, "partial": True}))
np.savez_compressed(a.out, **store, __meta__=json.dumps({"ckpt": a.ckpt, "args": vars(a), "report": report}))
print("CONTROLK_OK", flush=True)
print("CONTROL_OK", a.out, flush=True)
