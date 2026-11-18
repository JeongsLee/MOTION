"""MECHANISM-LEVEL response to spatially varying latent advection commands (revision R1, main Fig. 2d/e, 2026-10-19).
A uniform latent-velocity command is a Galilean translation (kinematic and dynamic references coincide). For spatially
varying commands they separate; the test is whether the model's response follows the PDE or a passive image warp.
  (d) SHEAR (solenoidal): dU_0 = A sin(2 pi x_1 / n), dU_1 = 0.  Families NS-Sines, incom_ns.
      references: kinematic flow map of the forecast by dU (RK4 characteristics, per component); dynamic NS solver
      with dU as an external advecting velocity of vorticity (NS-Sines; NSext as latent_vortex3). Mechanism observable:
      DIVERGENCE of the velocity response, div_ratio = ||div d|| / ||d|| (kinematic warp of a div-free field by a shear
      creates divergence; the incompressible solver, through the stream function / pressure, does not).
  (e) DIVERGENT command (irrotational source / sink, Gaussian potential). Incompressible families (NS-Sines,
      incom_ns): kinematic + dynamic (NS-Sines) references and div_ratio. Compressible families (CE-RP, CE-Gauss,
      com_ns): CONTINUITY reference for the density response at step 1, d rho = -div(rho dU) = -dU.grad rho
      - rho div dU (cells, one step); the kinematic warp has only the first term. Least-squares split of the model's
      d rho onto [-dU.grad rho, -rho div dU] -> coefficients (a_adv, a_comp); continuity predicts (1, 1), a warp (1, 0).
Command amplitude in std units of the family's u(h) (amp * mean eta cells/step; eta = one-step per-axis calibration on
trajectories A, as latent_vortex3); scored on held-out trajectories B at steps 1, 5, 10.
  python latent_control3.py --ckpt C --out OUT.npz [--fams ...]
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
ap.add_argument("--fams", default="NS-Sines,incom_ns,CE-RP,CE-Gauss,com_ns")
ap.add_argument("--nA", type=int, default=4); ap.add_argument("--nB", type=int, default=4); ap.add_argument("--K", type=int, default=10)
ap.add_argument("--amps", default="0.5,1,2", help="command peak in std units of u(h)")
ap.add_argument("--kinds", default="", help="comma list to restrict (shear,div,div-)")
ap.add_argument("--rad", type=float, default=12.0, help="vortex core radius (data cells)")
ap.add_argument("--ctrl_amp", type=float, default=1.0, help="amplitude at which the control commands are run")
ap.add_argument("--amp_unit", default="cells", choices=["cells", "std"], help="std: amplitude in units of the family's own latent-velocity spread (one rule for every family): peak command = amp * std(u(h)) -> amp * eta cells per step")
ap.add_argument("--horizon_all", type=int, default=0, help="1: full-field metrics at every step (commanded / base vs the solver with and without the command) and fields at steps 10, 20, ..., K")
ap.add_argument("--nu", type=float, default=1.5e-4); ap.add_argument("--dt", type=float, default=0.05)
a = ap.parse_args()

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_2D, _grid
import sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from ns_spectral import NS2D
if os.environ.get("LOCAL_UNITS"):                     # local run on units exported by export_units.py
    import pickle, types
    _LU = pickle.load(open(os.environ["LOCAL_UNITS"], "rb"))
    def _restore(u):
        u = dict(u); f = u.pop("fields"); full = np.zeros(f.shape[:-1] + (int(u.pop("_S")),), f.dtype)
        full[..., u.pop("_act")] = f; u["fields"] = full
        from data.symbolic import OPERATOR_VOCAB          # exported without MOTION_LEGACY_VOCAB: drop the appended token
        if "op_multihot" in u and len(u["op_multihot"]) == len(OPERATOR_VOCAB) + 1:
            assert float(np.asarray(u["op_multihot"])[-1]) == 0.0; u["op_multihot"] = np.asarray(u["op_multihot"])[:-1]
        elif "op_multihot" in u and len(u["op_multihot"]) == len(OPERATOR_VOCAB) - 1:   # exported WITH legacy vocab, used by a
            u["op_multihot"] = np.concatenate([np.asarray(u["op_multihot"]), [0.0]])          # 14-token (ablation) model: append the token
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
SAVE = (1, 5, 10)


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
        win = tf.concat([win[:, 1:], pn[:, None]], 1); fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        pg2, pg = pg, _grid(pn, gd, model.S)
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
        f_ = np.asarray(s["fields"])
        if f_.shape[0] < 11 + a.K:                              # long-horizon runs (--horizon_all): the targets beyond the data
            s["fields"] = np.concatenate([f_, np.repeat(f_[-1:], 11 + a.K - f_.shape[0], 0)], 0)   # are never scored (padding)
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



from scipy.ndimage import map_coordinates


def interp(q, i0, i1):                                   # periodic cubic sampling of a (n,n) field at fractional indices
    return map_coordinates(q, [i0, i1], order=3, mode="grid-wrap")


def command(kind, n, amp, R, rng=None):                  # steady displacement field per step (data cells), (n,n,2)
    c = (n - 1) / 2; i0, i1 = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); x0, x1 = i0 - c, i1 - c
    g = np.exp(-(x0 ** 2 + x1 ** 2) / (2 * R * R))
    if kind in ("curl", "curl-"):                       # psi = g: u0 = d psi/d x1, u1 = -d psi/d x0  (counter-clockwise for +)
        u0, u1 = -x1 / R ** 2 * g, x0 / R ** 2 * g
    elif kind == "div":
        u0, u1 = -x0 / R ** 2 * g, -x1 / R ** 2 * g   # phi = -g: outward source
    else:
        z = rng.standard_normal((n, n, 2)); F = np.fft.fft2(z, axes=(0, 1))
        k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")); F *= (k < 3 / n)[..., None]
        z = np.real(np.fft.ifft2(F, axes=(0, 1))); u0, u1 = z[..., 0], z[..., 1]
    U = np.stack([u0, u1], -1); U *= amp / np.abs(np.hypot(U[..., 0], U[..., 1])).max()
    return -U if kind == "curl-" else U


def flowmap(q, U, K, sub=4):                            # q advected by the steady field U for K steps (backward characteristics)
    n = q.shape[0]; i0, i1 = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); p0, p1 = i0.astype(float), i1.astype(float)
    h = -1.0 / sub
    for _ in range(K * sub):                              # RK4 backward in time
        a0, a1 = interp(U[..., 0], p0, p1), interp(U[..., 1], p0, p1)
        b0, b1 = interp(U[..., 0], p0 + .5 * h * a0, p1 + .5 * h * a1), interp(U[..., 1], p0 + .5 * h * a0, p1 + .5 * h * a1)
        c0, c1 = interp(U[..., 0], p0 + .5 * h * b0, p1 + .5 * h * b1), interp(U[..., 1], p0 + .5 * h * b0, p1 + .5 * h * b1)
        d0, d1 = interp(U[..., 0], p0 + h * c0, p1 + h * c1), interp(U[..., 1], p0 + h * c0, p1 + h * c1)
        p0 += h / 6 * (a0 + 2 * b0 + 2 * c0 + d0); p1 += h / 6 * (a1 + 2 * b1 + 2 * c1 + d1)
    return interp(q, p0, p1)


class NSext(NS2D):                                       # NS with an external steady advecting velocity (physical units)
    def __init__(self, n, nu, Uext):
        super().__init__(n, nu); self.U0, self.U1 = Uext[..., 0], Uext[..., 1]

    def rhs(self, wh, mean):
        u, v = self.vel(wh, mean)
        wx = np.real(np.fft.ifft2(1j * self.kx * wh)); wy = np.real(np.fft.ifft2(1j * self.ky * wh))
        return -np.fft.fft2((u + self.U0) * wx + (v + self.U1) * wy) * self.dealias - self.nu * self.k2 * wh


def vort(F):
    return (np.roll(F[..., 1], -1, -2) - np.roll(F[..., 1], 1, -2)) / 2 - (np.roll(F[..., 0], -1, -1) - np.roll(F[..., 0], 1, -1)) / 2


def score(d, r):
    d, r = d.ravel(), r.ravel()
    return (float(np.corrcoef(d, r)[0, 1]) if r.std() > 0 and d.std() > 0 else float("nan"), float((d * r).sum() / (r * r).sum()))



def shear_cmd(n, amp):
    i1 = np.arange(n)[None, :].repeat(n, 0); U = np.zeros((n, n, 2)); U[..., 0] = amp * np.sin(2 * np.pi * i1 / n); return U


def div2(F):                                              # central-difference divergence of (n,n,2), cells
    return g(F[..., 0], 0) + g(F[..., 1], 1)


def dratio(F):
    return float(np.linalg.norm(div2(F)) / (np.linalg.norm(F) + 1e-12))


store, report = {}, {}
INCOMP = ("NS-Sines", "incom_ns")
for fam in a.fams.split(","):
    units = list(itertools.islice(loader.units_for(fam, "test"), a.nA + a.nB))
    if len(units) < a.nA + a.nB: print("skip (units)", fam); continue
    bA = make(fam, units[:a.nA], range(a.nA)); bB = make(fam, units[a.nA:], range(a.nA, a.nA + a.nB))
    dims = tuple(int(x) for x in bA["dims"]); n = dims[0]; sl = [s for s in FAMILIES[fam].channels if s in SNAME]
    iu, iv = 0, 1; irho = 3 if 3 in sl else None             # SLOT ids (forecast arrays carry all S slots)
    baseA, fA = rollout(bA, a.K); baseB, _ = rollout(bB, a.K)
    PA = baseA.reshape(a.nA, a.K, *dims, -1); PB = baseB.reshape(a.nB, a.K, *dims, -1)
    uh = fA["u"]; Hb = uh.shape[1]; fp = n // Hb
    sd = [float(uh[..., c].std()) for c in (0, 1)]; eta = []
    for c in (0, 1):
        e = []
        for al in (-1.0, 1.0):
            st = np.zeros((1, 1, 1, 3), np.float32); st[..., c] = al * sd[c]
            p, _ = rollout(bA, 1, steer=st); Dd = p.reshape(a.nA, 1, *dims, -1) - PA[:, :1]
            e.append(np.mean([[-(Dd[i, 0, ..., s] * g(PA[i, 0, ..., s], c)).sum() / (g(PA[i, 0, ..., s], c) ** 2).sum()
                               for s in sl] for i in range(a.nA)]) / al)
        eta.append(float(np.mean(e)))
    scale = np.asarray(bB["scale"]).reshape(a.nB, -1)
    has_solver = fam == "NS-Sines"
    R = dict(eta=eta, slots=sl, conds={}); print(f"[{fam}] eta {eta[0]:.4f} / {eta[1]:.4f}", flush=True)
    kinds = (["shear", "curl"] if fam in INCOMP else []) + ["div", "div-"]
    if a.kinds and os.environ.get("MOTION_CTL_ANYFAM", "0") == "1":   # REVISION-R1 10-21: requested kinds on any family
        kinds = [k for k in ("shear", "curl", "div", "div-") if k in a.kinds.split(",")]
    if a.kinds: kinds = [k_ for k_ in kinds if k_ in a.kinds.split(",")]
    for kind in kinds:
        for amp in [float(x) for x in a.amps.split(",")]:
            amp_c = amp * float(np.mean(np.abs(eta)))
            U = (shear_cmd(n, amp_c) if kind == "shear" else command("curl", n, amp_c, a.rad) if kind == "curl"
                 else command("div", n, amp_c, a.rad) * (1 if kind == "div" else -1))
            Ulat = U.reshape(Hb, fp, Hb, fp, 2).mean((1, 3))
            st = np.zeros((1, Hb, Hb, 3), np.float32)
            for c in (0, 1): st[0, ..., c] = Ulat[..., c] / eta[c] * sd[c]
            res = []
            for i in range(a.nB):
                p, _ = rollout(one(bB, i), a.K, steer=st, persistent=True)
                if not np.isfinite(p).all(): res.append(None); continue
                P = p.reshape(a.K, *dims, -1); B = PB[i]; r = {}
                if has_solver:
                    u_ = units[a.nA + i]; F = np.asarray(u_["fields"], np.float64).reshape(-1, *dims, np.asarray(u_["fields"]).shape[-1])
                    s0 = F[9, ..., 0], F[9, ..., 1]; Uph = U * (1.0 / n) / a.dt
                    S0 = NS2D(n, a.nu).run(*s0, a.K + 1, a.dt)[1:]; S1 = NSext(n, a.nu, Uph).run(*s0, a.K + 1, a.dt)[1:]
                    Dyn = (S1 - S0) / scale[i, :2]
                if has_solver and a.horizon_all:            # REVISION-R1 10-20: long-horizon check (user)
                    cc = lambda x, y: float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
                    S1n, S0n = S1 / scale[i, :2], S0 / scale[i, :2]; H = []
                    for k in range(a.K):
                        Pk, Bk = P[k][..., [iu, iv]], B[k][..., [iu, iv]]
                        FMk = np.stack([flowmap(Bk[..., c_], U, k + 1) for c_ in range(2)], -1)   # kinematic edit of the base (user 10-20)
                        H.append(dict(r_fm_S1=cc(FMk, S1n[k]), r_fm_resp=cc(FMk - Bk, S1n[k] - S0n[k]), r_cmd_S1=cc(Pk, S1n[k]), r_base_S1=cc(Bk, S1n[k]), r_base_S0=cc(Bk, S0n[k]), r_S0_S1=cc(S0n[k], S1n[k]),
                                      r_resp=cc(Pk - Bk, S1n[k] - S0n[k]), change_model=float(np.linalg.norm(Pk - Bk) / np.linalg.norm(Bk)),
                                      change_solver=float(np.linalg.norm(S1n[k] - S0n[k]) / np.linalg.norm(S0n[k]))))
                    r["horizon"] = H
                    if i == 0:
                        for k in range(4, a.K, 5):                # fields every 5 steps (user 10-20)
                            store[f"{fam}|{kind}{amp:g}|hfields{k + 1}"] = np.stack([B[k][..., [iu, iv]], P[k][..., [iu, iv]], S0n[k], S1n[k],
                                np.stack([flowmap(B[k][..., s_], U, k + 1) for s_ in (iu, iv)], -1)]).astype(np.float32)
                for k in sorted({0, min(4, a.K - 1), a.K - 1}):
                    dP = (P[k] - B[k])[..., [iu, iv]]
                    Kin = np.stack([flowmap(B[k][..., s], U, k + 1) for s in (iu, iv)], -1) - B[k][..., [iu, iv]]
                    rk = dict(zip(("r_kin", "g_kin"), score(dP, Kin)))
                    _isc = 6 if 6 in sl else None                 # REVISION-R1 10-22: passive scalar scored separately
                    if _isc is not None:
                        _dc = P[k][..., _isc] - B[k][..., _isc]; _kc = flowmap(B[k][..., _isc], U, k + 1) - B[k][..., _isc]
                        rk.update(dict(zip(("r_kin_c", "g_kin_c"), score(_dc, _kc))))
                    rk.update(dict(zip(("r_kin_w", "g_kin_w"), score(vort(dP), vort(Kin)))))
                    rk.update(div_model=dratio(dP), div_kin=dratio(Kin))
                    if has_solver:
                        rk.update(dict(zip(("r_dyn", "g_dyn"), score(dP, Dyn[k]))))
                        rk.update(dict(zip(("r_dyn_w", "g_dyn_w"), score(vort(dP), vort(Dyn[k])))))
                        rk.update(div_dyn=dratio(Dyn[k]))
                    if irho is not None and k == 0 and kind != "shear":       # continuity split (compressible)
                        rho = np.asarray(bB["x_win"])[i, -1].reshape(*dims, -1)[..., irho]   # state the step starts from
                        if os.environ.get("MOTION_INPUT_ZSCORE", "0") == "1":   # REVISION-R1: z-score input -> add the mean back
                            rho = rho + float(np.asarray(bB["shift"])[i, irho] / np.asarray(bB["scale"])[i, irho])   # (units of sigma)
                        drho = P[0][..., irho] - B[0][..., irho]
                        t_adv = -(U[..., 0] * g(rho, 0) + U[..., 1] * g(rho, 1)); t_cmp = -rho * div2(U)
                        X = np.stack([t_adv.ravel(), t_cmp.ravel()], 1); coef = np.linalg.lstsq(X, drho.ravel(), rcond=None)[0]
                        cont = t_adv + t_cmp; kinr = flowmap(rho, U, 1) - rho
                        rk.update(a_adv=float(coef[0]), a_comp=float(coef[1]),
                                  r_cont=score(drho, cont)[0], g_cont=score(drho, cont)[1],
                                  r_kinrho=score(drho, kinr)[0], g_kinrho=score(drho, kinr)[1])
                        # REVISION-R1 10-20 (user): mass balance of control volumes centred on the command — model mass change vs
                        # the change the continuity equation requires (flux through the CV boundary); no density response = 0
                        cc_ = (n - 1) / 2; I0_, I1_ = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); Rr_ = np.hypot(I0_ - cc_, I1_ - cc_)
                        rk.update(cv_R=[5, 10, 15, 20, 25, 30, 40], cv_model=[float(drho[Rr_ <= q_].sum()) for q_ in (5, 10, 15, 20, 25, 30, 40)],
                                  cv_req=[float(cont[Rr_ <= q_].sum()) for q_ in (5, 10, 15, 20, 25, 30, 40)],
                                  net_model=float(drho.sum()), abs_model=float(np.abs(drho).sum()), mass_total=float(rho.sum()))
                        if i == 0 and amp == 1.0:
                            store[f"{fam}|{kind}|rho"] = np.stack([drho, cont, kinr]).astype(np.float32)
                        if i == 0:                       # FIELDS (main Fig. 2d): input, base step 1, commanded step 1,
                            store[f"{fam}|{kind}{amp:g}|rhofields"] = np.stack(        # base + continuity (GT), base + warp
                                [rho, B[0][..., irho], P[0][..., irho], B[0][..., irho] + cont, B[0][..., irho] + kinr]).astype(np.float32)
                            store[f"{fam}|{kind}{amp:g}|U"] = U.astype(np.float32)
                    r[k + 1] = rk
                if i == 0 and amp == 1.0:
                    store[f"{fam}|{kind}|vel"] = np.stack([(P - B)[[0, a.K - 1]][..., [iu, iv]],
                        np.stack([np.stack([flowmap(B[k][..., s], U, k + 1) for s in (iu, iv)], -1) - B[k][..., [iu, iv]] for k in (0, a.K - 1)])]).astype(np.float32)
                    if has_solver: store[f"{fam}|{kind}|dyn"] = Dyn[[0, a.K - 1]].astype(np.float32)
                if i == 0:                                       # FIELDS at step K (main Fig. 2c): model base, model
                    kk = a.K - 1                                 # commanded, flow map of the base, solver with command
                    Fl = [B[kk][..., [iu, iv]], P[kk][..., [iu, iv]],
                          np.stack([flowmap(B[kk][..., s], U, kk + 1) for s in (iu, iv)], -1)]
                    if has_solver: Fl += [S1[kk] / scale[i, :2], S0[kk] / scale[i, :2]]
                    store[f"{fam}|{kind}{amp:g}|fields"] = np.stack(Fl).astype(np.float32)
                    store[f"{fam}|{kind}{amp:g}|U"] = U.astype(np.float32)
                    store[f"{fam}|{kind}|U"] = U.astype(np.float32)
                res.append(r)
            R["conds"][f"{kind}{amp:g}"] = res
            ok = [x for x in res if x is not None]
            m = lambda k, key: np.nanmean([x[k][key] for x in ok if key in x[k]]) if any(key in x[k] for x in ok) else float("nan")
            line = f"[{fam}] {kind:6s} amp {amp:g}: " + " | ".join(
                f"k{k} kin r {m(k,'r_kin'):+.2f} g {m(k,'g_kin'):.2f}" + (f" scal r {m(k,'r_kin_c'):+.2f} g {m(k,'g_kin_c'):.2f}" if 6 in sl else "") + (f" dyn r {m(k,'r_dyn'):+.2f} g {m(k,'g_dyn'):.2f}" if has_solver else "")
                + f" div m {m(k,'div_model'):.3f} kin {m(k,'div_kin'):.3f}" + (f" dyn {m(k,'div_dyn'):.3f}" if has_solver else "")
                + (f" rho a_adv {m(k,'a_adv'):.2f} a_comp {m(k,'a_comp'):.2f} r_cont {m(k,'r_cont'):+.2f} r_kin {m(k,'r_kinrho'):+.2f}" if (irho is not None and k == 1 and kind != 'shear') else "")
                for k in sorted(k_ for k_ in ok[0] if not isinstance(k_, str)))   # skip the "horizon" entry
            print(line, flush=True)
    report[fam] = R
    np.savez_compressed(a.out, **store, __meta__=json.dumps({"args": vars(a), "report": report, "partial": True}))
np.savez_compressed(a.out, **store, __meta__=json.dumps({"args": vars(a), "report": report}))
print("CONTROL3_OK", a.out, flush=True)
