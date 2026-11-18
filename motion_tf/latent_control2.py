"""Latent transport control with SPATIALLY VARYING commands (revision R1-3; A18).
A uniform latent-velocity offset gives a Galilean translation, which any shift-equivariant model reproduces
easily. Here the latent transport velocity is changed by a steady, spatially varying field dU(x), applied at every
step on held-out trajectories B, and the forecast is compared with the exact kinematic flow map of dU:
  shear : dU_0(x) = s(x_1), periodic sinusoid; flow map = per-row shift x_0 -> x_0 + K s(x_1)
  vortex: differential rotation about the centre, angle theta * w(r), w = 1 in the core (r < R) and a smooth
          taper to 0 at 1.6 R; flow map = rotation of each radius by its angle
  random: smooth random field with the same rms as the vortex field (control; no reference)
Latent units from the calibration on trajectories A: eta_c = cells per unit alpha per step for latent component c
along grid axis c (alpha = steer / std u(h)_c), so the latent field is alpha_c(x) = dU_c(x) [cells/step] / eta_c.
Reference: Ref = FlowMap_K(base forecast). Scores (per physical field): r_diff = corr(steered - base, Ref - base)
over the region the command moves (|displacement| > 0.25 cells), and relL2 of steered and of base against Ref there.
Also the velocity-independent physics check: the commanded velocity is not added to the physical flow, so the
reference is kinematic (what the command asks for), not a solution of the PDE.
  python latent_control2.py --ckpt C --out OUT.npz [--fams ...]
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
ap.add_argument("--thetas", default="30,60", help="rotation angle at the vortex centre after K steps (deg)")
ap.add_argument("--shears", default="2,4", help="shear amplitude: max displacement after K steps (cells)")
ap.add_argument("--R", type=float, default=0.22, help="vortex core radius (fraction of the domain)")
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
model = V3Model(S=NUM_SLOTS, d=ARCH[0], depth=ARCH[1], d_w=ARCH[2], n_p=16, d_cond=128, dims2=(64, 64), dims3=(32, 32, 32),
                t_in=10, transport=True, n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS, gate_cond=1, dict_decode=0, fam_head=1)
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
    bd = (int(BOX_TABLE_2D.get(fam, 64)),) * 2; exs = []
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



from scipy.ndimage import map_coordinates


def flow(q, d0, d1):                              # content moves by (d0, d1) cells: q_new(x) = q(x - d), periodic
    n0, n1 = q.shape; i0, i1 = np.meshgrid(np.arange(n0), np.arange(n1), indexing="ij")
    return map_coordinates(q, [i0 - d0, i1 - d1], order=3, mode="grid-wrap")


def vortex_disp(n, theta, R):                     # displacement after K steps for the differential rotation
    c = (n - 1) / 2; i0, i1 = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); x0, x1 = i0 - c, i1 - c
    r = np.hypot(x0, x1) / n; t = np.clip((r - R) / (0.6 * R), 0, 1); w = 1 - t * t * (3 - 2 * t)
    ph = np.deg2rad(theta) * w                    # rotate position by +ph: source point = R(-ph) x
    s0 = np.cos(ph) * x0 + np.sin(ph) * x1; s1 = -np.sin(ph) * x0 + np.cos(ph) * x1
    return x0 - s0, x1 - s1, w


def latent_field(kind, n_lat, amp, R, K, rng=None):  # per-step displacement field (cells of the DATA grid) on the latent grid
    c = (n_lat - 1) / 2; i0, i1 = np.meshgrid(np.arange(n_lat), np.arange(n_lat), indexing="ij")
    if kind == "shear":
        return np.stack([amp / K * np.sin(2 * np.pi * i1 / n_lat), np.zeros_like(i0, float)], -1)
    if kind == "vortex":                          # instantaneous velocity of the differential rotation (angle/step x radius)
        x0, x1 = (i0 - c) / n_lat, (i1 - c) / n_lat; r = np.hypot(x0, x1)
        t = np.clip((r - R) / (0.6 * R), 0, 1); w = 1 - t * t * (3 - 2 * t); om = np.deg2rad(amp) / K * w
        return np.stack([-om * (i1 - c), om * (i0 - c)], -1)        # in latent cells/step; scaled to data cells below
    z = rng.standard_normal((n_lat, n_lat, 2)); F = np.fft.fft2(z, axes=(0, 1))
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n_lat), np.fft.fftfreq(n_lat), indexing="ij")); F *= (k < 3 / n_lat)[..., None]
    return np.real(np.fft.ifft2(F, axes=(0, 1)))


def score(P, B, Ref, mask):
    dP, dR = (P - B)[mask], (Ref - B)[mask]
    rl = lambda x: float(np.linalg.norm((x - Ref)[mask]) / np.linalg.norm(Ref[mask]))
    return dict(r_diff=float(np.corrcoef(dP, dR)[0, 1]) if dR.std() > 0 else np.nan,
                gain=float((dP * dR).sum() / (dR * dR).sum()), relL2_steer=rl(P), relL2_base=rl(B))


store, report = {}, {}
for fam in a.fams.split(","):
    units = list(itertools.islice(loader.units_for(fam, "test"), a.nA + a.nB))
    if len(units) < a.nA + a.nB: print("skip (units)", fam); continue
    bA = make(fam, units[:a.nA], range(a.nA)); bB = make(fam, units[a.nA:], range(a.nA, a.nA + a.nB))
    if not bool(bA["dense2d"]): print("skip", fam); continue
    dims = tuple(int(x) for x in bA["dims"]); n = dims[0]; sl = [s for s in FAMILIES[fam].channels if s in SNAME]
    baseA, fA = rollout(bA, a.K); baseB, fB = rollout(bB, a.K)
    PA = baseA.reshape(a.nA, a.K, *dims, -1); PB = baseB.reshape(a.nB, a.K, *dims, -1)
    uh = fA["u"]; Hb = uh.shape[1]; fp = n // Hb
    # ---- calibration on A: eta_c (cells per alpha per step along axis c for latent component c), alpha = +/-1 std
    sd = [float(uh[..., c].std()) for c in (0, 1)]; eta = []
    for c in (0, 1):
        e = []
        for al in (-1.0, 1.0):
            st = np.zeros((1, 1, 1, 3), np.float32); st[..., c] = al * sd[c]
            p, _ = rollout(bA, 1, steer=st); D = p.reshape(a.nA, 1, *dims, -1) - PA[:, :1]
            e.append(np.mean([[-(D[i, 0, ..., s] * g(PA[i, 0, ..., s], c)).sum() / (g(PA[i, 0, ..., s], c) ** 2).sum()
                               for s in sl] for i in range(a.nA)]) / al)
        eta.append(float(np.mean(e)))
    R = dict(eta=eta, sd_uh=sd, slots=sl, Hb=Hb); print(f"[{fam}] eta axis0 {eta[0]:.4f} axis1 {eta[1]:.4f} cells/alpha/step", flush=True)
    rng = np.random.default_rng(0); conds = [("shear", s) for s in map(float, a.shears.split(","))] + \
        [("vortex", t) for t in map(float, a.thetas.split(","))] + [("random", float(a.thetas.split(",")[-1]))]
    for kind, amp in conds:
        f_lat = latent_field(kind, Hb, amp, a.R, a.K, rng)                          # per-step displacement, latent cells
        if kind != "shear": f_lat = f_lat * fp                                     # -> data cells (shear already in data cells)
        if kind == "random":                                                       # same rms as the vortex command
            v = latent_field("vortex", Hb, amp, a.R, a.K) * fp; f_lat *= np.sqrt((v ** 2).mean() / (f_lat ** 2).mean())
        st = np.zeros((1, Hb, Hb, 3), np.float32)
        for c in (0, 1): st[0, ..., c] = f_lat[..., c] / eta[c] * sd[c]
        if kind == "shear":
            i1 = np.arange(n)[None, :].repeat(n, 0); d0, d1 = amp * np.sin(2 * np.pi * i1 / n), np.zeros((n, n)); w = np.abs(d0) > 0.25
        elif kind == "vortex":
            d0, d1, wv = vortex_disp(n, amp, a.R); w = np.hypot(d0, d1) > 0.25
        res = []
        for i in range(a.nB):
            p, _ = rollout(one(bB, i), a.K, steer=st, persistent=True)
            if not np.isfinite(p).all(): res.append(None); continue
            P1 = p.reshape(1, a.K, *dims, -1)[0, -1]; B1 = PB[i, -1]
            if kind == "random":
                res.append({SNAME[s]: dict(rms_change=float(np.sqrt(((P1[..., s] - B1[..., s]) ** 2).mean()) /
                                                          np.sqrt((B1[..., s] ** 2).mean()))) for s in sl}); continue
            sc = {}
            for s in sl:
                Ref = flow(B1[..., s].astype(np.float64), d0, d1); sc[SNAME[s]] = score(P1[..., s], B1[..., s], Ref, w)
                if i == 0: store[f"{fam}|{kind}{amp:g}|{SNAME[s]}"] = np.stack([B1[..., s], P1[..., s], Ref]).astype(np.float32)
            res.append(sc)
        R[f"{kind}{amp:g}"] = res
        ok = [r for r in res if r is not None]
        if kind == "random":
            print(f"[{fam}] random (vortex rms): rel. change " + ", ".join(f"{k} {np.mean([r[k]['rms_change'] for r in ok]):.3f}" for k in ok[0]), flush=True)
        else:
            print(f"[{fam}] {kind} {amp:g}: " + " | ".join(
                f"{k} r_diff {np.mean([r[k]['r_diff'] for r in ok]):+.2f} gain {np.mean([r[k]['gain'] for r in ok]):.2f} "
                f"relL2 {np.mean([r[k]['relL2_steer'] for r in ok]):.3f}/{np.mean([r[k]['relL2_base'] for r in ok]):.3f}"
                for k in ok[0]), flush=True)
    report[fam] = R
    np.savez_compressed(a.out, **store, __meta__=json.dumps({"ckpt": a.ckpt, "args": vars(a), "report": report, "partial": True}))
np.savez_compressed(a.out, **store, __meta__=json.dumps({"ckpt": a.ckpt, "args": vars(a), "report": report}))
print("CONTROL2_OK", a.out, flush=True)
