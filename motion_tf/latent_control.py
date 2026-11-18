"""Latent transport CONTROL (revision R1-3): can the forecast's transport be commanded through the learned latent velocity?
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
ap.add_argument("--targets", default="-6,-4,-2,-1,1,2,4,6")
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


HB = {"p": None}                                      # unsteered forecast (K, Nn, S) for --hist_base


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
    # ---- calibration 2: eta on A (alpha = +/-1, first step), mean over fields and trajectories
    sd0 = float(uh[..., 0].std()); etas = []
    for al in (-1.0, 1.0):
        st = np.zeros((1, 1, 1, 3), np.float32); st[..., 0] = al * sd0
        p, _ = rollout(bA, 1, steer=st); D = p.reshape(a.nA, 1, *dims, -1) - PA[:, :1]
        etas.append(np.mean([[-(D[i, 0, ..., s] * g(PA[i, 0, ..., s], 0)).sum() / (g(PA[i, 0, ..., s], 0) ** 2).sum() for s in sl]
                             for i in range(a.nA)]) / al)
    eta = float(np.mean(etas))                                                          # cells per unit alpha per step
    R = dict(ax_slot=ax_slot, gain=gain, kappa=kappa, kappa_term=best, kappa_r2=kappa_r2, eta=eta, sd_uh=sd0, scaleB=scaleB.tolist(), slots=sl)
    print(f"[{fam}] calib: axis-0 velocity slot {ax_slot[0]}, gain {gain:.3f}, dt/dx {kappa:.3f} ({best}, R2 {kappa_r2:.2f}), eta {eta:.4f} cells/alpha/step", flush=True)
    # ---- control on B
    yB = np.asarray(bB["y_node"]).reshape(a.nB, -1, *dims, np.asarray(bB["y_node"]).shape[-1])
    for mode in ("ucal", "ueta"):
        res = {}
        for D in TG:
            ach = np.zeros((a.nB, len(sl))); lat = np.zeros((a.nB, len(sl))); alphas = []
            for i in range(a.nB):
                if mode == "ucal":   # physical velocity change dU (raw) = D/(K kappa); normalized = dU/scale; latent = gain*that
                    al = (D / (a.K * kappa)) / scaleB[i] * gain / sd0
                else:
                    al = D / (a.K * eta)
                alphas.append(al)
                st = np.zeros((1, 1, 1, 3), np.float32); st[..., 0] = al * sd0
                if a.hist_base: HB["p"] = PB[i].reshape(a.K, -1, PB.shape[-1]).astype(np.float32)
                p, _ = rollout(one(bB, i), a.K, steer=st, persistent=True); HB["p"] = None
                if p is None or not np.isfinite(p).all(): ach[i] = np.nan; continue
                P1 = p.reshape(1, a.K, *dims, -1)[0, -1]
                if a.per_step and mode == "ueta":       # achieved displacement after every step (linear in k if no compounding)
                    Pk = p.reshape(1, a.K, *dims, -1)[0]
                    res.setdefault(f"{D:g}|per_step", []).append(
                        [[best_shift(Pk[k][..., s], PB[i, k, ..., s], 0) for s in sl] for k in range(a.K)])
                for j, s in enumerate(sl):
                    ach[i, j] = best_shift(P1[..., s], PB[i, -1, ..., s], 0)
                    lat[i, j] = best_shift(P1[..., s], PB[i, -1, ..., s], 1, -5, 5)
                if mode == "ueta" and D == 4.0 and i == 0:
                    store[f"{fam}|contour|steer"] = P1[..., sl].astype(np.float32)
                    store[f"{fam}|contour|base"] = PB[i, -1][..., sl].astype(np.float32)
                    store[f"{fam}|contour|gt"] = yB[i, a.K - 1][..., sl].astype(np.float32)
            res[str(D)] = dict(achieved=ach.tolist(), lateral=lat.tolist(), alpha=alphas)
            print(f"[{fam}] {mode} D*={D:+.0f}: alpha {np.mean(alphas):+.2f} | achieved per field " +
                  ", ".join(f"{SNAME[s]} {np.nanmean(ach[:, j]):+.2f}" for j, s in enumerate(sl)) +
                  f" | lateral {np.nanmean(np.abs(lat)):.2f}", flush=True)
        R[mode] = res
    # contour similarity (D* = +4, mode ueta, trajectory B0): steered vs translated base / translated ground truth
    if f"{fam}|contour|steer" in store:
        S_, B_, G_ = store[f"{fam}|contour|steer"], store[f"{fam}|contour|base"], store[f"{fam}|contour|gt"]
        sim = {}
        for j, s in enumerate(sl):
            TB, TG_ = tshift(B_[..., j], 4.0, 0), tshift(G_[..., j], 4.0, 0)
            rl = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
            sim[SNAME[s]] = dict(r_steer_vs_Tbase=float(np.corrcoef(S_[..., j].ravel(), TB.ravel())[0, 1]),
                                 relL2_steer_vs_Tbase=rl(S_[..., j], TB), relL2_base_vs_Tbase=rl(B_[..., j], TB),
                                 relL2_steer_vs_Tgt=rl(S_[..., j], TG_), relL2_base_vs_Tgt=rl(B_[..., j], TG_))
        R["contour_sim"] = sim; print(f"[{fam}] contour sim {sim}", flush=True)
    report[fam] = R
    np.savez_compressed(a.out, **store, __meta__=json.dumps({"ckpt": a.ckpt, "args": vars(a), "report": report, "partial": True}))
np.savez_compressed(a.out, **store, __meta__=json.dumps({"ckpt": a.ckpt, "args": vars(a), "report": report}))
print("CONTROL_OK", a.out, flush=True)
