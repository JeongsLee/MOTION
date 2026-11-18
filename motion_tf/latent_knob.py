"""Mechanism knobs (revision R1-3): change the relative strength of two prescribed operator pathways of a trained
MOTION model and compare the change of the forecast with a numerical solver in which the corresponding TERM of the
governing equation is changed by the same factor.
  ACE      (Allen-Cahn, u_t = Lap u - eps^2 u(u^2-1)):  knob 'diff' scales the diffusion heads, 'react' the reaction heads;
           reference: u_t = sd Lap u - sr eps^2 u(u^2-1)
  NS-Sines (omega_t + u.grad omega = nu Lap omega):      knob 'adv' scales the advection head, 'diff' the diffusion heads;
           reference: omega_t + alpha u.grad omega = s nu Lap omega
A knob multiplies the head's gated contribution to the tendency W (banks.ko_mask, the runtime knockout switch: factor
0 is the knockout) at every step of a 10-step rollout from the last observed frame. Response fields
(knob - base forecast) are compared with (modified solve - base solve) from the same observed frame: pattern
correlation r and gain (1 = the solver's change). Specificity: r of every knob against every solver change, plus the
solver-solver correlation of the two term changes (how separable the two references are at all).
  python latent_knob.py --ckpt C --fam ACE --out OUT.npz
"""
import argparse, itertools, json, os, zlib
import numpy as np
import tensorflow as tf
tf.random.set_seed(0); tf.config.experimental.enable_op_determinism()
try:                                                  # Keras 2.15 eager leak (see latent_vortex3.py)
    from keras.src.layers.activation import softmax as _smod
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
ap.add_argument("--fam", default="ACE"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--K", type=int, default=10)
ap.add_argument("--factors", default="0.5,0.8,1.25,2")
ap.add_argument("--diff_heads", default="diffusion", help="heads scaled by the diffusion knob (comma list)")
ap.add_argument("--only_traj", type=int, default=-1, help=">=0: run only this test trajectory (maps stored for it)")
ap.add_argument("--react_heads", default="reaction,reaction_nl", help="heads scaled by the reaction knob (comma list; H0 = state head)")
ap.add_argument("--nu", type=float, default=1.5e-4); ap.add_argument("--dt", type=float, default=0.05)
ap.add_argument("--ace_dt", type=float, default=2e-4 / 19)
ap.add_argument("--calib", type=int, default=0, help="1: fit the equivalent equation change (both term multipliers) to each knob response")
ap.add_argument("--lap_steer_heads", default="", help="non-empty: the 'diff' knob x f becomes the PRESCRIBED diffusion command: "
                "(f-1)*Lap(h) added to the Laplacian feature of these heads (no head bias scaled; banks.lap_steer)")
ap.add_argument("--calib_x0", default="cmd", help="calibration start: cmd = matched term x f (legacy); one = no change (unbiased check)")
ap.add_argument("--only_knob", default="", help="non-empty: run only this knob (e.g. diff)")
ap.add_argument("--track", type=int, default=0, help="1: diffusion knob only; fit ONE diffusion multiplier on step 1, then compare every step")
a = ap.parse_args()

from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS, SEMANTIC_SLOTS
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import BOX_TABLE_2D, _grid
import sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from ns_spectral import NS2D
from ac_spectral import AC2D, FHN2D
if os.environ.get("LOCAL_UNITS"):
    import pickle, types
    _LU = {}
    for p_ in os.environ["LOCAL_UNITS"].split(":"): _LU.update(pickle.load(open(p_, "rb")))
    def _restore(u):
        u = dict(u); f = u.pop("fields"); full = np.zeros(f.shape[:-1] + (int(u.pop("_S")),), f.dtype)
        full[..., u.pop("_act")] = f; u["fields"] = full
        from data.symbolic import OPERATOR_VOCAB
        if "op_multihot" in u and len(u["op_multihot"]) == len(OPERATOR_VOCAB) + 1:
            assert float(np.asarray(u["op_multihot"])[-1]) == 0.0; u["op_multihot"] = np.asarray(u["op_multihot"])[:-1]
        elif "op_multihot" in u and len(u["op_multihot"]) == len(OPERATOR_VOCAB) - 1:   # legacy export, 14-token model
            u["op_multihot"] = np.concatenate([np.asarray(u["op_multihot"]), [0.0]])
        return u
    loader = types.SimpleNamespace(units_for=lambda fam, split: (_restore(u) for u in _LU[fam]))

model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=16, n_p=16, d_cond=128, dims2=(64, 64), dims3=(32, 32, 32),
                t_in=10, transport=True, n_fams=len(FAMILIES), n_semantic=SEMANTIC_SLOTS, gate_cond=1, dict_decode=0, fam_head=1)
model.warm_build(); V = model.trainable_variables
load_ckpt(V, a.ckpt, d_w=16, names=[v.name for v in V])

KNOBS = {"ACE": {"diff": a.diff_heads.split(","), "react": ["reaction", "reaction_nl"]},
         "diff_react": {"diff": a.diff_heads.split(","), "react": a.react_heads.split(",")},
         "NS-Sines": {"adv": ["advection"], "diff": a.diff_heads.split(",")}}[a.fam]
HEADS = sorted({h for v in KNOBS.values() for h in v})
model.banks.ko_index = {h: i for i, h in enumerate(HEADS)}
model.banks.ko_mask = tf.Variable(tf.ones([len(HEADS)]), trainable=False, name="knob_mask")


def set_knob(knob=None, f=1.0):
    m = np.ones(len(HEADS), np.float32)
    model.banks.lap_steer = None
    if a.lap_steer_heads and knob == "diff":          # prescribed-operator command instead of a head scale
        model.banks.lap_steer = float(f) - 1.0; model.banks.lap_steer_heads = tuple(a.lap_steer_heads.split(","))
        model.banks.ko_mask.assign(m); return
    if knob is not None:
        for h in KNOBS[knob]: m[model.banks.ko_index[h]] = f
    model.banks.ko_mask.assign(m)


def rollout(b, K):
    T = lambda k: tf.constant(b[k])
    cn, xw, fm, gn = T("coords_node"), T("x_win"), T("fmask"), T("geom_node")
    qall = tf.concat([cn, T("coords_q")], 1); gall = tf.concat([gn, T("geom_q")], 1)
    u_prev = tf.concat([xw[:, -1], T("ic_q")], 1); gd = tuple(int(x) for x in b["dims"])
    Nn = int(cn.shape[1])
    if b.get("mode") == "ss":                          # single-shot families (ACE, diff_react): one whole-horizon call
        P = model.step_ss(cn, xw, fm, gn, int(b["K"]), [int(r) for r in b["roles"]], T("cond"), T("op_multihot"), qall, gall,
                          u_prev, kf=K, fam_id=T("fam_id"), sdf_box=T("sdf_box"), box_dims=tuple(int(v) for v in b["box_dims"]))
        return tf.cast(P, tf.float32).numpy()[0][:, :Nn]
    win, fmask = xw, fm; pg, pg2 = _grid(xw[:, -1], gd, model.S), _grid(xw[:, -2], gd, model.S)
    preds = []
    for k in range(K):
        pred = model.step(cn, win, fmask, gn, int(b["K"]), [int(r) for r in b["roles"]], T("cond"), T("op_multihot"), qall,
                          gall, u_prev, steady=False, prev_grid=pg, grid_dims=gd, fam_id=T("fam_id"), sdf_box=T("sdf_box"),
                          box_dims=tuple(int(v) for v in b["box_dims"]), fbmask=T("fbmask"), surf=False, prev2_grid=pg2)
        pred = tf.cast(pred[0] if isinstance(pred, (tuple, list)) else pred, tf.float32)
        preds.append(pred[:, :Nn].numpy()); u_prev = pred; pn = pred[:, :Nn]
        win = tf.concat([win[:, 1:], pn[:, None]], 1); fmask = tf.concat([fmask[:, 1:], tf.ones_like(fmask[:, :1])], 1)
        pg2, pg = pg, _grid(pn, gd, model.S)
    return np.stack(preds, 1)[0]


def probe_heads(b):                                   # mean |gated contribution| of every head at the first step
    model.banks.probe = {}
    if b.get("mode") == "ss":
        rollout(b, a.K); p = {k: float(v) for k, v in model.banks.probe.items() if not k.endswith("_field")}
        model.banks.probe = None; return p
    T = lambda k: tf.constant(b[k]); cn, xw = T("coords_node"), T("x_win"); gd = tuple(int(x) for x in b["dims"])
    model.step(cn, xw, T("fmask"), T("geom_node"), int(b["K"]), [int(r) for r in b["roles"]], T("cond"), T("op_multihot"),
               tf.concat([cn, T("coords_q")], 1), tf.concat([T("geom_node"), T("geom_q")], 1),
               tf.concat([xw[:, -1], T("ic_q")], 1), steady=False, prev_grid=_grid(xw[:, -1], gd, model.S), grid_dims=gd,
               fam_id=T("fam_id"), sdf_box=T("sdf_box"), box_dims=tuple(int(v) for v in b["box_dims"]), fbmask=T("fbmask"),
               surf=False, prev2_grid=_grid(xw[:, -2], gd, model.S))
    p = {k: float(v) for k, v in model.banks.probe.items() if not k.endswith("_field")}; model.banks.probe = None
    return p


def score(d, r):
    d, r = d.ravel(), r.ravel()
    ok = r.std() > 0 and d.std() > 0
    return (float(np.corrcoef(d, r)[0, 1]) if ok else float("nan"), float((d * r).sum() / (r * r).sum()) if ok else float("nan"))


def vort(F):
    return (np.roll(F[..., 1], -1, -2) - np.roll(F[..., 1], 1, -2)) / 2 - (np.roll(F[..., 0], -1, -1) - np.roll(F[..., 0], 1, -1)) / 2


class NSmod(NS2D):                                    # omega_t + alpha u.grad omega = s nu Lap omega
    def __init__(self, n, nu, alpha=1.0, s=1.0):
        super().__init__(n, nu * s); self.alpha = alpha

    def rhs(self, wh, mean):
        u, v = self.vel(wh, mean)
        wx = np.real(np.fft.ifft2(1j * self.kx * wh)); wy = np.real(np.fft.ifft2(1j * self.ky * wh))
        return -self.alpha * np.fft.fft2(u * wx + v * wy) * self.dealias - self.nu * self.k2 * wh


def solve(F9, knob, f):                               # (K+1, n, n, C) from the observed frame 9, term `knob` scaled by f
    n = F9.shape[0]
    if a.fam == "ACE":
        kw = {"diff": dict(sd=f), "react": dict(sr=f)}.get(knob, {})
        return AC2D(n, **kw).run(F9[..., 0], a.K + 1, a.ace_dt)[..., None]
    if a.fam == "diff_react":                         # PDEBench FHN, Neumann, L 2, dt_frame 0.05 (identified, R^2 .999)
        kw = {"diff": dict(sd=f), "react": dict(sr=f)}.get(knob, {})
        return FHN2D(n, **kw).run(F9, a.K + 1, 0.05)
    kw = {"adv": dict(alpha=f), "diff": dict(s=f)}.get(knob, {})
    return NSmod(n, a.nu, **kw).run(F9[..., 0], F9[..., 1], a.K + 1, a.dt)


def solve2(F9, m):                                   # both term multipliers at once: m = {knob: factor}
    n = F9.shape[0]
    if a.fam == "diff_react":
        return FHN2D(n, sd=m.get("diff", 1.0), sr=m.get("react", 1.0)).run(F9, a.K + 1, 0.05)
    if a.fam == "NS-Sines":
        return NSmod(n, a.nu, alpha=m.get("adv", 1.0), s=m.get("diff", 1.0)).run(F9[..., 0], F9[..., 1], a.K + 1, a.dt)
    raise ValueError(a.fam)


def calibrate(F9, S0, dP, sc, kn, f):
    """Equivalent equation change: multipliers (m1, m2) of the two terms whose solver response best reproduces the
    knob response dP over all K frames (Nelder-Mead in log space, start = matched term x f). Returns m, R^2, rel err."""
    from scipy.optimize import minimize
    ks = list(KNOBS); y = dP.ravel(); yy = float((y * y).sum()) + 1e-30
    def loss(z):
        m = dict(zip(ks, np.exp(np.clip(z, -4, 4))))
        D = ((solve2(F9, m)[1:] - S0) / sc).ravel(); return float(((y - D) ** 2).sum()) / yy
    z0 = np.array([np.log(max(f, 0.05)) if (k == kn and a.calib_x0 == "cmd") else 0.0 for k in ks])
    r = minimize(loss, z0, method="Nelder-Mead", options=dict(xatol=0.02, fatol=1e-4, maxiter=(60 if a.calib_x0 == "cmd" else 150), initial_simplex=np.array([z0, z0 + [0.4, 0], z0 + [0, 0.4]])))
    return {k: float(np.exp(np.clip(v, -4, 4))) for k, v in zip(ks, r.x)}, float(1 - r.fun), int(r.nfev)


def track(F9, S0, B, P, sc, f):
    """Diffusion-only calibration on step 1, prediction test on steps 2..K.  Returns per-step dict."""
    from scipy.optimize import minimize_scalar
    y1 = (P[0] - B[0]).ravel()
    def loss(z):
        S1 = solve2(F9, {"diff": float(np.exp(z))})[1:2]; d = ((S1[0] - S0[0]) / sc).ravel()
        return float(((y1 - d) ** 2).sum())
    zs = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20.0)), method="bounded", options=dict(xatol=0.01)).x
    sfit = float(np.exp(zs)); Sf = solve2(F9, {"diff": sfit})[1:]
    out = {"s_fit": sfit, "steps": {}}
    for k in range(a.K):
        dP, dS = P[k] - B[k], (Sf[k] - S0[k]) / sc
        r_, g_ = score(dP, dS)
        e_knob = float(np.linalg.norm(P[k] - Sf[k] / sc) / np.linalg.norm(Sf[k] / sc))
        e_base = float(np.linalg.norm(B[k] - S0[k] / sc) / np.linalg.norm(S0[k] / sc))
        out["steps"][str(k + 1)] = {"r": r_, "gain": g_, "err_knob": e_knob, "err_base": e_base,
                                    "rel_change_solver": float(np.linalg.norm(dS) / np.linalg.norm(S0[k] / sc))}
        if NSF:
            out["steps"][str(k + 1)]["r_w"] = score(vort(dP), vort(dS))[0]
    return out, Sf


units = list(itertools.islice(loader.units_for(a.fam, "test"), a.n))
dims = tuple(int(x) for x in units[0]["dims"]); n = dims[0]
SL = list(FAMILIES[a.fam].channels)[:1] if a.fam == "ACE" else list(FAMILIES[a.fam].channels) if a.fam == "diff_react" else [0, 1]
NSF = a.fam == "NS-Sines"                             # vorticity scores / maps only for the flow
facs = [float(x) for x in a.factors.split(",")]
knobs = list(KNOBS); report = {"heads": KNOBS, "probe": [], "traj": []}; store = {}
for i, u in enumerate(units):
    if a.only_traj >= 0 and i != a.only_traj: continue
    bd = (int(BOX_TABLE_2D.get(a.fam, 64)),) * 2
    b = stack_batch([make_example_ar(u, 10, a.K, 4, 2048, 8192, np.random.default_rng(zlib.crc32(f"{a.fam}:{i}:0".encode())),
                                     box_dims=bd)])
    sc = np.asarray(b["scale"]).reshape(-1)[SL]
    set_knob(); report["probe"].append(probe_heads(b))
    B = rollout(b, a.K).reshape(a.K, *dims, -1)[..., SL]
    F = np.asarray(u["fields"], np.float64).reshape(-1, *dims, np.asarray(u["fields"]).shape[-1])[..., SL]
    S0 = solve(F[9], None, 1.0)[1:]
    tr = {"model_err": [float(np.linalg.norm(B[k] * sc - F[10 + k]) / np.linalg.norm(F[10 + k])) for k in range(min(a.K, F.shape[0] - 10))],
          "solver_err": [float(np.linalg.norm(S0[k] - F[10 + k]) / np.linalg.norm(F[10 + k])) for k in range(min(a.K, F.shape[0] - 10))],
          "conds": {}}
    for f in facs:
        Dsol = {kn: (solve(F[9], kn, f)[1:] - S0) / sc for kn in knobs}
        kk = knobs[0], knobs[1]
        ss = {str(k + 1): score(Dsol[kk[0]][k], Dsol[kk[1]][k])[0] for k in range(a.K)}   # separability of the references
        for kn in knobs:
            if a.only_knob and kn != a.only_knob: continue
            set_knob(kn, f); P = rollout(b, a.K).reshape(a.K, *dims, -1)[..., SL]; set_knob()
            dP = P - B; r = {"solver_solver_r": ss}
            for k in range(a.K):
                e = {f"r_{rn}": score(dP[k], Dsol[rn][k])[0] for rn in knobs}
                e["gain"] = score(dP[k], Dsol[kn][k])[1]
                # joint regression on both reference changes (the two term changes are correlated with each other):
                # dP ~ c1 D1 + c2 D2, coefficients in units of each reference (1 = the solver's change), and R^2
                X = np.stack([Dsol[rn][k].ravel() for rn in knobs], 1); y = dP[k].ravel()
                cf, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ cf
                e.update({f"c_{rn}": float(c) for rn, c in zip(knobs, cf)}); e["R2_joint"] = float(1 - res.var() / (y.var() + 1e-30))
                e["rel_change"] = float(np.linalg.norm(dP[k]) / (np.linalg.norm(B[k]) + 1e-12))
                if NSF:
                    for rn in knobs: e[f"r_{rn}_w"] = score(vort(dP[k]), vort(Dsol[rn][k]))[0]
                r[str(k + 1)] = e
            if a.track and kn == "diff" and f != 1.0:
                tk, Sf = track(F[9], S0, B, P, sc, f); r["track"] = tk
                print(f"   [track] diff x{f:g} -> nu/D x{tk['s_fit']:.2f} (fit on step 1) | " + " ".join(
                    f"k{k}: r {tk['steps'][str(k)]['r']:+.2f} g {tk['steps'][str(k)]['gain']:.2f} err {tk['steps'][str(k)]['err_knob']:.3f}/{tk['steps'][str(k)]['err_base']:.3f}"
                    for k in sorted({1, 2, 3, min(5, a.K), a.K})), flush=True)
                if i == 0:
                    fld = vort if NSF else (lambda X: X[..., 0])
                    ks_ = sorted({0, min(4, a.K - 1), a.K - 1})
                    store[f"track{f:g}|model"] = np.stack([fld(P[k] - B[k]) for k in ks_]).astype(np.float32)
                    store[f"track{f:g}|solver"] = np.stack([fld((Sf[k] - S0[k]) / sc) for k in ks_]).astype(np.float32)
            if a.calib and f != 1.0:
                m_, r2_, nf_ = calibrate(F[9], S0, dP, sc, kn, f)
                r["calib"] = {"mult": m_, "R2": r2_, "nfev": nf_}
                print(f"   [calib] {kn} x{f:g} -> equivalent equation change " + " ".join(f"{k} x{v:.2f}" for k, v in m_.items())
                      + f"  (R2 {r2_:.2f}, {nf_} solves)", flush=True)
            tr["conds"][f"{kn}{f:g}"] = r
            if i == max(a.only_traj, 0):
                fld = vort if NSF else (lambda X: X[..., 0])
                store[f"{kn}{f:g}|model"] = np.stack([fld(dP[k]) for k in sorted({0, min(4, a.K - 1), a.K - 1})]).astype(np.float32)
                store[f"{kn}{f:g}|model_uv"] = np.stack([dP[k] for k in sorted({0, min(4, a.K - 1), a.K - 1})]).astype(np.float32)   # raw response (velocity for NS)
                store[f"{kn}{f:g}|solver"] = np.stack([fld(Dsol[kn][k]) for k in sorted({0, min(4, a.K - 1), a.K - 1})]).astype(np.float32)
            print(f"[traj {i}] {kn} x{f:g}: " + " | ".join(
                f"k{k} " + " ".join(f"r_{rn} {r[str(k)]['r_' + rn]:+.2f}" for rn in knobs) + f" g {r[str(k)]['gain']:.2f} c " + "/".join(f"{r[str(k)]['c_' + rn]:+.2f}" for rn in knobs) + f" R2 {r[str(k)]['R2_joint']:.2f} (sol-sol {ss[str(k)]:+.2f})"
                for k in sorted({1, min(5, a.K), a.K})), flush=True)
    if i == max(a.only_traj, 0):
        store["base"] = np.stack([(vort(B[k]) if NSF else B[k][..., 0]) for k in sorted({0, min(4, a.K - 1), a.K - 1})]).astype(np.float32)
    print(f"[traj {i}] model err {np.round(tr['model_err'][::3], 3)} solver err {np.round(tr['solver_err'][::3], 3)} | probe "
          + " ".join(f"{k} {v:.3g}" for k, v in sorted(report['probe'][-1].items(), key=lambda x: -x[1])[:8]), flush=True)
    report["traj"].append(tr)
    np.savez_compressed(a.out, **store, __meta__=json.dumps({"args": vars(a), "report": report, "partial": True}))
np.savez_compressed(a.out, **store, __meta__=json.dumps({"args": vars(a), "report": report}))
print("KNOB_OK", a.out, flush=True)
