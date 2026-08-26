"""bylfa cfdbench conditional row-mirror TTA, corrected channel semantics (server GPU).

Finding: cfdbench prebuilt slots 0/1 are BOTH u-like scalars (corr~0.99), no
transverse velocity channel; flow is along +x with top/bottom walls. Under the
row (spanwise) mirror both channels are EVEN -> no sign flip. The earlier
blowup came from flipping a fake 'v'.

Legality per case, input-only: solid mask (zero cells across the input window)
must be row-mirror symmetric. Diagnostics: fr under {nosign, sign0, sign1}.
"""
import sys, glob, importlib
sys.path.insert(0, "/eu/code_mirror")
import numpy as np, tensorflow as tf
CFG = "motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa"
cfg = importlib.import_module(CFG).Cfg
if bool(getattr(cfg, "bf16", False)): tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
from motion_tf.model import PhysicsOperatorMixture
from motion_tf.utils import ckpt as ckptlib
from motion_tf.data import stream as streammod
_, test = streammod.make_train_stream(cfg, 8, "/eu/data/prose/prebuilt")
model = PhysicsOperatorMixture(cfg)
cks = sorted(glob.glob("/eu/results/prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa/ckpt_*.npz"),
             key=lambda p: int(p.split("_")[-1].split(".")[0]))
ckptlib.load(model, cks[-1]); print("loaded", cks[-1], flush=True)
names = list(test["fam_names"]); fi = names.index("cfdbench")
sel = np.where(test["fam"] == fi)[0][:100]
xin, xtg = test["xin"][sel], test["xtg"][sel]
dsc, cf = test["desc"][sel], test["coef"][sel]
mean, std = test["mean"][sel], test["std"][sel]
geom = np.ones((100, xin.shape[2], xin.shape[3], 1), np.float32)
_AR = int(getattr(cfg, "ar_seg", 0)); _SL = int(cfg.Nt) - 1

@tf.function
def ev(u0, dd, cc, g):
    if _AR >= 2:
        preds = []; win = u0
        for _k in range(_AR):
            pk = tf.cast(model.call_with_gate(win, dd, cc, geom_mask=g)[0], tf.float32)
            preds.append(pk[:, 1:1 + _SL])
            if _k < _AR - 1:
                win = tf.concat([win[:, _SL:], tf.cast(pk[:, 1:1 + _SL], u0.dtype)], axis=1)
        return tf.concat(preds, axis=1)
    return tf.cast(model.call_with_gate(u0, dd, cc, geom_mask=g)[0], tf.float32)[:, 1:1 + _SL]

def rowmir(a, sign):  # sign: () none, (0,) flip slot0, (1,) flip slot1
    a = np.flip(a, axis=a.ndim - 3).copy()
    for s in sign: a[..., s] *= -1
    return a

def predict(tf_in):  # tf_in: transform fn for inputs; inverse applied to outputs
    outs = []
    for i in range(0, 100, 20):
        xi = tf_in(xin[i:i+20])
        p = ev(tf.constant(xi), tf.constant(dsc[i:i+20]), tf.constant(cf[i:i+20]),
               tf.constant(geom[i:i+20])).numpy()[:, :10]
        outs.append(tf_in(p))
    return np.concatenate(outs, 0)

sc = std[:, None, None, None, :] + 1e-6; mc = mean[:, None, None, None, :]
gt = xtg[:, 1:11] * sc + mc
def rel(p, m=None):
    pp = p[:, :10] * sc + mc
    num = np.sqrt(((pp - gt) ** 2).sum((2, 3, 4))); den = np.sqrt((gt ** 2).sum((2, 3, 4))) + 1e-12
    r = 100 * (num / den).mean(1)
    return (r if m is None else r[m]).mean()

# input-only legality: solid mask (zeros across window, physical units) row-symmetric?
phys = xin * sc + mc
solid = (np.abs(phys[..., 0]) < 1e-6).all(axis=1) & (np.abs(phys[..., 1]) < 1e-6).all(axis=1)  # (B,H,W)
symok = np.array([(solid[b] == solid[b][::-1]).mean() for b in range(100)])
legal = symok > 0.999
print(f"solid-frac q: {np.round(np.quantile(solid.mean((1,2)), [0,.5,1]),3)}", flush=True)
print(f"row-symmetric mask: legal={legal.sum()}/100 (symok q {np.round(np.quantile(symok,[0,.25,.5,1]),4)})", flush=True)

Pid = predict(lambda a: a)
print(f"id = {rel(Pid):.3f}", flush=True)
variants = {"nosign": (), "sign0": (0,), "sign1": (1,)}
Pv = {}
for nm, sgn in variants.items():
    Pv[nm] = predict(lambda a, s=sgn: rowmir(a, s))
    line = f"fr[{nm}]: all={rel(Pv[nm]):.3f}"
    if legal.sum(): line += f"  legal-subset={rel(Pv[nm], legal):.3f} (id {rel(Pid, legal):.3f})"
    print(line, flush=True)
for nm in variants:
    ens = Pid.copy()
    ens[legal] = (Pid[legal] + Pv[nm][legal]) / 2
    print(f"conditional K2 ENS[{nm}] = {rel(ens):.3f} (single {rel(Pid):.3f}; legal subset ens "
          f"{rel(ens, legal) if legal.sum() else float('nan'):.3f} vs id {rel(Pid, legal) if legal.sum() else float('nan'):.3f})", flush=True)
print("DUMP_DONE", flush=True)

# ---- equivariance defect and errors, physical units, two real channels ----
_sc = sc[..., :2]; _mc = mc[..., :2]
_P0 = Pid[:, :10, ..., :2] * _sc + _mc
_Pf = Pv["nosign"][:, :10, ..., :2] * _sc + _mc
_GT = gt[..., :2]
def _rel(A):
    num = np.sqrt(((A - _GT) ** 2).sum(axis=(2, 3)))
    den = np.sqrt((_GT ** 2).sum(axis=(2, 3))) + 1e-12
    return float((num / den).mean() * 100)
_num = np.sqrt(((_Pf - _P0) ** 2).sum(axis=(2, 3)))
_den = np.sqrt((_P0 ** 2).sum(axis=(2, 3))) + 1e-12
_eps = float((_num / _den).mean() * 100)
print(f"[MOTION cfdbench PHYSICAL 2ch] eps={_eps:.3f}%  err_id={_rel(_P0):.3f}%  "
      f"err_mirror={_rel(_Pf):.3f}%  bound={_rel(_P0)+_rel(_Pf):.3f}%", flush=True)
print("DUMP_DONE", flush=True)
