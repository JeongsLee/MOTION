"""MOTION(bylfa) uncond K2 symmetry ensemble — LOCAL run.
Verify positional load by reproducing id=5.191 (valid 4-frame window), then K2 ens."""
import os, sys, importlib, numpy as np
sys.path.insert(0, "/mnt/d/working/2026/neuraladaf/v6/fig3_local/bylfa_code")
sys.modules.pop("code", None)
import tensorflow as tf
CFG = "motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_arh5_bylfa"
cfg = importlib.import_module(CFG).Cfg
if bool(getattr(cfg, "bf16", False)):
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16"); print("bf16 on", flush=True)
from motion_tf.model import PhysicsOperatorMixture
from motion_tf.utils import ckpt as ckptlib
model = PhysicsOperatorMixture(cfg)
d = np.load("/mnt/d/working/2026/neuraladaf/v6/fig3_local/bylfa_unc_test.npz")
xin, xtg = d["xin"], d["xtg"]
mean, std, dsc, cf = d["mean"], d["std"], d["desc"], d["coef"]
geom = np.ones((xin.shape[0], xin.shape[2], xin.shape[3], 1), np.float32)
_AR = int(getattr(cfg, "ar_seg", 0)); _SL = int(cfg.Nt) - 1
# build vars then positional load
_ = model.call_with_gate(tf.constant(xin[:1]), tf.constant(dsc[:1]), tf.constant(cf[:1]),
                         geom_mask=tf.constant(geom[:1]))
ckptlib.load(model, "/mnt/d/working/2026/neuraladaf/v6/pdefoundation/paper/weights/bylfa_final/ckpt_160000.npz")
print("loaded ckpt", flush=True)

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

def T(a, g):
    fr, fc = g
    axH = a.ndim - 3; axW = a.ndim - 2
    if fr: a = np.flip(a, axis=axH).copy(); a[..., 0] = -a[..., 0]
    if fc: a = np.flip(a, axis=axW).copy(); a[..., 1] = -a[..., 1]
    return a

VT = 4
sc = std[:, None, None, None, :] + 1e-6; mc = mean[:, None, None, None, :]
gt = xtg[:, 1:1+VT] * sc + mc
def rel(p):
    pp = p[:, :VT] * sc + mc
    num = np.sqrt(((pp - gt) ** 2).sum((2, 3, 4))); den = np.sqrt((gt ** 2).sum((2, 3, 4))) + 1e-12
    return 100 * (num / den).mean()

preds = {}
for g in [(0, 0), (1, 0)]:
    outs = []
    for i in range(0, 100, 4):
        xi = T(xin[i:i+4], g)
        p = ev(tf.constant(xi), tf.constant(dsc[i:i+4]), tf.constant(cf[i:i+4]),
               tf.constant(geom[i:i+4])).numpy()[:, :10]
        outs.append(T(p, g))
    preds[g] = np.concatenate(outs)
    print(f"g={g}: rel={rel(preds[g]):.3f}%  (id target 5.191)", flush=True)
ens = (preds[(0, 0)] + preds[(1, 0)]) / 2
print(f"K2-ENS = {rel(ens):.3f}%", flush=True)
