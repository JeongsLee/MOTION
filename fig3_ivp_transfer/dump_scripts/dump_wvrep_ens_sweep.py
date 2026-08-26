"""Wave repaired (wv0pad) symmetry-ensemble ANCHOR SWEEP — matches the 11.89 convention
(mean over anchors {0,1,2,4,8} of per-anchor full-traj medians, lead<=12)."""
import os, sys, importlib, numpy as np
sys.path.insert(0, "/tmp/w"); sys.modules.pop("code", None)
import tensorflow as tf
ck, out = sys.argv[1], sys.argv[2]
m = importlib.import_module("motion_tf.train.configs.poseidon_finetune_combo_158m_wv0_pad"); cfg = m.Cfg
if bool(getattr(cfg, "bf16", False)):
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
from motion_tf.model import PhysicsOperatorMixture
from motion_tf.data.poseidon import build_poseidon_stream
from motion_tf.utils import ckpt as ckptlib
model = PhysicsOperatorMixture(cfg)
_ = model.call_with_gate(tf.zeros((1, cfg.Nx, cfg.Nx, 8)), tf.zeros((1, cfg.desc_dim)),
                         tf.zeros((1, 4)), training=False)
ckptlib.load(model, ck); print("restored", flush=True)
_, test = build_poseidon_stream(cfg, cfg.batch)
xt = np.asarray(test["traj"], np.float32); cm = np.asarray(test["cm"], np.float32)
mean = np.asarray(test["mean"], np.float32); std = np.asarray(test["std"], np.float32)
NTD = int(getattr(cfg, "nt_data", 0)) or xt.shape[1]
V0D, V0S = int(cfg.v0_to_slot), int(cfg.v0_from_slot)
GROUP = [(0, 0), (1, 0), (0, 1), (1, 1)]
def T(a, g):
    fr, fc = g
    if fr: a = np.flip(a, axis=a.ndim-3).copy(); a[..., 0] = -a[..., 0]
    if fc: a = np.flip(a, axis=a.ndim-2).copy(); a[..., 1] = -a[..., 1]
    return a
def predict(u):
    o = []
    for i in range(0, len(u), 8):
        b = tf.constant(u[i:i+8])
        o.append(tf.cast(model.call_with_gate(b, tf.zeros((len(b), cfg.desc_dim)),
                                              tf.zeros((len(b), 4)), training=False)[0],
                         tf.float32).numpy())
    return np.concatenate(o)[:, :NTD]
meds_s, meds_e = [], []
for A in (0, 1, 2, 4, 8):
    u0 = np.array(xt[:, A])
    u0[..., V0D] = (xt[:, A, :, :, V0S] - xt[:, A-1, :, :, V0S]) if A >= 1 else 0.0
    pg = []
    for g in GROUP:
        pg.append(T(predict(T(u0, g)), g))
    L = min(12, NTD - 1 - A)
    sc = std[:, None, None, None, :]; mc = mean[:, None, None, None, :]
    m5 = cm[:, None, None, None, :]
    rT = xt[:, 1+A:1+A+L]*sc + mc
    def med(p):
        pT = p[:, 1:1+L]*sc + mc
        num = np.sum(np.abs(pT-rT)*m5, axis=(2, 3, 4)); den = np.sum(np.abs(rT)*m5, axis=(2, 3, 4))+1e-8
        return float(np.median((num/den).mean(axis=1)))
    ms, me = med(pg[0]), med(np.mean(pg, axis=0))
    meds_s.append(ms); meds_e.append(me)
    print(f"[A={A}] single={ms*100:.3f}%  ens={me*100:.3f}%", flush=True)
print(f"[SWEPT] single={np.mean(meds_s)*100:.3f}% (target 11.89)  ens={np.mean(meds_e)*100:.3f}%", flush=True)
np.savez(out, anchors=np.array([0,1,2,4,8]), single=np.array(meds_s), ens=np.array(meds_e))
print("DUMP_DONE", flush=True)
