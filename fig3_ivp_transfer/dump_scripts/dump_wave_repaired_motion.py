"""Wave-Layer REPAIRED (two-field IC) dump — wv0pad arm, anchor 0 (v0=0, true rest IC).
Verify: lead<=12 full-traj median vs asweep12-wv0pad anchor-0 record 11.475."""
import os, sys, importlib, numpy as np
sys.path.insert(0, "/tmp/w"); sys.modules.pop("code", None)
import tensorflow as tf
ck, out = sys.argv[1], sys.argv[2]
m = importlib.import_module("motion_tf.train.configs.poseidon_finetune_combo_158m_wv0_pad"); cfg = m.Cfg
if bool(getattr(cfg, "bf16", False)):
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16"); print("bf16 on", flush=True)
from motion_tf.model import PhysicsOperatorMixture
from motion_tf.data.poseidon import build_poseidon_stream
from motion_tf.utils import ckpt as ckptlib
model = PhysicsOperatorMixture(cfg)
_ = model.call_with_gate(tf.zeros((1, cfg.Nx, cfg.Nx, 8)), tf.zeros((1, cfg.desc_dim)),
                         tf.zeros((1, 4)), training=False)
ckptlib.load(model, ck); print("restored", ck, flush=True)
_, test = build_poseidon_stream(cfg, cfg.batch)
xt = np.asarray(test["traj"], np.float32); cm = np.asarray(test["cm"], np.float32)
mean = np.asarray(test["mean"], np.float32); std = np.asarray(test["std"], np.float32)
N = xt.shape[0]; NTD = int(getattr(cfg, "nt_data", 0)) or xt.shape[1]
V0D = int(cfg.v0_to_slot); print("test", xt.shape, "NTD", NTD, "V0D", V0D, flush=True)
u0 = np.array(xt[:, 0]); u0[..., V0D] = 0.0                    # anchor 0: v(0)=0 (true rest IC)

def predict(u0_):
    o = []
    for i in range(0, len(u0_), 8):
        b = tf.constant(u0_[i:i+8])
        o.append(tf.cast(model.call_with_gate(b, tf.zeros((len(b), cfg.desc_dim)),
                                              tf.zeros((len(b), 4)), training=False)[0],
                         tf.float32).numpy())
    return np.concatenate(o)[:, :NTD]

pred = predict(u0)
sc = std[:, None, None, None, :]; mc = mean[:, None, None, None, :]; m5 = cm[:, None, None, None, :]
pT = pred[:, 1:]*sc + mc; rT = xt[:, 1:NTD]*sc + mc
num = np.sum(np.abs(pT-rT)*m5, axis=(2, 3, 4)); den = np.sum(np.abs(rT)*m5, axis=(2, 3, 4))+1e-8
r = num/den
print(f"[VERIFY wave-repaired a0] full-horizon median={np.median(r.mean(axis=1))*100:.3f}%  "
      f"lead<=12 median={np.median(r[:, :12].mean(axis=1))*100:.3f}%  (target 11.475)", flush=True)
np.savez_compressed(out, pred16=pred[:16], gt16=xt[:16], mean=mean, std=std, cm=cm,
                    ft12=r[:, :12].mean(axis=1).astype(np.float32),
                    ftfull=r.mean(axis=1).astype(np.float32))
print("DUMP_DONE", out, flush=True)
