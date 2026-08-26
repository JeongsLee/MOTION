"""Poisson dump v2 — load BEFORE any forward (train_ivp order), diagnose lazy-var creation."""
import os, sys, importlib, numpy as np
sys.path.insert(0, "/tmp/w"); sys.modules.pop("code", None)
import tensorflow as tf
ck, out = sys.argv[1], sys.argv[2]
m = importlib.import_module("motion_tf.train.configs.poseidon_finetune_combo_158m"); cfg = m.Cfg
from motion_tf.model import PhysicsOperatorMixture
from motion_tf.data.poseidon import build_poseidon_stream
from motion_tf.utils import ckpt as ckptlib
model = PhysicsOperatorMixture(cfg)
print("nvars after __init__:", len(model.trainable_variables), flush=True)
ckptlib.load(model, ck); print("loaded (pre-forward)", flush=True)
_ = model.call_with_gate(tf.zeros((1, cfg.Nx, cfg.Nx, 8)), tf.zeros((1, cfg.desc_dim)),
                         tf.zeros((1, 4)), training=False)
print("nvars after forward:", len(model.trainable_variables), flush=True)
_, test = build_poseidon_stream(cfg, cfg.batch)
xt = np.asarray(test["traj"], np.float32); cm = np.asarray(test["cm"], np.float32)
mean = np.asarray(test["mean"], np.float32); std = np.asarray(test["std"], np.float32)
N, NT = xt.shape[0], xt.shape[1]

def predict(u0):
    o = []
    for i in range(0, len(u0), 8):
        b = tf.constant(u0[i:i+8])
        o.append(tf.cast(model.call_with_gate(b, tf.zeros((len(b), cfg.desc_dim)),
                                              tf.zeros((len(b), 4)), training=False)[0],
                         tf.float32).numpy())
    return np.concatenate(o)[:, :NT]

pred = predict(xt[:, 0])
sc = std[:, None, None, None, :]; mc = mean[:, None, None, None, :]; m5 = cm[:, None, None, None, :]
pT = pred[:, 1:]*sc + mc; rT = xt[:, 1:]*sc + mc
num = np.sum(np.abs(pT-rT)*m5, axis=(2, 3, 4)); den = np.sum(np.abs(rT)*m5, axis=(2, 3, 4))+1e-8
ft = (num/den).mean(axis=1)
print(f"[VERIFY Poisson v2] full-traj SOL median={np.median(ft)*100:.3f}%", flush=True)
np.savez_compressed(out, pred16=pred[:16], gt16=xt[:16], mean=mean, std=std, cm=cm,
                    ft_sol=ft.astype(np.float32), fin_sol=(num/den)[:, -1].astype(np.float32))
print("DUMP_DONE", out, flush=True)
