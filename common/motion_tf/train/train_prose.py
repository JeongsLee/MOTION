"""PROSE-FD training (Section A: conditioned, T_in frames → predict future window).

    python -m motion_tf.train.train_prose motion_tf.train.configs.prose_swe

Window = t_num frames. Input = first T_in frames; target = frames [T_in-1 : T_in-1+Nt]
(frame T_in-1 is the last input = hard-IC anchor, then Nt-1 future frames). Loss = c_mask-
weighted relative L² (only active physical slots). Reports time-avg rel-L² → compare to the
PROSE-FD table per dataset.
"""
from __future__ import annotations
import importlib
import sys
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

import glob
import json
from ..model import PhysicsOperatorMixture
from ..data import prose
from ..utils import ckpt as ckptlib


def masked_rel_l2(pred, ref, c_mask, eps=1e-6):
    m = c_mask[None, None, None, None, :]                          # (1,1,1,1,6)
    num = tf.reduce_sum(((pred - ref) ** 2) * m, axis=[2, 3, 4])
    den = tf.reduce_sum((ref ** 2) * m, axis=[2, 3, 4]) + eps
    return tf.reduce_mean(num / den)


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    jit = bool(getattr(cfg, "jit_compile", True))
    print(f"=== PROSE-FD {cfg.dataset} | T_in={cfg.T_in} Nt={cfg.Nt} | "
          f"GPU={tf.config.list_physical_devices('GPU')} jit={jit} ===", flush=True)

    d = prose.build(cfg.dataset, cfg.n_total, cfg.t_num, cfg.t_step, seed=cfg.seed)
    u_all = d["u"]; c_mask = tf.constant(d["c_mask"]); desc_vec = d["desc"]
    print(f"data u {u_all.shape}  c_mask {d['c_mask']}  desc {d['desc']}", flush=True)

    u_tr = u_all[d["tr"]]; u_va = u_all[d["va"]]; u_te = u_all[d["te"]]
    Ti, Nt = cfg.T_in, cfg.Nt

    def windows(u):
        u_in = u[:, :Ti]                         # (n,T_in,128,128,6)
        target = u[:, Ti - 1: Ti - 1 + Nt]       # (n,Nt,128,128,6); frame[0]=last input (IC)
        return u_in, target
    uin_tr, tgt_tr = windows(u_tr)
    uin_va, tgt_va = windows(u_va)
    uin_te, tgt_te = windows(u_te)

    N = uin_tr.shape[0]
    desc_tr = np.tile(desc_vec, (N, 1)).astype(np.float32)
    coef0 = np.zeros((N, 4), np.float32)
    uin_t = tf.constant(uin_tr); tgt_t = tf.constant(tgt_tr)
    desc_t = tf.constant(desc_tr); coef_t = tf.constant(coef0)

    model = PhysicsOperatorMixture(cfg)
    npar = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"model params: {npar:,}", flush=True)
    opt = tf.keras.optimizers.Adam(cfg.lr)
    lam = float(cfg.lambda_gate)

    @tf.function(jit_compile=jit)
    def step(uin, tgt, dsc, cf):
        with tf.GradientTape() as tape:
            pred, alpha = model.call_with_gate(uin, dsc, cf)
            loss = masked_rel_l2(pred, tgt, c_mask) + lam * tf.reduce_mean(tf.reduce_sum(alpha, -1))
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    # ---- resume from latest ckpt if present (VESSL preemption / continuation) ----
    save_dir = getattr(cfg, "save_dir", None)
    ckpt_every = int(getattr(cfg, "ckpt_every", 0))
    start = 1
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cks = sorted(glob.glob(os.path.join(save_dir, "ckpt_*.npz")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        if cks:
            ckptlib.load(model, cks[-1]); start = int(cks[-1].split("_")[-1].split(".")[0]) + 1
            print(f"  resumed from {cks[-1]} @ step {start}", flush=True)

    for s in range(start, cfg.steps + 1):
        idx = np.random.choice(N, cfg.batch, replace=False)
        loss = step(tf.gather(uin_t, idx), tf.gather(tgt_t, idx),
                    tf.gather(desc_t, idx), tf.gather(coef_t, idx))
        if s % 50 == 0 or s == 1:
            print(f"  step {s:4d} train={float(loss):.4e}", flush=True)
        if save_dir and ckpt_every and s % ckpt_every == 0:
            ckptlib.save(model, os.path.join(save_dir, f"ckpt_{s}.npz"))
            json.dump({"step": s, "loss": float(loss)},
                      open(os.path.join(save_dir, "state.json"), "w"))
            print(f"  [ckpt] saved step {s}", flush=True)
    if save_dir:
        ckptlib.save(model, os.path.join(save_dir, f"ckpt_{cfg.steps}.npz"))

    # test (time-avg rel-L²)
    nte = uin_te.shape[0]
    errs = []
    for i in range(0, nte, 8):
        p = model.call(tf.constant(uin_te[i:i+8]),
                       tf.constant(np.tile(desc_vec, (min(8, nte-i), 1)).astype(np.float32)),
                       tf.constant(np.zeros((min(8, nte-i), 4), np.float32)))
        errs.append(float(masked_rel_l2(p, tf.constant(tgt_te[i:i+8]), c_mask)) * min(8, nte-i))
    print(f"\n=== {cfg.dataset} test rel-L² = {sum(errs)/nte:.4e}  (params {npar:,}) ===", flush=True)
    gw = {k: float(v[0]) for k, v in model.gate_weights(tf.constant(desc_vec[None])).items()}
    print(f"gate: {gw}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.prose_swe")
