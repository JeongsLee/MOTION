"""Multi-family training for the Physics-Operator Mixture foundation operator.

    python -m motion_tf.train.train_multi motion_tf.train.configs.multi_family

Pretrains one shared model across several PDE families (Heat, Advection, ReacDiff, ADR) and
reports the [family × expert] gate fingerprint — the headline interpretability result: the
family gate should recover each family's operator composition (DESIGN.md §7).
"""
from __future__ import annotations
import importlib
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from ..model import PhysicsOperatorMixture
from ..data import families as fam


def per_frame_rel_l2(pred, ref, eps=1e-6):
    num = tf.reduce_sum((pred - ref) ** 2, axis=[2, 3])
    den = tf.reduce_sum(ref ** 2, axis=[2, 3]) + eps
    return tf.reduce_mean(num / den)


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    gpus = tf.config.list_physical_devices("GPU")
    jit = bool(getattr(cfg, "jit_compile", True))
    print(f"=== {cfg_path} | families={cfg.families} | experts={cfg.experts} ===", flush=True)
    print(f"devices: GPU={gpus} | jit_compile(XLA)={jit}", flush=True)

    tr = fam.generate_multi(cfg.families, cfg.n_per_train, Nx=cfg.Nx, Nt=cfg.Nt,
                            T=cfg.T_final, seed=cfg.seed)
    va = fam.generate_multi(cfg.families, cfg.n_per_val, Nx=cfg.Nx, Nt=cfg.Nt,
                            T=cfg.T_final, seed=cfg.seed + 7777)
    u_tr = tf.constant(tr["u"]); c_tr = tf.constant(tr["coeffs"])
    d_tr = tf.constant(fam.make_descriptor(tr["coeffs"]))
    u_va = tf.constant(va["u"]); c_va = tf.constant(va["coeffs"])
    d_va = tf.constant(fam.make_descriptor(va["coeffs"]))
    print(f"data: train {u_tr.shape} ({len(cfg.families)} families)", flush=True)

    model = PhysicsOperatorMixture(cfg)
    n_param = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"model params: {n_param:,}", flush=True)
    opt = tf.keras.optimizers.Adam(cfg.lr)
    lam = float(getattr(cfg, "lambda_gate", 0.0))

    @tf.function(jit_compile=jit)
    def train_step(u, desc, coeffs):
        with tf.GradientTape() as tape:
            pred, alpha = model.call_with_gate(u[:, 0], desc, coeffs)
            data = per_frame_rel_l2(pred, u)
            loss = data + lam * tf.reduce_mean(tf.reduce_sum(alpha, axis=-1))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return data

    n = u_tr.shape[0]
    for step in range(1, cfg.steps + 1):
        idx = np.random.choice(n, cfg.batch, replace=False)
        loss = train_step(tf.gather(u_tr, idx), tf.gather(d_tr, idx), tf.gather(c_tr, idx))
        if step % 50 == 0 or step == 1:
            vpred = model.call(u_va[:, 0], d_va, c_va)
            vloss = per_frame_rel_l2(vpred, u_va)
            print(f"step {step:4d}  train={float(loss):.4e}  val_relL2={float(vloss):.4e}",
                  flush=True)

    # ---- per-family val rel-L² ----
    print("\n=== per-family val rel-L² ===", flush=True)
    fidx = va["fam_idx"]
    for fi, fname in enumerate(cfg.families):
        m = fidx == fi
        p = model.call(tf.constant(va["u"][m][:, 0]),
                       tf.constant(fam.make_descriptor(va["coeffs"][m])),
                       tf.constant(va["coeffs"][m]))
        e = float(per_frame_rel_l2(p, tf.constant(va["u"][m])))
        print(f"  {fname:>10s}: {e:.4e}", flush=True)

    # ---- [family × expert] gate fingerprint (mean α_k over each family) ----
    print("\n=== gate fingerprint  [family × expert]  (mean α_k) ===", flush=True)
    gw = model.gate_weights(d_va)
    names = list(gw.keys())
    print(f"{'family':>10s} | " + "  ".join(f"{k:>10s}" for k in names), flush=True)
    for fi, fname in enumerate(cfg.families):
        m = fidx == fi
        row = "  ".join(f"{float(tf.reduce_mean(tf.boolean_mask(gw[k], m))):10.3f}"
                        for k in names)
        print(f"{fname:>10s} | {row}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.multi_family")
