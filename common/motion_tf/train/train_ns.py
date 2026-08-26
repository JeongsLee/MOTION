"""Quick NS Elliptic-expert validation + ablation.

    python -m motion_tf.train.train_ns motion_tf.train.configs.ns_quick

Trains our model on NS vorticity (narrow-ν band) and runs the causal ablation: with vs without
the Elliptic expert. NS RHS needs the streamfunction coupling u=∇⊥Δ⁻¹(−ω); if the Elliptic
expert (multi-dilation g-feedback, à la NTO-ADA NS) is doing its job, removing it should hurt.
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

from ..model import PhysicsOperatorMixture
from ..data import ns


def rel_l2(pred, ref, eps=1e-6):
    num = tf.reduce_sum((pred - ref) ** 2, axis=[2, 3])
    den = tf.reduce_sum(ref ** 2, axis=[2, 3]) + eps
    return tf.reduce_mean(num / den)


def train_eval(cfg, experts, data, jit):
    cfg.experts = experts
    model = PhysicsOperatorMixture(cfg)
    npar = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    opt = tf.keras.optimizers.Adam(cfg.lr)
    lam = float(cfg.lambda_gate)

    u = tf.constant(data["u_tr"])
    desc = tf.constant(ns.make_descriptor(data["nu_tr"]))
    coef = tf.constant(ns.coeffs(data["nu_tr"].shape[0]))
    N = u.shape[0]

    @tf.function(jit_compile=jit)
    def step(ub, db, cb):
        with tf.GradientTape() as tape:
            pred, alpha = model.call_with_gate(ub[:, 0], db, cb)
            loss = rel_l2(pred, ub) + lam * tf.reduce_mean(tf.reduce_sum(alpha, -1))
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    for s in range(1, cfg.steps + 1):
        idx = np.random.choice(N, cfg.batch, replace=False)
        loss = step(tf.gather(u, idx), tf.gather(desc, idx), tf.gather(coef, idx))
        if s % 500 == 0 or s == 1:
            print(f"    step {s:4d} train={float(loss):.4e}", flush=True)

    # test
    ute = data["u_te"]
    p = model.call(tf.constant(ute[:, 0]),
                   tf.constant(ns.make_descriptor(data["nu_te"])),
                   tf.constant(ns.coeffs(ute.shape[0])))
    e = float(rel_l2(p, tf.constant(ute)))
    gw = {k: float(v[0]) for k, v in
          model.gate_weights(tf.constant(ns.make_descriptor(data["nu_te"][:1]))).items()}
    return e, npar, gw


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    jit = bool(getattr(cfg, "jit_compile", True))
    print(f"=== NS Elliptic validation | GPU={tf.config.list_physical_devices('GPU')} ===", flush=True)
    data = ns.load(cfg.n_train, cfg.n_test, cfg.Nt, getattr(cfg, "nu_range", None))
    print(f"train {data['u_tr'].shape}  test {data['u_te'].shape}\n", flush=True)

    print("[A] WITH elliptic  (convective, diffusive, elliptic)", flush=True)
    eA, pA, gA = train_eval(cfg, ("convective", "diffusive", "elliptic"), data, jit)
    print(f"  → test rel-L²={eA:.4e}  params={pA:,}  gates={gA}\n", flush=True)

    print("[B] WITHOUT elliptic (convective, diffusive)  [ablation]", flush=True)
    eB, pB, gB = train_eval(cfg, ("convective", "diffusive"), data, jit)
    print(f"  → test rel-L²={eB:.4e}  params={pB:,}  gates={gB}\n", flush=True)

    print(f"=== ablation: elliptic removed → {eB/eA:.2f}× error "
          f"({'HELPS' if eB > eA else 'no help'}) ===", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.ns_quick")
