"""Quick 2-D compressible Euler shock validation + ablation (4-channel).

    python -m motion_tf.train.train_euler motion_tf.train.configs.euler_quick

Trains our multi-channel model on Euler (ρ,ρu,ρv,E) and runs the causal ablation: WITH vs
WITHOUT the Shock expert. If the shock expert (conservation-form state-gated upwind) is
capturing discontinuities, removing it should hurt — most in shock regions.
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
from ..data import euler


def rel_l2(pred, ref, eps=1e-6):
    # pred/ref: (B,Nt,H,W,C) — reduce over space AND channels
    num = tf.reduce_sum((pred - ref) ** 2, axis=[2, 3, 4])
    den = tf.reduce_sum(ref ** 2, axis=[2, 3, 4]) + eps
    return tf.reduce_mean(num / den)


def descriptor(n):
    return np.tile(np.array([1.0, 1.0, 0.2], np.float32), (n, 1))   # shock-dominant context


def train_eval(cfg, experts, u_tr, u_te, jit):
    cfg.experts = experts
    model = PhysicsOperatorMixture(cfg)
    npar = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    opt = tf.keras.optimizers.Adam(cfg.lr)
    lam = float(cfg.lambda_gate)
    u = tf.constant(u_tr); N = u.shape[0]
    desc = tf.constant(descriptor(N)); coef = tf.constant(np.zeros((N, 4), np.float32))

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

    nte = u_te.shape[0]
    p = model.call(tf.constant(u_te[:, 0]), tf.constant(descriptor(nte)),
                   tf.constant(np.zeros((nte, 4), np.float32)))
    e = float(rel_l2(p, tf.constant(u_te)))
    gw = {k: float(v[0]) for k, v in
          model.gate_weights(tf.constant(descriptor(1))).items()}
    return e, npar, gw


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    jit = bool(getattr(cfg, "jit_compile", True))
    print(f"=== Euler shock validation | GPU={tf.config.list_physical_devices('GPU')} ===", flush=True)
    tr = euler.generate(cfg.n_train, Nx=cfg.Nx, Nt=cfg.Nt, seed=cfg.seed)
    te = euler.generate(cfg.n_test, Nx=cfg.Nx, Nt=cfg.Nt, seed=cfg.seed + 999)
    u_tr, u_te = tr["u"], te["u"]
    print(f"train {u_tr.shape}  test {u_te.shape}\n", flush=True)

    print("[A] WITH shock  (shock, diffusive)", flush=True)
    eA, pA, gA = train_eval(cfg, ("shock", "diffusive"), u_tr, u_te, jit)
    print(f"  → test rel-L²={eA:.4e}  params={pA:,}  gates={gA}\n", flush=True)

    print("[B] WITHOUT shock (convective, diffusive)  [ablation]", flush=True)
    eB, pB, gB = train_eval(cfg, ("convective", "diffusive"), u_tr, u_te, jit)
    print(f"  → test rel-L²={eB:.4e}  params={pB:,}  gates={gB}\n", flush=True)

    print(f"=== ablation: shock removed → {eB/eA:.2f}× error "
          f"({'HELPS' if eB > eA else 'no help'}) ===", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.euler_quick")
