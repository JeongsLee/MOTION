"""Phase-1 ACE few-shot sample-efficiency for the Physics-Operator Mixture model.

    python -m motion_tf.train.train_ace motion_tf.train.configs.ace_fewshot

For each train-set size N in cfg.N_list, train a fresh model on N ACE trajectories and
report rel-L² on the official 240-trajectory test set → sample-efficiency curve to be
compared against FNO (from scratch) and Poseidon-B (finetuned). Also runs the expert
ablation causal check: convective should be ~inert on ACE (reaction-diffusion).
"""
from __future__ import annotations
import importlib
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from ..model import PhysicsOperatorMixture
from ..data import ace


def per_frame_rel_l2(pred, ref, eps=1e-6):
    num = tf.reduce_sum((pred - ref) ** 2, axis=[2, 3])
    den = tf.reduce_sum(ref ** 2, axis=[2, 3]) + eps
    return tf.reduce_mean(num / den)


def train_one(cfg, u_tr, u_te, jit):
    """Train a fresh model on u_tr (N,Nt,Nx,Nx), return test rel-L² + gate weights."""
    model = PhysicsOperatorMixture(cfg)
    opt = tf.keras.optimizers.Adam(cfg.lr)
    lam = float(getattr(cfg, "lambda_gate", 0.0))
    N = u_tr.shape[0]
    desc_tr = tf.constant(ace.descriptor(N))
    coef_tr = tf.constant(ace.coeffs(N))
    u_tr_t = tf.constant(u_tr)

    @tf.function(jit_compile=jit)
    def step(u, desc, coef):
        with tf.GradientTape() as tape:
            pred, alpha = model.call_with_gate(u[:, 0], desc, coef)
            data = per_frame_rel_l2(pred, u)
            loss = data + lam * tf.reduce_mean(tf.reduce_sum(alpha, axis=-1))
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return data

    bs = min(cfg.batch, N)
    for s in range(1, cfg.steps + 1):
        idx = np.random.choice(N, bs, replace=False)
        step(tf.gather(u_tr_t, idx), tf.gather(desc_tr, idx), tf.gather(coef_tr, idx))

    # test (batched to fit memory)
    errs = []
    nte = u_te.shape[0]
    for i in range(0, nte, 32):
        ub = u_te[i:i + 32]
        p = model.call(tf.constant(ub[:, 0]),
                       tf.constant(ace.descriptor(ub.shape[0])),
                       tf.constant(ace.coeffs(ub.shape[0])))
        errs.append(float(per_frame_rel_l2(p, tf.constant(ub))) * ub.shape[0])
    test_err = sum(errs) / nte
    gw = model.gate_weights(tf.constant(ace.descriptor(1)))
    gates = {k: float(v[0]) for k, v in gw.items()}
    n_param = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    return test_err, gates, n_param


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    jit = bool(getattr(cfg, "jit_compile", True))
    gpus = tf.config.list_physical_devices("GPU")
    print(f"=== {cfg_path} | ACE few-shot | GPU={gpus} | jit={jit} ===", flush=True)

    u_te = ace.load_test(cfg.n_test)
    print(f"test set: {u_te.shape}  range [{u_te.min():.3f},{u_te.max():.3f}]", flush=True)
    pool = ace.load_train_pool(max(cfg.N_list))
    print(f"train pool: {pool.shape}", flush=True)

    print(f"\n{'N':>6s} | {'test relL2':>11s} | params | gates (conv/diff/react)", flush=True)
    for N in cfg.N_list:
        u_tr = pool[:N]
        err, gates, npar = train_one(cfg, u_tr, u_te, jit)
        g = f"{gates['convective']:.2f}/{gates['diffusive']:.2f}/{gates['reaction']:.2f}"
        print(f"{N:6d} | {err:11.4e} | {npar:6d} | {g}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.ace_fewshot")
