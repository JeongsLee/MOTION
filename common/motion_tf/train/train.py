"""Unified training loop for the Physics-Operator Mixture foundation operator.

    python -m motion_tf.train.train motion_tf.train.configs.adr_smoke

v1 supervises a per-frame normalised L² loss (Eq. data-loss in the paper) on Track-A ADR
data. Reports the learned family-gate weights α_k at the end — the interpretability
read-out that should track the active operators (DESIGN.md §7).
"""
from __future__ import annotations
import importlib
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from ..model import PhysicsOperatorMixture
from ..data import adr


def per_frame_rel_l2(pred, ref, eps=1e-6):
    """pred, ref: (B, Nt, Nx, Nx). Mean over frames of ‖·‖²/‖ref‖²."""
    num = tf.reduce_sum((pred - ref) ** 2, axis=[2, 3])
    den = tf.reduce_sum(ref ** 2, axis=[2, 3]) + eps
    return tf.reduce_mean(num / den)


def load_cfg(dotted):
    mod = importlib.import_module(dotted)
    return mod.Cfg


def main(cfg_path):
    cfg = load_cfg(cfg_path)
    gpus = tf.config.list_physical_devices("GPU")
    jit = bool(getattr(cfg, "jit_compile", True))
    print(f"=== config {cfg_path} | experts={cfg.experts} ===", flush=True)
    print(f"devices: GPU={gpus} | jit_compile(XLA)={jit}", flush=True)

    gkw = {}
    for key in ("a_range", "nu_range", "r_range"):
        if getattr(cfg, key, None) is not None:
            gkw[key] = getattr(cfg, key)
    tr = adr.generate(n_samples=cfg.n_train, Nx=cfg.Nx, Nt=cfg.Nt,
                      T=cfg.T_final, seed=cfg.seed, **gkw)
    va = adr.generate(n_samples=cfg.n_val, Nx=cfg.Nx, Nt=cfg.Nt,
                      T=cfg.T_final, seed=cfg.seed + 999, **gkw)
    u_tr = tf.constant(tr["u"]); d_tr = tf.constant(adr.make_descriptor(tr["coeffs"]))
    c_tr = tf.constant(tr["coeffs"])
    u_va = tf.constant(va["u"]); d_va = tf.constant(adr.make_descriptor(va["coeffs"]))
    c_va = tf.constant(va["coeffs"])
    print(f"data: train u {u_tr.shape}  val u {u_va.shape}", flush=True)

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
            # gate-sparsity (L1 on α) → unused experts pushed to 0 → specialization (§6.4)
            sparsity = tf.reduce_mean(tf.reduce_sum(alpha, axis=-1))
            loss = data + lam * sparsity
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return data

    n = cfg.n_train
    for step in range(1, cfg.steps + 1):
        idx = np.random.choice(n, cfg.batch, replace=False)
        loss = train_step(tf.gather(u_tr, idx), tf.gather(d_tr, idx),
                          tf.gather(c_tr, idx))
        if step % 20 == 0 or step == 1:
            vpred = model.call(u_va[:, 0], d_va, c_va)
            vloss = per_frame_rel_l2(vpred, u_va)
            print(f"step {step:4d}  train={float(loss):.4e}  val_relL2={float(vloss):.4e}",
                  flush=True)

    # ------------------------------------------------------------------ #
    # Interpretability probe (DESIGN.md §7): does α_k recover the active
    # operator? Build descriptors for pure single-operator regimes and a
    # mixed regime, and read out the gate. Expect the matching expert to
    # dominate (advection-only → convective, etc.).
    # ------------------------------------------------------------------ #
    print("\n=== gate-vs-operator probe ===", flush=True)
    probes = {
        "advection-only": [1.0, 0.0, 0.0, 0.0],
        "diffusion-only": [0.0, 0.0, 0.02, 0.0],
        "reaction-only":  [0.0, 0.0, 0.0, 2.0],
        "adv+diff+react": [0.8, 0.0, 0.015, 1.5],
        "zero (none)":    [0.0, 0.0, 0.0, 0.0],
    }
    coeff_mat = np.array(list(probes.values()), dtype=np.float32)
    desc = tf.constant(adr.make_descriptor(coeff_mat))
    gw = model.gate_weights(desc)
    names = list(gw.keys())
    hdr = "  ".join(f"{k:>10s}" for k in names)
    print(f"{'regime':>16s} | {hdr}", flush=True)
    for i, rname in enumerate(probes):
        row = "  ".join(f"{float(gw[k][i]):10.3f}" for k in names)
        print(f"{rname:>16s} | {row}", flush=True)

    # correlation: per-val-sample α_k vs the matching coefficient magnitude.
    print("\n=== corr(α_k, |coeff_k|) over val set ===", flush=True)
    gw_va = model.gate_weights(d_va)
    cN = va["coeffs"]
    matched = {"convective": np.maximum(np.abs(cN[:, 0]), np.abs(cN[:, 1])),
               "diffusive": cN[:, 2], "reaction": cN[:, 3]}
    for k in names:
        if k in matched:
            a = np.asarray(gw_va[k]); b = matched[k]
            c = np.corrcoef(a, b)[0, 1] if a.std() > 1e-8 and b.std() > 1e-8 else float("nan")
            print(f"  {k:>10s}: corr = {c:+.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.adr_smoke")
