"""Hard-physics multi-family foundation training (heat·ADR·NS·Euler, one model).

    python -m motion_tf.train.train_multi_hard motion_tf.train.configs.multi_hard

ONE model over the full operator basis (convective/diffusive/reaction/elliptic/shock), common
4-channel layout with per-family masks. Reports per-family masked rel-L² and the headline
[family × expert] gate fingerprint — does the learned gate route each family to the right
operator(s): heat→diffusive, ADR→conv+diff+react, NS→elliptic, Euler→shock?
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
from ..data import multifamily_hard as mfh


def masked_rel_l2(pred, ref, mask, eps=1e-6):
    m = mask[:, None, None, None, :]
    num = tf.reduce_sum(((pred - ref) ** 2) * m, axis=[2, 3, 4])
    den = tf.reduce_sum((ref ** 2) * m, axis=[2, 3, 4]) + eps
    return tf.reduce_mean(num / den)


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    jit = bool(getattr(cfg, "jit_compile", True))
    print(f"=== hard multi-family foundation | experts={cfg.experts} | GPU={tf.config.list_physical_devices('GPU')} ===", flush=True)

    tr = mfh.build(cfg.n_per_train, Nx=cfg.Nx, Nt=cfg.Nt, seed=cfg.seed)
    te = mfh.build(cfg.n_per_test, Nx=cfg.Nx, Nt=cfg.Nt, seed=cfg.seed + 5000)
    print(f"train u {tr['u'].shape}  families={tr['families']}", flush=True)

    u = tf.constant(tr["u"]); desc = tf.constant(tr["desc"])
    coef = tf.constant(tr["coeffs"]); mask = tf.constant(tr["mask"])
    metr = tf.constant(tr["metric"])
    N = u.shape[0]

    model = PhysicsOperatorMixture(cfg)
    npar = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"model params: {npar:,}", flush=True)
    opt = tf.keras.optimizers.Adam(cfg.lr)
    lam = float(cfg.lambda_gate)

    @tf.function(jit_compile=jit)
    def step(ub, db, cb, mb, mt):
        with tf.GradientTape() as tape:
            pred, alpha = model.call_with_gate(ub[:, 0], db, cb, mt)
            loss = masked_rel_l2(pred, ub, mb) + lam * tf.reduce_mean(tf.reduce_sum(alpha, -1))
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    for s in range(1, cfg.steps + 1):
        idx = np.random.choice(N, cfg.batch, replace=False)
        loss = step(tf.gather(u, idx), tf.gather(desc, idx), tf.gather(coef, idx),
                    tf.gather(mask, idx), tf.gather(metr, idx))
        if s % 500 == 0 or s == 1:
            print(f"  step {s:4d} train={float(loss):.4e}", flush=True)

    # ---- per-family masked test error ----
    print("\n=== per-family test rel-L² ===", flush=True)
    fid = te["fam_idx"]
    for fi, fname in enumerate(te["families"]):
        sel = fid == fi
        ub = te["u"][sel]; db = te["desc"][sel]; cb = te["coeffs"][sel]; mb = te["mask"][sel]
        mt = te["metric"][sel]
        errs = []
        for i in range(0, ub.shape[0], 16):
            p = model.call(tf.constant(ub[i:i+16, 0]), tf.constant(db[i:i+16]),
                           tf.constant(cb[i:i+16]), tf.constant(mt[i:i+16]))
            errs.append(float(masked_rel_l2(p, tf.constant(ub[i:i+16]), tf.constant(mb[i:i+16]))) * min(16, ub.shape[0]-i))
        print(f"  {fname:>6s}: {sum(errs)/ub.shape[0]:.4e}", flush=True)

    # ---- [family × expert] gate fingerprint ----
    print("\n=== gate fingerprint [family × expert] (mean α_k) ===", flush=True)
    names = list(cfg.experts)
    print(f"{'family':>6s} | " + "  ".join(f"{k:>10s}" for k in names), flush=True)
    for fi, fname in enumerate(te["families"]):
        sel = fid == fi
        gw = model.gate_weights(tf.constant(te["desc"][sel]))
        row = "  ".join(f"{float(tf.reduce_mean(gw[k])):10.3f}" for k in names)
        print(f"{fname:>6s} | {row}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.multi_hard")
