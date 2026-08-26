"""Convergence-based ACE S=128 comparison: our model vs FNO, same task/metric/protocol.

    python -m motion_tf.train.train_ace_compare motion_tf.train.configs.ace_compare ours
    python -m motion_tf.train.train_ace_compare motion_tf.train.configs.ace_compare fno

Each run trains ONE model (separate process → no GPU OOM), early-stops on a held-out val
split, and records test rel-L² at best-val to a shared JSON. AG = FNO_test / ours_test is
then computed across the two runs (Poseidon Table-1 convention, S=128).
"""
from __future__ import annotations
import importlib
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

from ..model import PhysicsOperatorMixture
from ..baselines.fno2d import FNO2D
from ..data import ace


def per_frame_rel_l2(pred, ref, eps=1e-6):
    num = tf.reduce_sum((pred - ref) ** 2, axis=[2, 3])
    den = tf.reduce_sum(ref ** 2, axis=[2, 3]) + eps
    return tf.reduce_mean(num / den)


def eval_set(model, u, desc_fn, coef_fn, bs=32):
    errs, n = [], u.shape[0]
    for i in range(0, n, bs):
        ub = u[i:i + bs]
        p = model.call(tf.constant(ub[:, 0]),
                       tf.constant(desc_fn(ub.shape[0])),
                       tf.constant(coef_fn(ub.shape[0])))
        errs.append(float(per_frame_rel_l2(p, tf.constant(ub))) * ub.shape[0])
    return sum(errs) / n


def main(cfg_path, which):
    cfg = importlib.import_module(cfg_path).Cfg
    jit = bool(getattr(cfg, "jit_compile", True))
    print(f"=== ACE compare | model={which} | S={cfg.S} | GPU={tf.config.list_physical_devices('GPU')} ===",
          flush=True)

    pool = ace.load_train_pool(cfg.S + cfg.n_val)
    u_tr = pool[:cfg.S]
    u_val = pool[cfg.S:cfg.S + cfg.n_val]
    u_te = ace.load_test(cfg.n_test)
    print(f"train {u_tr.shape}  val {u_val.shape}  test {u_te.shape}", flush=True)

    if which == "ours":
        model = PhysicsOperatorMixture(cfg)
        lam = float(getattr(cfg, "lambda_gate", 0.0))
    else:
        model = FNO2D(cfg)
        lam = 0.0
    npar = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"params: {npar:,}", flush=True)
    opt = tf.keras.optimizers.Adam(cfg.lr)

    desc_fn = ace.descriptor if which == "ours" else (lambda n: np.zeros((n, 3), np.float32))
    coef_fn = ace.coeffs
    u_tr_t = tf.constant(u_tr)
    desc_tr = tf.constant(desc_fn(cfg.S)); coef_tr = tf.constant(coef_fn(cfg.S))

    @tf.function(jit_compile=jit)
    def step(u, desc, coef):
        with tf.GradientTape() as tape:
            if which == "ours":
                pred, alpha = model.call_with_gate(u[:, 0], desc, coef)
                loss = per_frame_rel_l2(pred, u) + lam * tf.reduce_mean(tf.reduce_sum(alpha, -1))
            else:
                pred = model.call(u[:, 0], desc, coef)
                loss = per_frame_rel_l2(pred, u)
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    best_val, test_at_best, best_step = 1e9, None, 0
    for s in range(1, cfg.max_steps + 1):
        idx = np.random.choice(cfg.S, min(cfg.batch, cfg.S), replace=False)
        step(tf.gather(u_tr_t, idx), tf.gather(desc_tr, idx), tf.gather(coef_tr, idx))
        if s % cfg.eval_every == 0:
            v = eval_set(model, u_val, desc_fn, coef_fn)
            if v < best_val:
                best_val = v
                test_at_best = eval_set(model, u_te, desc_fn, coef_fn)
                best_step = s
            print(f"  step {s:4d}  val={v:.4e}  best_val={best_val:.4e}  "
                  f"test@best={test_at_best:.4e} (step {best_step})", flush=True)

    # persist
    path = cfg.results_json
    res = {}
    if os.path.exists(path):
        res = json.load(open(path))
    res[which] = {"test_rel_l2": test_at_best, "params": npar, "best_step": best_step}
    json.dump(res, open(path, "w"), indent=2)
    print(f"\nRESULT[{which}] test={test_at_best:.4e}  params={npar:,}", flush=True)
    if "ours" in res and "fno" in res:
        ag = res["fno"]["test_rel_l2"] / res["ours"]["test_rel_l2"]
        print(f"\n=== Accuracy Gain (FNO/ours) @ S={cfg.S}: {ag:.2f}× ===", flush=True)
        print(f"  (from-scratch ceiling ~5× [scOT 5.2, CNO 4.6]; Poseidon-L pretrained 11.6×)",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ours")
