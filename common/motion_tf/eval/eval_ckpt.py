"""Evaluate a trained checkpoint → per-family test rel-L2 + gate alpha. Reliable way to get the
numbers (the live training logs bury [EVAL] under CUDA-graph warnings).

    python -m motion_tf.eval.eval_ckpt motion_tf.train.configs.prose_multi_5M_full
"""
from __future__ import annotations
import glob
import importlib
import os
import sys

import numpy as np
import tensorflow as tf

from ..model import PhysicsOperatorMixture
from ..utils import ckpt as ckptlib
from ..data import stream as streammod
from ..data.prose import SPEC


def per_sample_sq_rel_l2(pred, ref, cm, eps=1e-6):
    m = cm[:, None, None, None, :]
    num = tf.reduce_sum(((pred - ref) ** 2) * m, axis=[2, 3, 4])
    den = tf.reduce_sum((ref ** 2) * m, axis=[2, 3, 4]) + eps
    return tf.reduce_mean(num / den, axis=1)


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    cache_dir = os.environ.get("PREBUILT_DIR") or os.path.join(
        os.environ.get("PROSE_DATA_DIR", "/code-vol/data/prose"), "prebuilt")
    GLOBAL = 8
    _, test = streammod.make_train_stream(cfg, GLOBAL, cache_dir)
    model = PhysicsOperatorMixture(cfg)
    sd = cfg.save_dir
    cks = sorted(glob.glob(os.path.join("/code-vol", sd, "ckpt_*.npz")),
                 key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not cks:
        print("no ckpt under", sd); return
    ckptlib.load(model, cks[-1])
    print(f"loaded {cks[-1]}  | params {int(sum(np.prod(v.shape) for v in model.trainable_variables)):,}",
          flush=True)

    @tf.function
    def ev(u0, dsc, cf):
        pred, alpha = model.call_with_gate(u0, dsc, cf)        # training=False → no desc-dropout
        return tf.cast(pred, tf.float32), tf.cast(alpha, tf.float32)

    xin, xtg, cm, dsc, cf, fam = (test["xin"], test["xtg"], test["cm"], test["desc"],
                                  test["coef"], test["fam"])
    EB = 16
    pers, alphas = [], []
    for i in range(0, len(xin), EB):
        p, a = ev(xin[i:i + EB], dsc[i:i + EB], cf[i:i + EB])
        pers.append(per_sample_sq_rel_l2(p, xtg[i:i + EB], cm[i:i + EB]).numpy())
        alphas.append(a.numpy())
    per = np.concatenate(pers); alpha = np.concatenate(alphas)     # (N,K) per-sample gate weights
    print("=== per-family test rel-L2 ===", flush=True)
    for fi, nm in enumerate(test["fam_names"]):
        msk = fam == fi
        if msk.any():
            print(f"  {nm:16s} {100*np.sqrt(per[msk].mean()):.3f}%  (n={int(msk.sum())})", flush=True)
    print(f"  overall          {100*np.sqrt(per.mean()):.3f}%", flush=True)
    field_gate = bool(getattr(cfg, "field_gate", False)) or bool(getattr(cfg, "input_router", False))
    names = list(model.expert_names)
    if field_gate:
        # field-conditioned: α varies per-sample → report per-family MEAN ± STD (std>0 ⇒ the gate
        # responds to within-family regime variation, the whole point of field-gating).
        print("=== gate alpha per family (field-conditioned: mean ± std over samples) ===", flush=True)
        for fi, ds in enumerate(cfg.datasets):
            msk = fam == fi
            if not msk.any():
                continue
            mu = alpha[msk].mean(0); sd = alpha[msk].std(0)
            print(f"  {ds:16s} " + "  ".join(f"{k}={m:.3f}±{s:.3f}" for k, m, s in zip(names, mu, sd)),
                  flush=True)
    else:
        print("=== gate alpha per family ===", flush=True)
        for fi, ds in enumerate(cfg.datasets):
            d = np.array(SPEC[ds]["desc"], np.float32)[None]
            w = model.gate_weights(d)
            print(f"  {ds:16s} " + "  ".join(f"{k}={float(v[0]):.3f}" for k, v in w.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.prose_multi_5M_full")
