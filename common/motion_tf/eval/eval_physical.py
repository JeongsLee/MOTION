"""PHYSICAL-space rel-L2 = PROSE's REPORTED metric: (1/T) Σ_t ||u_t - û_t|| / (||u_t|| + eps), on
ORIGINAL-scale fields (mean INCLUDED in the denominator) → directly comparable to PROSE's published
numbers (SWE 0.28 / com_ns 1.53 / incom_ns 2.84 / NS-cond 10.76). This is NOT our strict
mean-subtracted "normalized" metric (where mean-prediction = 100%).

For instance-norm models (instance_norm=True): the model is normalized-native, so we normalize the
input by its input-window (mean,std), run, then DENORMALIZE the output back to original scale before the
metric. For physical-native models (instance_norm=False): feed original input directly.

    python -m motion_tf.eval.eval_physical motion_tf.train.configs.prose_150M_relc_dec
"""
from __future__ import annotations
import glob, importlib, os, sys
import numpy as np
import tensorflow as tf

from ..model import PhysicsOperatorMixture
from ..utils import ckpt as ckptlib
from ..data import stream as streammod


def phys_rel_l2(pred, tgt, cm, eps=1e-7):          # original scale; (B,Nt,H,W,C),(B,C) → (B,Nt)
    m = cm[:, None, None, None, :]
    num = tf.sqrt(tf.reduce_sum(((pred - tgt) ** 2) * m, axis=[2, 3, 4]))
    den = tf.sqrt(tf.reduce_sum((tgt ** 2) * m, axis=[2, 3, 4])) + eps    # ||u|| mean-INCLUDED = physical
    return (num / den).numpy()


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    inorm = bool(getattr(cfg, "instance_norm", False))
    cache_dir = os.environ.get("PREBUILT_DIR") or os.path.join(
        os.environ.get("PROSE_DATA_DIR", "/code-vol/data/prose"), "prebuilt")

    class CfgRaw(cfg):                              # materialize test at ORIGINAL scale (no instance-norm)
        instance_norm = False
    _, test = streammod.make_train_stream(CfgRaw, 8, cache_dir)

    model = PhysicsOperatorMixture(cfg)
    cks = sorted(glob.glob(os.path.join("/code-vol", cfg.save_dir, "ckpt_*.npz")),
                 key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not cks:
        print("no ckpt under", cfg.save_dir); return
    ckptlib.load(model, cks[-1])
    print(f"loaded {cks[-1]} | instance_norm(model)={inorm} | Nt={cfg.Nt}", flush=True)

    xin, xtg, cm, dsc, cf, fam = (test["xin"], test["xtg"], test["cm"], test["desc"],
                                  test["coef"], test["fam"])

    @tf.function
    def run(u0, d, c):
        return tf.cast(model.call_with_gate(u0, d, c)[0], tf.float32)

    EB = 8
    pf = []
    for i in range(0, len(xin), EB):
        a = xin[i:i + EB].astype(np.float32)        # ORIGINAL-scale input window
        if inorm:
            m = a.mean(axis=(1, 2, 3), keepdims=True)
            s = a.std(axis=(1, 2, 3), keepdims=True) + 1e-6
            pred = run((a - m) / s, dsc[i:i + EB], cf[i:i + EB]).numpy()
            pred = pred * s + m                     # denormalize → original scale
        else:
            pred = run(a, dsc[i:i + EB], cf[i:i + EB]).numpy()
        pf.append(phys_rel_l2(tf.constant(pred), tf.constant(xtg[i:i + EB].astype(np.float32)),
                              cm[i:i + EB]))
    pf = np.concatenate(pf)                          # (N, Nt)
    f0 = int(cfg.T_in) if bool(getattr(cfg, "ic_first", False)) else 1
    n_future = pf.shape[1] - f0
    print("=== PHYSICAL rel-L2 (PROSE metric ||u-u^||/||u||, original scale) ===", flush=True)
    print("    PROSE published: shallow_water 0.28 | com_ns 1.53 | incom_ns 2.84 | pdearena_ns(NS-cond) 10.76", flush=True)
    for T in [4, 10]:
        if T > n_future:
            continue
        score = pf[:, f0:f0 + T].mean(axis=1)
        print(f"--- T={T} ---", flush=True)
        for fi, nm in enumerate(test["fam_names"]):
            msk = fam == fi
            if msk.any():
                print(f"  {nm:16s} {100 * score[msk].mean():.3f}%  (n={int(msk.sum())})", flush=True)
        print(f"  overall          {100 * score.mean():.3f}%", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.prose_150M_relc_dec")
