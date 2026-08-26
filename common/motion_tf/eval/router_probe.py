"""Per-family ROUTER allocation probe — for each fluid family, the mean per-expert route weight
(_last_alpha = the attn-router softmax allocation, Σ_k≈1, spatial-mean). Tests whether experts are
DIFFERENTIATED per problem (families → different dominant experts) or HOMOGENEOUS (≈uniform across
families → weak specialization). Also reports the route std ACROSS families per expert (the
differentiation signal: ~0 = no specialization).

    python -m motion_tf.eval.router_probe motion_tf.train.configs.prose_150M_fluid5_6x_lup
"""
from __future__ import annotations
import glob, importlib, os, sys
import numpy as np
import tensorflow as tf
from ..model import PhysicsOperatorMixture
from ..utils import ckpt as ckptlib
from ..data import stream as streammod


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    if bool(getattr(cfg, "bf16", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    cache = os.environ.get("PREBUILT_DIR") or os.path.join(
        os.environ.get("PROSE_DATA_DIR", "/code-vol/data/prose"), "prebuilt")
    _, test = streammod.make_train_stream(cfg, 8, cache)
    model = PhysicsOperatorMixture(cfg)
    ck = os.environ.get("CKPT")
    if not ck:
        cks = sorted(glob.glob(os.path.join("/code-vol", cfg.save_dir, "ckpt_*.npz")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        ck = cks[-1]
    ckptlib.load(model, ck)
    print(f"loaded {ck}", flush=True)

    @tf.function
    def ev(u0, dsc, cf):
        _, alpha = model.call_with_gate(u0, dsc, cf)
        return tf.cast(alpha, tf.float32)

    xin, dsc, cf, fam = test["xin"], test["desc"], test["coef"], test["fam"]
    EB = 16
    alphas = []
    for i in range(0, len(xin), EB):
        alphas.append(ev(xin[i:i + EB], dsc[i:i + EB], cf[i:i + EB]).numpy())
    alpha = np.concatenate(alphas)                       # (N,K) per-sample route
    names = list(model.expert_names)

    print("\n=== per-family MEAN route (attn-router softmax, Σ_k≈1) ===", flush=True)
    print("  family            " + "  ".join(f"{n[:9]:>9s}" for n in names), flush=True)
    fmeans = []
    for fi, ds in enumerate(cfg.datasets):
        msk = fam == fi
        if not msk.any():
            continue
        mu = alpha[msk].mean(0)
        fmeans.append(mu)
        top = names[int(np.argmax(mu))]
        print(f"  {ds:16s} " + "  ".join(f"{v:9.3f}" for v in mu) + f"   → top: {top}", flush=True)
    fmeans = np.stack(fmeans)
    print("\n  route STD across families (per expert):  "
          + "  ".join(f"{n[:9]}={s:.3f}" for n, s in zip(names, fmeans.std(0))), flush=True)
    print(f"  mean route-RANGE across families (max-min, avg over experts): "
          f"{float(np.mean(fmeans.max(0) - fmeans.min(0))):.3f}", flush=True)
    print(f"  uniform baseline would be {1.0/len(names):.3f} each; std≈0 ⇒ no per-family specialization",
          flush=True)
    print("ROUTER_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.prose_150M_fluid5_6x_lup")
