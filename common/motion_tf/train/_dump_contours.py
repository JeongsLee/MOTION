"""Dump physical-space prediction vs ground-truth tensors for a few test samples per family, using a
trained PhysicsOperatorMixture checkpoint. Mirrors train_prose_mgpu's model build / stream test set /
denormalization exactly, so the dumped fields are in the same physical space as the eval metric.
Run:  python -u -m motion_tf.train._dump_contours <config_module>   (CKPT/ env override optional)
Saves /code-vol/results/phase1task1_contours/<family>.npz  (pred, gt, cmask, mean, std).
"""
from __future__ import annotations
import importlib, os, sys
import numpy as np
import tensorflow as tf

from ..model import PhysicsOperatorMixture
from ..data import stream
from ..utils import ckpt as ckptlib

K = int(os.environ.get("K_SAMPLES", "3"))           # samples dumped per family


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    if bool(getattr(cfg, "bf16", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("mixed_bfloat16 policy enabled", flush=True)
    cache_dir = getattr(cfg, "cache_dir", None) or os.environ.get("PREBUILT_DIR")
    Ti, Nt = cfg.T_in, cfg.Nt

    # --- test set materialized exactly like training ---
    _, test = stream.make_train_stream(cfg, cfg.batch, cache_dir)
    xin, xtg = test["xin"], test["xtg"]
    cm, desc, coef, fam = test["cm"], test["desc"], test["coef"], test["fam"]
    mean, std = test["mean"], test["std"]
    fam_names = test["fam_names"]
    print(f"test materialized: {xin.shape}  families={fam_names}", flush=True)

    # --- build model, instantiate vars (warmup), restore checkpoint ---
    model = PhysicsOperatorMixture(cfg)
    _ = model.call_with_gate(xin[:1].astype(np.float32), desc[:1].astype(np.float32),
                             coef[:1].astype(np.float32))           # build all variables
    ck = os.environ.get("CKPT") or os.path.join("/code-vol", cfg.save_dir, "ckpt_%d.npz" % cfg.steps)
    if not os.path.exists(ck):
        # fall back to latest ckpt in the dir
        import glob
        cks = sorted(glob.glob(os.path.join("/code-vol", cfg.save_dir, "ckpt_*.npz")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        ck = cks[-1]
    ckptlib.load(model, ck)
    print(f"restored {ck}", flush=True)

    out_dir = os.environ.get("DUMP_DIR", "/code-vol/results/phase1task1_contours")
    os.makedirs(out_dir, exist_ok=True)
    for fi, nm in enumerate(fam_names):
        idx = np.where(fam == fi)[0][:K]
        if len(idx) == 0:
            print(f"  [{nm}] no test samples", flush=True); continue
        u0 = xin[idx].astype(np.float32)
        ref = xtg[idx].astype(np.float32)
        d_, c_ = desc[idx].astype(np.float32), coef[idx].astype(np.float32)
        pred, _ = model.call_with_gate(u0, d_, c_)
        pred = tf.cast(pred, tf.float32).numpy()
        sc = std[idx][:, None, None, None, :] + 1e-6                 # denormalize -> physical
        mc = mean[idx][:, None, None, None, :]
        pred_ph = pred * sc + mc
        gt_ph = ref * sc + mc
        np.savez_compressed(os.path.join(out_dir, f"{nm}.npz"),
                            pred=pred_ph.astype(np.float32), gt=gt_ph.astype(np.float32),
                            cmask=cm[idx].astype(np.float32), Ti=Ti, Nt=Nt, family=nm)
        # quick per-sample physical rel-L2 sanity (over real channels, last frame)
        m = cm[idx][:, None, None, None, :]
        num = np.sqrt((((pred_ph - gt_ph) ** 2) * m).sum(axis=(1, 2, 3, 4)))
        den = np.sqrt(((gt_ph ** 2) * m).sum(axis=(1, 2, 3, 4))) + 1e-6
        print(f"  [{nm}] dumped {len(idx)} | rel-L2(full-rollout) = "
              f"{', '.join(f'{v*100:.2f}%' for v in (num/den))}", flush=True)
    print("DUMP_CONTOURS_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.prose_150M_fluid5_6x")
