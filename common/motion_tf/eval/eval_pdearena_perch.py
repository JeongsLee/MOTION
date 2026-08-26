"""Per-channel rel-L2 breakdown for pdearena_ns (Vx, Vy, scalar) on a Phase-1 ckpt — to find which channel
drives the family's high error. Same data/model/metric as eval_prose_exact, single-shot; the only change is
the channel mask is restricted to one slot at a time (joint = the family headline; per-slot = the breakdown).
Run: EVAL_CKPT_PATH=... EVAL_CKPT_ROOT=/eu python -m motion_tf.eval.eval_pdearena_perch <cfg>
"""
from __future__ import annotations
import glob, importlib, os, sys
import numpy as np
import tensorflow as tf
from .. import data as _d
from ..data import stream as streammod
from ..model import PhysicsOperatorMixture
from ..utils import ckpt as ckptlib
from .eval_prose_exact import per_frame_norm_rel_l2


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    if bool(getattr(cfg, "bf16", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    cache_dir = os.environ.get("PREBUILT_DIR") or os.path.join(
        os.environ.get("PROSE_DATA_DIR", "/code-vol/data/prose"), "prebuilt")
    _, test = streammod.make_train_stream(cfg, 8, cache_dir)
    model = PhysicsOperatorMixture(cfg)
    ck = os.environ.get("EVAL_CKPT_PATH") or os.path.join(
        os.environ.get("EVAL_CKPT_ROOT", "/code-vol"), cfg.save_dir,
        f"ckpt_{os.environ.get('EVAL_CKPT_STEP','')}.npz")
    if not os.path.exists(ck):
        cks = sorted(glob.glob(os.path.join(os.environ.get("EVAL_CKPT_ROOT","/code-vol"), cfg.save_dir, "ckpt_*.npz")),
                     key=lambda p:int(p.split("_")[-1].split(".")[0]))
        ck = cks[-1] if cks else ck
    ckptlib.load(model, ck); print(f"loaded {ck}", flush=True)

    @tf.function
    def ev(u0, dsc, cf):
        pred, _ = model.call_with_gate(u0, dsc, cf); return tf.cast(pred, tf.float32)

    xin, xtg, cm, dsc, cf, fam = (test["xin"], test["xtg"], test["cm"], test["desc"], test["coef"], test["fam"])
    mean, std, tm, fam_names = test["mean"], test["std"], test["tm"], test["fam_names"]
    fi = fam_names.index("pdearena_ns")
    sel = np.where(np.asarray(fam) == fi)[0]
    print(f"pdearena_ns: {len(sel)} test trajectories  (family idx {fi})", flush=True)
    C = xin.shape[-1]
    # masks: joint(0,1,2) + each slot alone
    base = cm[sel].copy()                                            # (n,C) = [1,1,1,0,..]
    masks = {"joint(u,v,s)": base.copy()}
    names = {0: "Vx (u)", 1: "Vy (v)", 2: "scalar"}
    for s, nm in names.items():
        m = np.zeros_like(base); m[:, s] = 1.0; masks[nm] = m
    acc = {k: [] for k in masks}
    EB = 16
    for j in range(0, len(sel), EB):
        idx = sel[j:j+EB]
        sc = std[idx][:, None, None, None, :] + 1e-6; mc = mean[idx][:, None, None, None, :]
        p = ev(xin[idx], dsc[idx], cf[idx]).numpy()
        p_phys = p[:, 1:] * sc + mc; t_phys = xtg[idx][:, 1:] * sc + mc
        tmf = tm[idx][:, 1:1+p_phys.shape[1]].astype(np.float32)
        for k, mfull in masks.items():
            r = per_frame_norm_rel_l2(p_phys, t_phys, mfull[j:j+EB]).numpy()   # (b,T)
            per = (r*tmf).sum(1)/(tmf.sum(1)+1e-7)                              # mean over valid frames
            acc[k].append(per)
    print("\n=== pdearena_ns per-channel rel-L2 (physical, PROSE-exact) ===", flush=True)
    for k in masks:
        v = np.concatenate(acc[k]); print(f"  {k:14s} = {100*v.mean():.3f}%", flush=True)
    print("PERCH_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.phase1_eff_ada")
