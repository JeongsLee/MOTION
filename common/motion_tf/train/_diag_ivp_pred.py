"""Diagnostic: load the current Phase-2 IVP checkpoint and, per family, predict the final frame from the IC
and report per-channel pred-std vs gt-std + rel-L1. Tells whether a stuck family (NS-Sines) is being
SMOOTHED (pred std << gt std → MSE-blurring / not learning structure) or mis-predicted some other way.
Run: python -u -m motion_tf.train._diag_ivp_pred <config>
"""
from __future__ import annotations
import glob, importlib, os, sys
import numpy as np
import tensorflow as tf
from ..model import PhysicsOperatorMixture
from ..data import poseidon
from ..utils import ckpt as ckptlib

CH = {0: "vx", 1: "vy", 2: "tracer", 3: "rho", 4: "p", 5: "energy"}


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    if bool(getattr(cfg, "bf16", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    _, test = poseidon.build_poseidon_stream(cfg, cfg.batch)
    xt, cmt, mt, st, famt, names = (test["traj"], test["cm"], test["mean"], test["std"],
                                    test["fam"], test["fam_names"])
    model = PhysicsOperatorMixture(cfg)
    _ = model.call_with_gate(xt[:1, 0], tf.zeros((1, cfg.desc_dim)), tf.zeros((1, 4)))
    ck = os.environ.get("CKPT")
    if not ck:
        cks = sorted(glob.glob(os.path.join("/code-vol", cfg.save_dir, "ckpt_*.npz")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        ck = cks[-1]
    ckptlib.load(model, ck); print(f"restored {ck}", flush=True)
    Nt = cfg.Nt

    for fi, nm in enumerate(names):
        idx = np.where(famt == fi)[0][:32]
        if len(idx) == 0:
            continue
        u0 = xt[idx, 0].astype(np.float32)
        pred = tf.cast(model.call_with_gate(u0, tf.zeros((len(idx), cfg.desc_dim)),
                                            tf.zeros((len(idx), 4)))[0], tf.float32).numpy()
        sc = st[idx][:, None, None, :]; mc = mt[idx][:, None, None, :]
        pf = pred[:, Nt - 1] * sc + mc                              # physical final pred
        rf = xt[idx, Nt - 1] * sc + mc                             # physical final gt
        active = np.where(cmt[idx][0] > 0)[0]
        parts = []
        for c in active:
            g = rf[..., c]; p = pf[..., c]
            rel = np.sqrt(((g - p) ** 2).sum()) / (np.sqrt((g ** 2).sum()) + 1e-9)
            parts.append(f"{CH.get(c, c)}: gtstd={g.std():.3f} predstd={p.std():.3f} relL1n/a relL2={rel*100:.1f}%")
        # joint rel-L1 (eval metric)
        m = cmt[idx][:, None, None, :]
        num = np.sum(np.abs(pf - rf) * m, axis=(1, 2, 3)); den = np.sum(np.abs(rf) * m, axis=(1, 2, 3)) + 1e-9
        print(f"\n[{nm}] joint median rel-L1@final = {np.median(num/den)*100:.2f}%", flush=True)
        for s in parts:
            print(f"    {s}", flush=True)
        if os.environ.get("DUMP_ARR"):                              # save physical final-frame pred+gt for plots
            od = os.environ.get("DUMP_DIR", "/code-vol/results/phase2_vel"); os.makedirs(od, exist_ok=True)
            np.savez_compressed(os.path.join(od, f"{nm}.npz"),
                                pred=pf[:6].astype(np.float32), gt=rf[:6].astype(np.float32),
                                cmask=cmt[idx][:6].astype(np.float32))
            print(f"    [saved {nm} arrays]", flush=True)
    print("DIAG_IVP_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.poseidon_pretrain_ivp")
