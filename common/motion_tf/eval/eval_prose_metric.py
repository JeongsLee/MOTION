"""Evaluate a checkpoint with PROSE-FD's EXACT metric + horizon, for a fair head-to-head.

PROSE metric = time-averaged relative L2 NORM: (1/T) Σ_{t=1..T} ||u_t - ũ_t|| / (||u_t|| + 1e-7),
averaged over the T OUTPUT frames (frame 0 is the hard-IC, excluded). PROSE uses T=10 output, except
PDEArena NS which uses **T=4**. We report BOTH T=4 and T=10 per family so the PDEArena number is
comparable to PROSE's 10.76% (NS-cond) / 6.34% (NS).

    python -m motion_tf.eval.eval_prose_metric motion_tf.train.configs.prose_multi_mid
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


def per_frame_rel_l2(pred, tgt, cm, eps=1e-7, mean_off=None):
    """(B,Nt,H,W,C),(B,C) → (B,Nt) per-frame relative L2 NORM over active (masked) channels.
    mean_off:(B,C) — if given, the denominator uses (tgt - mean_off) instead of tgt. This yields PROSE's
    NORMALIZED-space metric: PROSE normalizes u by per-trajectory input-window (mean,std); the std cancels
    in the ratio, leaving rel-L2_norm = ||u-û|| / ||u-mean||. So we get PROSE-comparable numbers from a
    PHYSICALLY-trained model just by subtracting the input-window mean in the denominator (no retrain)."""
    m = cm[:, None, None, None, :]
    num = tf.sqrt(tf.reduce_sum(((pred - tgt) ** 2) * m, axis=[2, 3, 4]))
    ref = tgt if mean_off is None else (tgt - mean_off[:, None, None, None, :])
    den = tf.sqrt(tf.reduce_sum((ref ** 2) * m, axis=[2, 3, 4])) + eps
    return num / den


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    cache_dir = os.environ.get("PREBUILT_DIR") or os.path.join(
        os.environ.get("PROSE_DATA_DIR", "/code-vol/data/prose"), "prebuilt")
    _, test = streammod.make_train_stream(cfg, 8, cache_dir)
    model = PhysicsOperatorMixture(cfg)
    _step = os.environ.get("EVAL_CKPT_STEP")                  # pin a specific ckpt (e.g. r1's 40000) so a
    if _step:                                                 # concurrent run writing newer ckpts to the same
        ck = os.path.join("/code-vol", cfg.save_dir, f"ckpt_{_step}.npz")  # save_dir doesn't change "latest"
        if not os.path.exists(ck):
            print("no ckpt", ck); return
    else:
        cks = sorted(glob.glob(os.path.join("/code-vol", cfg.save_dir, "ckpt_*.npz")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        if not cks:
            print("no ckpt under", cfg.save_dir); return
        ck = cks[-1]
    ckptlib.load(model, ck)
    print(f"loaded {ck}  | Nt={cfg.Nt} (frame0=IC, future=1..{cfg.Nt-1})", flush=True)

    @tf.function
    def ev(u0, dsc, cf):
        pred, _ = model.call_with_gate(u0, dsc, cf)
        return tf.cast(pred, tf.float32)

    xin, xtg, cm, dsc, cf, fam = (test["xin"], test["xtg"], test["cm"], test["desc"],
                                  test["coef"], test["fam"])
    # per-trajectory input-window mean per channel (PROSE's normalization offset) → PROSE-space denom
    m_in = xin.mean(axis=(1, 2, 3)).astype(np.float32)        # (N, C)
    EB = 16
    pf_phys, pf_norm = [], []
    for i in range(0, len(xin), EB):
        p = ev(xin[i:i + EB], dsc[i:i + EB], cf[i:i + EB])
        pf_phys.append(per_frame_rel_l2(p, xtg[i:i + EB], cm[i:i + EB]).numpy())
        pf_norm.append(per_frame_rel_l2(p, xtg[i:i + EB], cm[i:i + EB], mean_off=m_in[i:i + EB]).numpy())
    pf_phys = np.concatenate(pf_phys); pf_norm = np.concatenate(pf_norm)   # (N, Nt)
    # frame index where the FUTURE starts: ic_first → after the T_in input-reconstruction frames; else
    # frame 1 (frame 0 = hard-IC). The T-horizon averages run from there.
    _f0 = int(cfg.T_in) if bool(getattr(cfg, "ic_first", False)) else 1
    n_future = pf_phys.shape[1] - _f0

    # --- COLLAPSE DIAGNOSTIC: did the model regress to the per-trajectory mean? ---
    # (a) spatial-std ratio std(pred)/std(tgt) per future frame (→0 = collapsed to a flat/mean field;
    #     ≈1 = amplitude preserved).  (b) per-frame normalized error (→100% late = losing dynamics).
    preds = []
    for i in range(0, len(xin), EB):
        preds.append(ev(xin[i:i + EB], dsc[i:i + EB], cf[i:i + EB]).numpy())
    P = np.concatenate(preds)                                # (N,Nt,H,W,C)
    Tg = xtg                                                 # (N,Nt,H,W,C)
    cmb = cm[:, None, None, None, :]
    sp_p = np.sqrt((( (P - P.mean(axis=(2, 3), keepdims=True))**2) * cmb).sum(axis=(2, 3, 4)))   # (N,Nt)
    sp_t = np.sqrt((((Tg - Tg.mean(axis=(2, 3), keepdims=True))**2) * cmb).sum(axis=(2, 3, 4)))
    ratio = sp_p / (sp_t + 1e-7)                             # (N,Nt) pred-amplitude / tgt-amplitude
    print("=== COLLAPSE DIAGNOSTIC (per family) ===", flush=True)
    for fi, nm in enumerate(test["fam_names"]):
        msk = fam == fi
        if not msk.any():
            continue
        Tl = min(10, n_future)
        r = ratio[msk][:, _f0:_f0 + Tl].mean()               # avg std-ratio over future
        e1 = 100*pf_norm[msk][:, _f0].mean(); e_last = 100*pf_norm[msk][:, _f0 + Tl - 1].mean()
        print(f"  {nm:16s} std(pred)/std(tgt)={r:.3f}  | norm-err f1={e1:.1f}% f{Tl}={e_last:.1f}%",
              flush=True)
    for space, pf in [("NORMALIZED (PROSE-comparable: den=||u-mean||)", pf_norm),
                      ("physical (den=||u||)", pf_phys)]:
        for T in [4, 10]:
            if T > n_future:
                continue
            score = pf[:, _f0:_f0 + T].mean(axis=1)          # time-avg over the first T future frames
            print(f"=== {space} | rel-L2, horizon T={T} ===", flush=True)
            for fi, nm in enumerate(test["fam_names"]):
                msk = fam == fi
                if msk.any():
                    print(f"  {nm:16s} {100*score[msk].mean():.3f}%  (n={int(msk.sum())})", flush=True)
            print(f"  overall          {100*score.mean():.3f}%", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "motion_tf.train.configs.prose_multi_mid")
