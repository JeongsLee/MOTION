"""PROSE-EXACT evaluation — replicate felix-lyx/prose's reported rel-L2 metric on OUR model, so the
numbers are directly comparable to PROSE-FD / BCAT Table-1.

Verified against felix-lyx/prose source (2026-06-16):
  * prose_fd/evaluate.py ALWAYS denormalizes BEFORE the metric:
        data_output = data_output*std + mean ;  data_label = data_label*std + mean
    (the meanvar instance-norm is only for the model input / optional loss). So the metric is in
    ORIGINAL/PHYSICAL scale — the denominator ||u|| INCLUDES the mean. We mirror this exactly with the
    per-sample input-window (mean,std) captured by data/stream.py (test["mean"], test["std"]).
  * prose_fd/utils/metrics.py "_l2_error" (batched), per sample i, frame t:
        num   = sqrt( Σ_{x,y,active-C} (u - û)^2 )            # joint L2 NORM over active channels
        scale = 1e-7 + sqrt( Σ_{x,y,active-C} u^2 )
        r_{i,t} = num/scale ;  a_i = mean_t r_{i,t}            # ARITHMETIC mean over output frames
    family number = mean_i a_i ;  AVE_BY_CLASS = mean over families.
    (NOT our training [EVAL]'s sqrt(mean(r^2)) RMS — that is biased high.)
  * Horizon: input_len=10, t_num=20 → 10 output frames for every family, EXCEPT incom_ns_arena_u
    (uncond) which evaluate.py hard-limits to 14-input_len = 4 frames. We reproduce this with
    T_fam = min(n_future, valid_t - T_in): 10 for SWE/com_ns/incom_ns/pdearena_ns/cfdbench, 4 for
    pdearena_uncond. PROSE step_1/step_5/step_10 columns are cumulative-prefix means (output[:, :k]).

    python -m motion_tf.eval.eval_prose_exact motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_r1
    EVAL_CKPT_STEP=40000  # pin a specific ckpt (else uses the latest in save_dir)
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


def per_frame_norm_rel_l2(pred, tgt, cm, eps=1e-7):
    """(B,T,H,W,C),(B,C) → (B,T) per-frame relative L2 NORM (joint over active/masked channels), PROSE form.
    NOTE: pred,tgt must already be in PHYSICAL space (denormalized)."""
    m = cm[:, None, None, None, :]
    num = tf.sqrt(tf.reduce_sum(((pred - tgt) ** 2) * m, axis=[2, 3, 4]))      # ||u-û||_2 per frame
    den = tf.sqrt(tf.reduce_sum((tgt ** 2) * m, axis=[2, 3, 4])) + eps          # ||u||_2  (mean INCLUDED)
    return num / den                                                           # (B,T) per-frame rel-L2 norm


def main(cfg_path):
    cfg = importlib.import_module(cfg_path).Cfg
    if bool(getattr(cfg, "bf16", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")           # match training build
        print("mixed_bfloat16 policy enabled", flush=True)

    cache_dir = os.environ.get("PREBUILT_DIR") or os.path.join(
        os.environ.get("PROSE_DATA_DIR", "/code-vol/data/prose"), "prebuilt")
    _, test = streammod.make_train_stream(cfg, 8, cache_dir)
    model = PhysicsOperatorMixture(cfg)

    # EVAL_CKPT_PATH pins an exact file; else look under EVAL_CKPT_ROOT/<save_dir> (root default /code-vol;
    # set EVAL_CKPT_ROOT=/eu to eval a checkpoint that lives on the capella cluster volume).
    ckpt_path = os.environ.get("EVAL_CKPT_PATH")
    root = os.environ.get("EVAL_CKPT_ROOT", "/code-vol")
    step = os.environ.get("EVAL_CKPT_STEP")
    if ckpt_path:
        ck = ckpt_path
        if not os.path.exists(ck):
            print("no ckpt", ck); return
    elif step:
        ck = os.path.join(root, cfg.save_dir, f"ckpt_{step}.npz")
        if not os.path.exists(ck):
            print("no ckpt", ck); return
    else:
        cks = sorted(glob.glob(os.path.join(root, cfg.save_dir, "ckpt_*.npz")),
                     key=lambda p: int(p.split("_")[-1].split(".")[0]))
        if not cks:
            print("no ckpt under", os.path.join(root, cfg.save_dir)); return
        ck = cks[-1]
    ckptlib.load(model, ck)
    print(f"loaded {ck}  | T_in={cfg.T_in} Nt={cfg.Nt} (frame0=IC, future=1..{cfg.Nt-1})", flush=True)

    @tf.function
    def ev(u0, dsc, cf, geom):
        pred, _ = model.call_with_gate(u0, dsc, cf, geom_mask=geom)             # geom_mask REQUIRED for cfdbench_geom
        return tf.cast(pred, tf.float32)

    xin, xtg, cm, dsc, cf, fam = (test["xin"], test["xtg"], test["cm"], test["desc"],
                                  test["coef"], test["fam"])
    mean, std = test["mean"], test["std"]                                      # per-sample per-channel (N,C)
    geom = test.get("geom")                                                    # per-sample geom_mask (cfdbench boundary / ones)
    if geom is None:
        geom = np.ones((xin.shape[0], xin.shape[2], xin.shape[3], 1), np.float32)
    tm = test["tm"]                                                            # (N,Nt) temporal validity mask
    fam_names = test["fam_names"]
    T_in = int(cfg.T_in)

    # EVAL_VEL_ONLY: restrict the channel mask to VELOCITY (slots 0,1) so the joint metric stops being
    # density-dominated (com_ns headline = density tracking; velocity is the real failure hidden behind it).
    cm_eval = cm.copy()
    if os.environ.get("EVAL_VEL_ONLY"):
        cm_eval[:, 2:] = 0.0                                                   # keep vx,vy only
        print("EVAL_VEL_ONLY: channel mask = velocity (slots 0,1) only", flush=True)
    # EVAL_AR: AUTOREGRESSIVE 1-step rollout instead of single-shot. The single-shot ADA basis degrades over
    # the horizon (com_ns velocity step1 14% → step10 25%); AR predicts ONE future per call (where the model
    # is most accurate), slides the T_in window appending the prediction, and repeats n_future times (10x
    # inference). Tests whether slow error accumulation beats the single-shot long-horizon degradation.
    AR = bool(os.environ.get("EVAL_AR"))
    nf = int(cfg.Nt) - 1
    if AR:
        print(f"EVAL_AR: autoregressive {nf}-step rollout (sliding T_in={T_in} window, 1 future/call)", flush=True)
    EB = int(os.environ.get("EVAL_EB", "8"))   # folded batch = EB*N_p; match inline eval (EVAL_BATCH=8) so the
                                                # pw_batched dilated-conv (SpaceToBatchND) path stays in its tested regime
    pf = []                                                                    # (N, n_future) PHYSICAL rel-L2 norm
    for i in range(0, len(xin), EB):
        sc = std[i:i + EB][:, None, None, None, :] + 1e-6
        mc = mean[i:i + EB][:, None, None, None, :]
        if AR:
            K = int(os.environ.get("EVAL_AR_K", "1"))                          # frames predicted+committed per call
            win = xin[i:i + EB].copy()                                         # (b,T_in,H,W,C) normalized window
            ar = []
            while len(ar) < nf:
                p1 = ev(win, dsc[i:i + EB], cf[i:i + EB], geom[i:i + EB]).numpy()  # (b,Nt,H,W,C)
                take = min(K, nf - len(ar))
                chunk = p1[:, 1:1 + take]                                      # (b,take,H,W,C) first `take` futures
                for j in range(take):
                    ar.append(chunk[:, j])
                win = np.concatenate([win[:, take:], chunk], axis=1)           # slide by `take`, append preds
            p_phys = np.stack(ar, axis=1) * sc + mc                            # (b,nf,H,W,C) physical AR futures
            t_phys = xtg[i:i + EB][:, 1:1 + nf] * sc + mc
        else:
            p = ev(xin[i:i + EB], dsc[i:i + EB], cf[i:i + EB], geom[i:i + EB]).numpy()  # (b,Nt,H,W,C) single-shot
            p_phys = p[:, 1:] * sc + mc                                        # futures (drop frame0=IC)
            t_phys = xtg[i:i + EB][:, 1:] * sc + mc
        r = per_frame_norm_rel_l2(p_phys, t_phys, cm_eval[i:i + EB]).numpy()
        pf.append(r)
    pf = np.concatenate(pf)                                                    # (N, n_future)
    n_future = pf.shape[1]
    tmf = tm[:, 1:1 + n_future].astype(np.float32)                            # (N, n_future) future validity mask
    #   PROSE evaluates each family over its VALID output frames only: 10 for full families, 4 for
    #   pdearena_uncond (valid_t=14, input_len=10 → 14-10=4; evaluate.py hard-limits it). Using the SAME
    #   temporal mask the training built guarantees we drop the clip-repeat padded frames.

    print(f"\n=== PROSE-EXACT rel-L2 (physical, time-avg of joint-channel L2 norm, eps=1e-7) ===", flush=True)
    print(f"{'family':16s} {'T':>3s} {'n':>5s} {'rel_l2':>9s} {'step_1':>9s} {'step_5':>9s} {'step_10':>9s}",
          flush=True)
    fam_vals = []
    for fi, nm in enumerate(fam_names):
        msk = fam == fi
        if not msk.any():
            continue
        sub = pf[msk]                                                          # (n, n_future)
        wm = tmf[msk].copy()                                                   # (n, n_future) validity
        # PROSE evaluates pdearena_uncond on 14-input_len = 4 frames (its native horizon is 14; the t20
        # cache clip-repeat-pads frames 14..19). Force that horizon here even if the deployed temporal
        # mask left them unmasked, so the uncond number isn't inflated by frozen padding frames.
        hard = {"pdearena_uncond": 4}.get(nm)
        if hard is not None:
            wm[:, hard:] = 0.0
        T = int(round(wm[0].sum()))                                           # valid future frames for this family
        rel = 100 * (sub * wm).sum(axis=1) / np.maximum(wm.sum(axis=1), 1e-8)  # per-sample masked time-avg
        rel = rel.mean()                                                       # PROSE _l2_error (mean over samples)
        kf = lambda k: 100 * (sub[:, :k] * wm[:, :k]).sum() / max(float((wm[:, :k]).sum()), 1e-8)
        s1, s5, s10 = kf(1), kf(min(5, n_future)), kf(min(10, n_future))       # cumulative-prefix (valid-masked)
        fam_vals.append(rel)
        print(f"{nm:16s} {T:3d} {int(msk.sum()):5d} {rel:8.3f}% {s1:8.3f}% {s5:8.3f}% {s10:8.3f}%", flush=True)
    print(f"{'AVE_BY_CLASS':16s} {'':>3s} {'':>5s} {np.mean(fam_vals):8.3f}%   (mean over "
          f"{len(fam_vals)} families — PROSE's headline aggregation)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_r1")
