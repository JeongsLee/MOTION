"""MOTION-NCS smoke: the 2026-08-06 re-architecture, synthetic inputs only (no corpus).

  python -m tests.test_motion_smoke

Checks
  1. vortex-stretching feature is IDENTICALLY zero for K=2 and non-zero for K=3
  2. gradient reaches the vortex gate on a K=3 step and is exactly zero on K=2
  3. step_ss shapes; frame-k consistency: its t=1 frame equals the AR step's output at init
     (shared readout calculus: mean over panels == last partial sum)
  4. hard IC: step_ss deltas vanish as t -> 0+ relative to frame scale (z(t) -> 0)
  5. fam_head=1: outputs differ across fam_id, and the shared head `o` receives no gradient
  6. motion_families() is exactly the 19-family corpus, no cfdbench, no geometry
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from core.banks import TendencyBanks
from data.registry import FAMILIES, MOTION_FAMILIES, motion_families
from v3.model import V3Model
from v3.policy import AR_MODE, ar_mode

RED, GRN, END = "\033[91m", "\033[92m", "\033[0m"
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{GRN+'PASS'+END if ok else RED+'FAIL'+END}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main():
    tf.random.set_seed(0)
    np.random.seed(0)

    # ---------------------------------------------------------------- registry / policy
    print("[1] corpus & policy", flush=True)
    fams = [f.name for f in motion_families()]
    check("19 families", len(fams) == 19, f"({len(fams)})")
    check("no cfdbench / geometry",
          "cfdbench" not in fams and all(FAMILIES[f].geom_kind in ("none",) or f == "cfdbench"
                                         for f in fams),
          "")
    n3d = sum(1 for f in fams if FAMILIES[f].K == 3)
    check("3 x 3D", n3d == 3, f"({n3d})")
    check("every family has a mode", all(f in AR_MODE for f in fams))
    check("ss set", {f for f in fams if ar_mode(f, False) == "ss"}
          == {"ACE", "diff_react", "Wave-Layer"})

    # ---------------------------------------------------------------- vortex feature
    print("[2] vortex-stretching bank", flush=True)
    banks = TendencyBanks(out=32, d=16, name="tb")
    h2 = tf.random.normal([1, 8, 8, 16])
    h3 = tf.random.normal([1, 8, 8, 8, 16])
    u2 = banks.vel(h2)
    u3 = banks.vel(h3)
    f2 = banks._vortex_feat(u2, 2)
    f3 = banks._vortex_feat(u3, 3)
    check("K=2 feature == 0 exactly", float(tf.reduce_max(tf.abs(f2))) == 0.0,
          f"max|f2|={float(tf.reduce_max(tf.abs(f2))):.2e}")
    check("K=3 feature != 0", float(tf.reduce_max(tf.abs(f3))) > 1e-8,
          f"max|f3|={float(tf.reduce_max(tf.abs(f3))):.2e}")

    with tf.GradientTape() as tp:
        W3 = banks(h3, K=3)
        L3 = tf.reduce_sum(tf.square(W3))
    g3 = tp.gradient(L3, banks.gates["vortex"])
    with tf.GradientTape() as tp:
        W2 = banks(h2, K=2)
        L2 = tf.reduce_sum(tf.square(W2))
    g2 = tp.gradient(L2, banks.gates["vortex"])
    check("K=3 vortex gate grad != 0", g3 is not None and float(tf.reduce_max(tf.abs(g3))) > 0)
    check("K=2 vortex gate grad == 0",
          g2 is None or float(tf.reduce_max(tf.abs(g2))) == 0.0)

    # ---------------------------------------------------------------- model: ss vs ar
    print("[3] step_ss / fam_head", flush=True)
    S, KF = 16, 10
    m = V3Model(S=S, d=32, depth=1, d_w=16, n_p=16, d_cond=32, dims2=(8, 8), dims3=(4, 4, 4),
                t_in=4, transport=True, n_fams=len(FAMILIES), n_semantic=12,
                gate_cond=1, dict_decode=0, fam_head=1)
    from v3.cond import COND_DIM
    m.warm_build()
    B, N, Q, K = 2, 64, 12, 2
    cn = tf.constant(np.stack([np.stack(np.meshgrid(
        (np.arange(8) + .5) / 8, (np.arange(8) + .5) / 8, indexing="ij"), -1).reshape(-1, 2)] * B)
        .astype(np.float32))
    win = tf.random.normal([B, 4, N, S])
    fm = tf.ones([B, 4])
    gn = tf.zeros([B, N, 8])
    cond = tf.random.normal([B, COND_DIM])
    op = tf.ones([B, 13])
    cq = tf.random.uniform([B, Q, K])
    gq = tf.zeros([B, Q, 8])
    up = tf.random.normal([B, Q, S])
    fid = tf.constant([1, 3], tf.int32)
    sb = tf.zeros([B, 8, 8, 1])

    preds = m.step_ss(cn, win, fm, gn, K, [0, 1], cond, op, cq, gq, up, kf=KF,
                      fam_id=fid, sdf_box=sb, box_dims=(8, 8))
    check("step_ss shape", tuple(preds.shape) == (B, KF, Q, S), f"{tuple(preds.shape)}")

    one = m.step(cn, win, fm, gn, K, [0, 1], cond, op, cq, gq, up,
                 steady=False, prev_grid=tf.reshape(win[:, -1], [B, 8, 8, S]),
                 grid_dims=(8, 8), fam_id=fid, sdf_box=sb, box_dims=(8, 8))
    diff = float(tf.reduce_max(tf.abs(preds[:, -1] - one)))
    check("ss t=1 == ar step (init)", diff < 1e-4, f"max diff {diff:.2e}")

    # hard IC: the increment |u(t_k) - u_prev| must shrink toward t -> 0
    d1 = float(tf.reduce_mean(tf.abs(preds[:, 0] - up)))
    dK = float(tf.reduce_mean(tf.abs(preds[:, -1] - up)))
    check("hard IC (delta grows from t=0)", d1 < dK, f"|d(t1)|={d1:.3e} |d(tK)|={dK:.3e}")

    o2 = m.step_ss(cn, win, fm, gn, K, [0, 1], cond, op, cq, gq, up, kf=KF,
                   fam_id=tf.constant([2, 4], tf.int32), sdf_box=sb, box_dims=(8, 8))
    check("fam_head: fam_id changes output",
          float(tf.reduce_max(tf.abs(o2 - preds))) > 1e-8)

    with tf.GradientTape() as tp:
        p = m.step_ss(cn, win, fm, gn, K, [0, 1], cond, op, cq, gq, up, kf=KF,
                      fam_id=fid, sdf_box=sb, box_dims=(8, 8))
        L = tf.reduce_sum(tf.square(p))
    go = tp.gradient(L, m.dec.o.trainable_variables)
    check("shared head o gets NO gradient (fam_only)",
          all(g is None for g in go), "")

    print(("\n" + RED + f"{len(FAILS)} FAILURES: " + ", ".join(FAILS) + END) if FAILS
          else "\n" + GRN + "ALL PASS" + END, flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
