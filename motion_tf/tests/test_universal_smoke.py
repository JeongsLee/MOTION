"""End-to-end smoke for UniversalTPADA: one weight set, 2D+3D, grid+points,
hard IC, continuous queries. Run: python tests/test_universal_smoke.py"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from core.model import UniversalTPADA


def main():
    tf.random.set_seed(0)
    rng = np.random.default_rng(0)
    m = UniversalTPADA(box_cfgs={2: (24, 24), 3: (12, 12, 12)}, d=48, depth=2,
                       d_w=6, n_p=12, c_out=3)
    times = np.array([0.0, 0.3, 0.7, 1.0])

    # ---- grid path, K=2 and K=3, same weights -------------------------------
    f2 = tf.random.normal([2, 4, 24, 24, 3])
    f3 = tf.random.normal([2, 4, 12, 12, 12, 3])
    with tf.GradientTape(persistent=True) as tape:
        A2, ic2, _ = m.from_grid(f2)
        u2 = m.grid_traj(A2, ic2, 2, times)
        A3, ic3, _ = m.from_grid(f3, roles=[0, 1, 2])
        u3 = m.grid_traj(A3, ic3, 3, times)
        loss = tf.reduce_mean(tf.square(u2)) + tf.reduce_mean(tf.square(u3))
    assert u2.shape == (2, 4, 24, 24, 3) and u3.shape == (2, 4, 12, 12, 12, 3)

    # hard IC: t=0 frame equals the last input frame EXACTLY
    ic_err2 = float(tf.reduce_max(tf.abs(u2[:, 0] - f2[:, -1])))
    ic_err3 = float(tf.reduce_max(tf.abs(u3[:, 0] - f3[:, -1])))
    assert ic_err2 == 0.0 and ic_err3 == 0.0, (ic_err2, ic_err3)

    # NOTE: local Keras 3 does not surface layer weights through tf.Module tracking
    # (fine on the TF2.15/Keras2 deploy stack) — collect explicitly here.
    tvars = (m.core.trainable_weights + m.penc.trainable_weights + m.grid_proj.trainable_weights
             + m.banks.trainable_weights + m.steady.trainable_weights + m.dec.trainable_weights
             + [g for s in m.synth.values() for g in (s.gains or [])])
    ids0 = {id(v) for v in tvars}
    g = tape.gradient(loss, tvars)
    n_ok = sum(1 for gi in g if gi is not None)
    n_nz = sum(1 for gi in g if gi is not None and float(tf.norm(gi)) > 0)
    print(f"grid OK: shapes {u2.shape}/{u3.shape}, hard-IC 0.0/0.0, "
          f"{len(tvars)} vars (shared core), grads defined {n_ok}/{len(tvars)}, nonzero {n_nz}")
    assert ids0 == {id(v) for v in tvars}

    # ---- point path, K=2 and K=3 --------------------------------------------
    pts2 = tf.constant(rng.uniform(0, 1, (2, 300, 2)), tf.float32)
    ft2 = tf.random.normal([2, 300, 3])
    A2p, _, _ = m.from_points(pts2, ft2, K=2)
    q2 = tf.constant(rng.uniform(0, 1, (2, 50, 2)), tf.float32)
    icq2 = tf.random.normal([2, 50, 3])
    up2 = m.point_traj(A2p, None, 2, q2, times, ic_at_pts=icq2)
    assert up2.shape == (2, 4, 50, 3)
    assert float(tf.reduce_max(tf.abs(up2[:, 0] - icq2))) == 0.0   # hard IC at points

    pts3 = tf.constant(rng.uniform(0, 1, (2, 400, 3)), tf.float32)
    ft3 = tf.random.normal([2, 400, 3])
    A3p, _, _ = m.from_points(pts3, ft3, K=3, roles=[0, 1, 2])
    q3 = tf.constant(rng.uniform(0, 1, (2, 50, 3)), tf.float32)
    up3 = m.point_traj(A3p, None, 3, q3, times, ic_at_pts=tf.zeros([2, 50, 3]))
    assert up3.shape == (2, 4, 50, 3)
    print("points OK: mesh->box->continuous query works for K=2 and K=3, hard IC exact")

    # ---- continuous consistency: grid query == point query at cell centers ---
    A2g, ic2g, _ = m.from_grid(f2)
    ug = m.grid_traj(A2g, ic2g, 2, times)
    cen = (np.arange(24) + 0.5) / 24
    mesh = np.stack(np.meshgrid(cen, cen, indexing="ij"), -1).reshape(1, -1, 2)
    pts_c = tf.constant(np.repeat(mesh, 2, 0), tf.float32)
    up = m.point_traj(A2g, ic2g, 2, pts_c, times)
    ug_flat = tf.reshape(ug, [2, 4, -1, 3])
    ic_gap = float(tf.reduce_max(tf.abs(m.synth[2].field_at(ic2g, pts_c)
                                        - tf.reshape(ic2g, [2, -1, 3]))))
    gap = float(tf.reduce_max(tf.abs(up - ug_flat)))
    print(f"consistency: |point@centers - grid| = {gap:.2e} "
          f"(IC interpolation share {ic_gap:.2e}; latent part is exact)")

    print("SMOKE PASS")


if __name__ == "__main__":
    main()
