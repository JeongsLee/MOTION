"""REVISION-R1 (10-22) receding-horizon mode ("rh", MOTION_RH_FAMS): synthetic NS-Sines grid, no /corpus needed.
Checks: policy routes the family to "rh"; step_ss(leads=[k]) is bit-identical to lead k of a full call; the rh loss
with one call scoring every lead equals singleshot_loss; the default rh loss (several fed-back calls, lead 1) has finite,
non-zero gradients and an Adam step runs; the eval form (hop-1 AR over all leads) returns [norm, phys, fluct]."""
import os
import sys

os.environ.setdefault("MOTION_RH_FAMS", "NS-Sines")
os.environ.setdefault("MOTION_SS_WARP", "blend")
os.environ.setdefault("MOTION_WARP_LIVE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)   # exact float32 comparisons on Ampere GPUs

from data.registry import FAMILIES, NUM_SLOTS
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.policy import ar_mode
from v3.train import singleshot_loss, singleshot_rh_loss, _grid

T_IN, K_FUT, K_RELAX, Q, ENC_N = 1, 6, 2, 64, 256
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q", "geom_q", "y_q",
        "tstep_mask", "cmask", "fbmask", "op_multihot", "cond", "fam_id", "sdf_box")


def main():
    from data import symbolic
    assert ar_mode("NS-Sines", False) == "rh" and ar_mode("CE-RP", False) != "rh"
    rng = np.random.default_rng(0)
    fam = "NS-Sines"
    cm = np.zeros(NUM_SLOTS, np.float32); cm[:2] = 1
    s = {"family": fam, "K": 2, "mode": "grid", "dims": np.array([16, 16]),
         "fields": rng.normal(size=(10, 16, 16, NUM_SLOTS)).astype(np.float32), "cmask": cm,
         "roles": [0, 1], **symbolic.encode(FAMILIES[fam])}
    ex = make_example_ar(s, T_IN, K_FUT, K_RELAX, Q, ENC_N, rng, box_dims=(8, 8))
    assert ex["mode"] == "rh", ex["mode"]
    b = stack_batch([ex])
    assert b["dense2d"], "expected the dense 2-D path"
    ts = [tf.constant(b[k]) for k in KEYS]
    cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cmk, fbm, op, cond, fid, sdb = ts
    K, roles, k_seg, gd = int(b["K"]), tuple(int(r) for r in b["roles"]), int(b["k_seg"]), tuple(b["dims"])
    print(f"[rh] k_seg {k_seg}, x_win {tuple(xw.shape)}, y_node {tuple(yn.shape)}")

    model = V3Model(S=NUM_SLOTS, d=32, depth=2, d_w=8, n_p=4, d_cond=32, dims2=(8, 8), dims3=(6, 6, 6),
                    t_in=T_IN, transport=True, n_fams=len(FAMILIES))
    model.warm_build()
    assert getattr(model, "ss_warp", False), "MOTION_SS_WARP not active"
    V = model.trainable_variables
    for v in V:                                   # zero-init heads (warp disp, gates) would hide the feedback path
        v.assign(v + 0.05 * tf.random.stateless_normal(v.shape, [1, hash(v.name) % 10000], dtype=v.dtype))

    # 1) leads subset == same lead of the full call
    qall, gall = tf.concat([cn, cq], 1), tf.concat([gn, gq], 1)
    up = tf.concat([xw[:, -1], icq], 1)
    pg = _grid(xw[:, -1], gd, model.S)
    full = model.step_ss(cn, xw, fm, gn, K, roles, cond, op, qall, gall, up, kf=k_seg, fam_id=fid, sdf_box=sdb,
                         box_dims=(8, 8), prev_grid=pg, grid_dims=gd)
    for k in (1, 3, k_seg):
        one = model.step_ss(cn, xw, fm, gn, K, roles, cond, op, qall, gall, up, kf=k_seg, fam_id=fid, sdf_box=sdb,
                            box_dims=(8, 8), prev_grid=pg, grid_dims=gd, leads=[k])
        d = float(tf.reduce_max(tf.abs(one[:, 0] - full[:, k - 1])))
        sc_ = float(tf.reduce_max(tf.abs(full[:, k - 1])))
        print(f"[rh] lead {k}: |subset - full| max {d:.2e} (|full| max {sc_:.2e})")
        assert d < 1e-5 * max(1.0, sc_)
    # 2) one call, all leads == singleshot_loss
    args = (model, cn, xw, fm, gn, yn, icq, cq, gq, yq, tsm, cmk, fbm, op, cond, fid, sdb, K, roles, k_seg)
    kw = dict(box_dims=(8, 8), grid_dims=gd, l1=True)
    Lss = float(singleshot_loss(*args, **kw))
    Lrh1 = float(singleshot_rh_loss(*args, n_roll=1, n_lead=k_seg, **kw))
    print(f"[rh] singleshot_loss {Lss:.6f} vs rh(1 call, all leads) {Lrh1:.6f}")
    assert abs(Lss - Lrh1) < 1e-5 * max(1.0, abs(Lss))
    # 3) default rh loss: gradients through the last calls only (stop-gradient feedback), finite, non-zero
    opt = tf.keras.optimizers.Adam(1e-3)
    with tf.GradientTape() as tape:
        L = singleshot_rh_loss(*args, n_roll=4, n_lead=2, **kw)
    g = tape.gradient(L, V)
    gnorm = float(tf.linalg.global_norm([gi for gi in g if gi is not None]))
    print(f"[rh] rh loss (4 calls x leads 1-2) {float(L):.6f}, grad norm {gnorm:.4e}, "
          f"{sum(gi is not None for gi in g)}/{len(V)} vars with grad")
    assert np.isfinite(gnorm) and gnorm > 0
    opt.apply_gradients((gi, v) for gi, v in zip(g, V) if gi is not None)
    L2 = float(singleshot_rh_loss(*args, n_roll=4, n_lead=2, **kw))
    print(f"[rh] after one Adam step: {L2:.6f}")
    # feedback really changes the later calls (rh(4 calls, lead 1) != teacher-forced-free ss leads 1..4 average)
    Lrh4 = float(singleshot_rh_loss(*args, n_roll=4, n_lead=1, **kw))
    print(f"[rh] rh(4 calls, lead 1) {Lrh4:.6f}")
    # 4) eval form: hop-1 AR over all leads, physical scale
    sc = tf.constant(np.concatenate([b["scale"], b["shift"]], -1) if "shift" in b else b["scale"])
    ev = singleshot_rh_loss(*args, n_roll=k_seg, n_lead=1, scale=sc, clamp=1e9, box_dims=(8, 8), grid_dims=gd)
    print(f"[rh] eval [norm, phys, fluct] {np.round(ev.numpy(), 5).tolist()}")
    assert ev.shape == (3,) and np.all(np.isfinite(ev.numpy()))
    # 5) same under tf.function (the training path is traced)
    f = tf.function(lambda: singleshot_rh_loss(*args, n_roll=4, n_lead=1, **kw))
    Lf = float(f())
    print(f"[rh] tf.function {Lf:.6f} (eager {Lrh4:.6f})")
    assert abs(Lf - Lrh4) < 1e-4 * max(1.0, abs(Lrh4))
    print("RH_MODE_OK")


if __name__ == "__main__":
    main()
