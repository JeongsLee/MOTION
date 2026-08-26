"""v3 smoke: ONE weight set runs full-step AR across 2D-grid / 3D-grid / mesh / steady
buckets, gradients flow, no new variables appear after warm_build. Runnable as a script
or via pytest. Synthetic data (no /corpus needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from data.registry import FAMILIES
from v3.data_ar import make_example_ar, stack_batch
from v3.model import V3Model
from v3.train import rollout_loss

T_IN, K_FUT, K_RELAX, Q, ENC_N = 4, 3, 2, 64, 128


def _cond_arrays(fam):
    from data import symbolic
    return symbolic.encode(FAMILIES[fam])


def _sample_grid2d(rng):
    fam = "incom_ns"
    s = {"family": fam, "K": 2, "mode": "grid", "dims": np.array([16, 16]),
         "fields": rng.normal(size=(8, 16, 16, 9)).astype(np.float32),
         "cmask": np.array([1, 1, 0, 0, 0, 0, 1, 0, 0], np.float32),
         "roles": [0, 1], **_cond_arrays(fam)}
    return s


def _sample_grid3d(rng):
    fam = [f for f in FAMILIES if f.startswith("pdebench3d")][0]
    s = {"family": fam, "K": 3, "mode": "grid", "dims": np.array([8, 8, 8]),
         "fields": rng.normal(size=(6, 8, 8, 8, 9)).astype(np.float32),
         "cmask": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], np.float32),
         "roles": [0, 1, 2], **_cond_arrays(fam)}
    return s


def _sample_mesh_steady(rng, fam, K):
    N = 300
    s = {"family": fam, "K": K, "mode": "point",
         "coords": rng.uniform(size=(N, K)).astype(np.float32),
         "fields": rng.normal(size=(1, N, 9)).astype(np.float32),
         "geom": rng.normal(size=(N, 3)).astype(np.float32),
         "cmask": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], np.float32),
         "roles": list(range(K)), **_cond_arrays(fam)}
    s["sdf_vol"] = rng.normal(size=(12,) * K).astype(np.float32)   # GINO dense shape context
    return s


def main():
    rng = np.random.default_rng(0)
    model = V3Model(S=9, d=32, depth=2, d_w=8, n_p=4, d_cond=32,
                    dims2=(8, 8), dims3=(6, 6, 6), t_in=T_IN, transport=True,
                    n_fams=len(FAMILIES), n_slices=16)
    model.warm_build()
    nvars = len(model.trainable_variables)
    print(f"[smoke] vars after warm_build: {nvars}")

    samples = [_sample_grid2d(rng), _sample_grid3d(rng),
               _sample_mesh_steady(rng, "airfrans", 2),
               _sample_mesh_steady(rng, "drivaernet_pressure", 3)]
    opt = tf.keras.optimizers.Adam(1e-3)
    V = model.trainable_variables

    for s in samples:
        bd = (8, 8) if int(s['K']) == 2 else (6, 6, 6)
        ex = make_example_ar(s, T_IN, K_FUT, K_RELAX, Q, ENC_N, rng, box_dims=bd)
        b = stack_batch([ex, ex] if not ex["dense2d"] else [ex])
        keys = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q",
                "coords_q", "geom_q", "y_q", "tstep_mask", "cmask", "op_multihot",
                "cond", "fam_id", "sdf_box")
        ts = tuple(tf.constant(b[k]) for k in keys)
        gd = tuple(b.get("dims", ())) or None
        st = (int(b["K"]), bool(b["steady"]), bool(b["dense2d"]),
              tuple(int(r) for r in b["roles"]), int(b["k_seg"]), gd)
        losses = []
        for it in range(3):
            with tf.GradientTape() as tape:
                L = rollout_loss(model, *ts, *st)
            g = tape.gradient(L, V)
            n_live = sum(1 for gi in g if gi is not None and float(tf.reduce_max(tf.abs(gi))) > 0)
            opt.apply_gradients((gi, v) for gi, v in zip(g, V) if gi is not None)
            losses.append(float(L))
        assert np.isfinite(losses).all(), (s["family"], losses)
        print(f"[smoke] {s['family']:24s} K={st[0]} steady={st[1]} dense2d={st[2]} "
              f"kseg={st[4]} loss {losses[0]:.3f}->{losses[-1]:.3f} live-grads {n_live}")

    assert len(model.trainable_variables) == nvars, "new vars appeared after warm_build"
    print(f"[smoke] OK — one weight set ({nvars} vars) served all buckets")


def test_v3_smoke():
    main()


if __name__ == "__main__":
    main()
