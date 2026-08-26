"""End-to-end: loader example -> UniversalTPADA -> collocation loss -> gradient.
Covers grid transient (K=2), 3D (K=3), point steady (K=3, mesh). Run directly."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from core.model import UniversalTPADA
from core.train_step import loss_on_example
from data import loader
from data.registry import NUM_SLOTS


def _grid(K, T, dims):
    return {"family": "f", "K": K, "mode": "grid",
            "fields": np.random.randn(T, *dims, NUM_SLOTS).astype(np.float32),
            "dims": np.array(dims), "cmask": np.ones(NUM_SLOTS, np.float32),
            "roles": np.array(range(K)), "op_ids": np.zeros(8, np.int32),
            "op_multihot": np.zeros(13, np.float32), "param_ids": np.zeros(6, np.int32),
            "param_feats": np.zeros((6, 18), np.float32), "param_mask": np.zeros(6, np.float32)}


def _point_steady(K, N):
    s = _grid(K, 1, ())
    s["mode"] = "point"; s.pop("dims")
    s["fields"] = np.random.randn(1, N, NUM_SLOTS).astype(np.float32)
    s["coords"] = np.random.rand(N, K).astype(np.float32)
    s["geom"] = np.random.rand(N, 1).astype(np.float32)          # SDF channel
    return s


def main():
    tf.random.set_seed(0)
    m = UniversalTPADA(box_cfgs={2: (16, 16), 3: (10, 10, 10)}, d=48, depth=2,
                       d_w=6, n_p=12, c_out=NUM_SLOTS)
    tvars = (m.core.trainable_weights + m.penc.trainable_weights + m.grid_proj.trainable_weights
             + m.banks.trainable_weights + m.steady.trainable_weights + m.dec.trainable_weights
             + [g for s in m.synth.values() for g in (s.gains or [])])

    cases = {
        "grid2d-transient": loader.make_example(_grid(2, 20, (16, 16)), 10, 400, np.random.default_rng(0)),
        "grid3d-transient": loader.make_example(_grid(3, 21, (10, 10, 10)), 10, 300, np.random.default_rng(1)),
        "point3d-steady": loader.make_example(_point_steady(3, 2000), 10, 512, np.random.default_rng(2)),
    }
    for name, ex in cases.items():
        with tf.GradientTape() as tape:
            loss, pred = loss_on_example(m, ex)
        g = tape.gradient(loss, tvars)
        n_ok = sum(1 for gi in g if gi is not None)
        n_nz = sum(1 for gi in g if gi is not None and float(tf.norm(gi)) > 0)
        assert np.isfinite(float(loss)) and pred.shape[-1] == NUM_SLOTS
        print(f"PASS {name}: loss={float(loss):.3f} pred={tuple(pred.shape)} "
              f"grads {n_ok}/{len(tvars)} defined, {n_nz} nonzero")

    # one Adam step reduces loss on a fixed example (learning works)
    ex = cases["grid2d-transient"]
    opt = tf.keras.optimizers.Adam(1e-2)
    l0 = float(loss_on_example(m, ex)[0])
    for _ in range(15):
        with tf.GradientTape() as tape:
            loss, _ = loss_on_example(m, ex)
        opt.apply_gradients(zip(tape.gradient(loss, tvars), tvars))
    l1 = float(loss_on_example(m, ex)[0])
    print(f"overfit check: loss {l0:.3f} -> {l1:.3f} ({'DOWN' if l1 < l0 else 'UP'})")
    assert l1 < l0, "loss did not decrease"
    print("\nALL TRAIN-STEP TESTS PASS")


if __name__ == "__main__":
    main()
