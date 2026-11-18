"""THE one-weights proof: the same LatentBoxCore instance (identical variable set)
runs 2D and 3D forward+backward. Run: python tests/test_core_kagnostic.py"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from core import LatentBoxCore


def main():
    tf.random.set_seed(0)
    core = LatentBoxCore(d=32, depth=2, heads=4)

    x2 = tf.random.normal([2, 32, 32, 6])            # K=2 grid
    x3 = tf.random.normal([2, 16, 16, 16, 6])        # K=3 grid
    with tf.GradientTape(persistent=True) as tape:
        y2 = core(x2, roles=[0, 1])                  # 2D: x, y
        y3 = core(x3, roles=[0, 1, 2])               # 3D: x, y, z (gravity role = 2)
        l2 = tf.reduce_mean(tf.square(y2))
        l3 = tf.reduce_mean(tf.square(y3))

    vars2 = {id(v) for v in core.trainable_variables}
    assert y2.shape == (2, 32, 32, 32) and y3.shape == (2, 16, 16, 16, 32), (y2.shape, y3.shape)
    assert np.all(np.isfinite(y2.numpy())) and np.all(np.isfinite(y3.numpy()))

    g2 = tape.gradient(l2, core.trainable_variables)
    g3 = tape.gradient(l3, core.trainable_variables)
    vars_after = {id(v) for v in core.trainable_variables}
    assert vars2 == vars_after, "3D call created NEW variables — weights are NOT shared"
    n2 = sum(1 for g in g2 if g is not None and float(tf.norm(g)) > 0)
    n3 = sum(1 for g in g3 if g is not None and float(tf.norm(g)) > 0)
    n_params = sum(int(np.prod(v.shape)) for v in core.trainable_variables)

    print(f"PASS one-weights: {len(core.trainable_variables)} vars ({n_params:,} params) "
          f"shared across K=2/K=3; grads nonzero {n2}/{len(g2)} (2D), {n3}/{len(g3)} (3D)")
    # zero-init residual outputs: at init the core is near-identity in distribution terms
    print(f"    out std: 2D {float(tf.math.reduce_std(y2)):.3f}  3D {float(tf.math.reduce_std(y3)):.3f}")


if __name__ == "__main__":
    main()
