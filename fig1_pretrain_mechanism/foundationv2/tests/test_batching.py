"""Batched loss + bucketing + multi-step training smoke (synthetic). Run directly."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from core.batching import bucketed_batches
from core.model import UniversalTPADA
from core.train_step import batched_loss
from data import loader
from data.registry import NUM_SLOTS


def _grid(K, T, dims, fam):
    return {"family": fam, "K": K, "mode": "grid",
            "fields": np.random.randn(T, *dims, NUM_SLOTS).astype(np.float32),
            "dims": np.array(dims), "cmask": np.ones(NUM_SLOTS, np.float32),
            "roles": np.array(range(K)), "op_ids": np.zeros(8, np.int32),
            "op_multihot": np.zeros(13, np.float32), "param_ids": np.zeros(6, np.int32),
            "param_feats": np.zeros((6, 18), np.float32), "param_mask": np.zeros(6, np.float32)}


def _point(K, N, fam):
    s = _grid(K, 1, (), fam); s["mode"] = "point"; s.pop("dims")
    s["fields"] = np.random.randn(1, N, NUM_SLOTS).astype(np.float32)
    s["coords"] = np.random.rand(N, K).astype(np.float32)
    return s


def main():
    tf.random.set_seed(0)
    rng = np.random.default_rng(0)
    # mixed corpus: two 2D-grid families (same 16x16 -> bucket together), a 3D grid, mesh points
    exs = []
    for _ in range(4):
        exs.append(loader.make_example(_grid(2, 20, (16, 16), "A"), 10, 256, rng))
    for _ in range(4):
        exs.append(loader.make_example(_grid(2, 20, (16, 16), "B"), 10, 256, rng))
    for _ in range(2):
        exs.append(loader.make_example(_grid(3, 21, (10, 10, 10), "C3d"), 10, 256, rng))
    for _ in range(3):
        exs.append(loader.make_example(_point(3, rng.integers(1500, 4000), "mesh"), 10, 256, rng))

    batches = list(bucketed_batches(iter(exs), batch_size=4, S=NUM_SLOTS, enc_n=2048, seed=0))
    keys = [(b["K"], b["coords_node"].shape[1], len(b["families"])) for b in batches]
    print("buckets (K, N_enc, B):", keys)
    assert all(b["coords_node"].shape[1] == 2048 for b in batches)   # every example bounded to enc_n
    assert any(b["coords_node"].shape[0] == 4 for b in batches)      # full B=4 batch formed
    assert {b["K"] for b in batches} == {2, 3}                       # both dims bucketed

    m = UniversalTPADA(box_cfgs={2: (16, 16), 3: (10, 10, 10)}, d=48, depth=2,
                       d_w=6, n_p=12, c_out=NUM_SLOTS)
    V = None
    opt = tf.keras.optimizers.Adam(5e-3)
    l0 = None
    for it in range(20):
        b = batches[it % len(batches)]
        with tf.GradientTape() as tape:
            loss = batched_loss(m, b)
        if V is None:
            V = (m.core.trainable_weights + m.penc.trainable_weights + m.grid_proj.trainable_weights
                 + m.banks.trainable_weights + m.steady.trainable_weights + m.dec.trainable_weights
                 + [g for s in m.synth.values() for g in (s.gains or [])])
        g = tape.gradient(loss, V)
        g = [gi if gi is not None else tf.zeros_like(vi) for gi, vi in zip(g, V)]
        opt.apply_gradients(zip(g, V))
        if l0 is None:
            l0 = float(loss)
    l1 = float(batched_loss(m, batches[0]))
    print(f"batched loss: shapes ok, B=4 stacks work; loss {l0:.3f} -> {l1:.3f}")
    assert np.isfinite(l1)
    print("\nBATCHING SMOKE PASS")


if __name__ == "__main__":
    main()
