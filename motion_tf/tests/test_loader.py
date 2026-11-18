"""Collocation dataloader test (make_example logic, no corpus needed).
Run: python tests/test_loader.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import loader
from data.registry import NUM_SLOTS


def _grid_sample(K, T, dims, family="incom_ns"):
    return {"family": family, "K": K, "mode": "grid",
            "fields": np.random.randn(T, *dims, NUM_SLOTS).astype(np.float32),
            "dims": np.array(dims), "cmask": np.ones(NUM_SLOTS, np.float32),
            "roles": np.array(range(K)), "op_ids": np.zeros(8, np.int32),
            "op_multihot": np.zeros(13, np.float32), "param_ids": np.zeros(6, np.int32),
            "param_feats": np.zeros((6, 18), np.float32), "param_mask": np.zeros(6, np.float32)}


def _point_sample(K, T, N, family="shapenet_car"):
    s = _grid_sample(K, T, (), family)
    s["mode"] = "point"; s.pop("dims")
    s["fields"] = np.random.randn(T, N, NUM_SLOTS).astype(np.float32)
    s["coords"] = np.random.rand(N, K).astype(np.float32)
    return s


def test_grid_transient():
    rng = np.random.default_rng(0)
    ex = loader.make_example(_grid_sample(2, 20, (16, 16)), t_in=10, n_colloc=512, rng=rng)
    assert ex["mode"] == "grid" and ex["K"] == 2 and not ex["steady"]
    assert ex["x_frames"].shape == (10, 256, NUM_SLOTS)           # ti frames, 16*16 nodes
    assert ex["coords_q"].shape == (512, 2) and ex["y_q"].shape == (512, NUM_SLOTS)
    assert ex["ic_q"].shape == (512, NUM_SLOTS)
    assert ex["t_q"].min() > 0 and ex["t_q"].max() <= 1 + 1e-6    # pure future window
    print("PASS grid transient: x", ex["x_frames"].shape, "ic_q", ex["ic_q"].shape,
          "t_q in (%.2f,%.2f]" % (ex["t_q"].min(), ex["t_q"].max()))


def test_grid_3d():
    rng = np.random.default_rng(1)
    ex = loader.make_example(_grid_sample(3, 21, (8, 8, 8), "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08"),
                             t_in=10, n_colloc=256, rng=rng)
    assert ex["K"] == 3 and ex["coords_q"].shape == (256, 3)
    assert ex["x_frames"].shape == (10, 512, NUM_SLOTS)
    print("PASS grid 3D: nodes", ex["coords_node"].shape, "colloc t range",
          round(float(ex["t_q"].min()), 2), round(float(ex["t_q"].max()), 2))


def test_point_and_steady():
    rng = np.random.default_rng(2)
    ex = loader.make_example(_point_sample(3, 1, 4000), t_in=10, n_colloc=1024, rng=rng)  # steady
    assert ex["mode"] == "point" and ex["steady"]
    assert ex["x_frames"].shape == (1, 4000, NUM_SLOTS)          # zeros (no field leakage)
    assert np.all(ex["x_frames"] == 0)                           # input is geometry, not the answer
    assert np.all(ex["t_q"] == 1.0) and np.all(ex["ic_q"] == 0)  # relaxation endpoint, no prior field
    assert ex["coords_q"].shape == (1024, 3)
    print("PASS point/steady: x-zeros(no leak), t_q==1, ic_q==0, y_q from field", ex["y_q"].shape)


if __name__ == "__main__":
    for k, v in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        v()
    print("\nALL LOADER TESTS PASS")
