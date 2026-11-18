"""Unit tests for the registry/symbolic/schema/splits layer (pure numpy).
Run: python tests/test_preprocess.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import registry as R
from data import schema as S
from data import splits as SP
from data import symbolic as SY


def test_registry_coverage():
    fams = R.FAMILIES
    assert len(fams) >= 20
    # every family channel maps into valid slots; K in {2,3}; roles length ok
    for f in fams.values():
        assert f.K in (2, 3), f.name
        assert all(0 <= c < R.NUM_SLOTS for c in f.channels), f.name
        assert len(f.roles) >= f.K
        assert len(f.bc) >= f.K
        assert f.stage in ("pretrain", "downstream")
        assert f.time in ("transient", "steady")
    assert any(f.K == 3 for f in fams.values())               # 3D present
    assert {f.name for f in R.downstream_families()} >= {"ACE", "Poisson-Gauss", "shapenet_car"}
    # gravity role appears on families that have a vertical axis meaning (buoyancy)
    print(f"PASS registry: {len(fams)} families, "
          f"{len(R.pretrain_families())} pretrain / {len(R.downstream_families())} downstream, "
          f"3D: {[f.name for f in R.by_dim(3)]}")


def test_symbolic_encode():
    f = R.FAMILIES["pdearena_ns"]
    c = SY.encode(f, param_values={"nu": 1e-4, "buoyancy": 0.5})
    assert c["op_ids"].shape == (SY.MAX_OPS,)
    assert c["op_multihot"][SY.OP2ID["buoyancy"]] == 1.0
    assert c["param_feats"].shape == (SY.MAX_PARAMS, SY.PARAM_FEAT_DIM)
    assert c["param_mask"][:2].sum() == 2 and c["param_mask"][2:].sum() == 0
    # log-Fourier keeps decades bounded & distinguishes magnitudes
    a = SY._logfourier(1e-6); b = SY._logfourier(1e-1)
    assert np.all(np.isfinite(a)) and not np.allclose(a, b)
    # null / absent conditioning
    cn = SY.encode(R.FAMILIES["shallow_water"])
    assert cn["param_mask"].sum() == 0                       # SWE has no scalar params
    print(f"PASS symbolic: op vocab {len(SY.OPERATOR_VOCAB)}, param feat dim {SY.PARAM_FEAT_DIM}, "
          f"pdearena_ns ops -> {[SY.OPERATOR_VOCAB[i] for i in c['op_ids'] if i]}")


def test_schema_grid_and_point():
    f = R.FAMILIES["com_ns"]                                  # 4 raw channels -> slots 0,1,3,4
    cond = SY.encode(f)
    raw = np.random.default_rng(0).standard_normal((20, 32, 32, 4)).astype(np.float32)
    g = S.grid_sample("com_ns", 2, raw, f, cond, dims=(32, 32))
    assert g.cmask.sum() == 4 and g.cmask[0] == 1 and g.cmask[2] == 0  # slot2(vz) unused in 2D
    assert g.fields[..., 3].std() > 0                        # density slot filled

    # 3D grid
    f3 = R.by_dim(3)[0]
    raw3 = np.random.default_rng(1).standard_normal((5, 8, 8, 8, 5)).astype(np.float32)
    g3 = S.grid_sample(f3.name, 3, raw3, f3, SY.encode(f3), dims=(8, 8, 8))
    assert g3.fields.shape == (5, 8, 8, 8, R.NUM_SLOTS) and g3.cmask.sum() == 5

    # point mesh sample + coord normalization
    fm = R.FAMILIES["shapenet_car"]
    xyz = np.random.default_rng(2).uniform(-3, 3, (500, 3))
    coords, (lo, hi) = S.normalize_coords(xyz)
    fr = np.random.default_rng(3).standard_normal((1, 500, 1)).astype(np.float32)
    p = S.point_sample("shapenet_car", 3, coords, fr, fm, SY.encode(fm))
    assert p.mode == "point" and p.fields.shape == (1, 500, R.NUM_SLOTS)
    assert p.cmask[4] == 1.0                                  # pressure slot
    print("PASS schema: grid 2D/3D + point mesh, slot mapping & coord-norm validated")


def test_splits_deterministic_and_leakproof():
    units = [f"traj_{i}" for i in range(1000)]
    b1 = SP.deterministic_split(units, seed=0)
    b2 = SP.deterministic_split(list(reversed(units)), seed=0)
    # order-independent
    assert set(b1["train"]) == set(b2["train"])
    # disjoint + complete
    allids = set(b1["train"]) | set(b1["val"]) | set(b1["test"])
    assert allids == set(units)
    assert not (set(b1["train"]) & set(b1["test"]))
    fr = {k: len(v) / 1000 for k, v in b1.items()}
    assert abs(fr["train"] - 0.8) < 0.05 and abs(fr["test"] - 0.1) < 0.03
    m = SP.make_manifest(R.FAMILIES["incom_ns"], units, seed=0)
    assert m["counts"]["train"] + m["counts"]["val"] + m["counts"]["test"] == 1000
    print(f"PASS splits: deterministic 80/10/10 {fr}, order-independent, disjoint")


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        fn()
    print("\nALL PREPROCESS TESTS PASS")
