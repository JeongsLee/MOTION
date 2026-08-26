"""Did the load-balance loss force families to SHARE experts?

r11 added the Switch aux loss and then most families got slightly worse at once (NS-Sines +3.1,
Wave-Layer +2.9, pdearena_ns +2.8, NS-Gauss +2.8, Poisson +2.1 ...). There is a specific reason to
expect that here rather than the usual Switch trade-off: a batch in this driver is bucketed by
family, so EVERY batch contains exactly ONE family, and a load-balance term computed inside a
batch therefore asks each family on its own to spread uniformly over all 16 experts. That is the
opposite of what the upcycle is for — it forbids a family from owning a subset of experts, and the
families then collide in the same weights.

This measures it directly: per family and per block, the router's load vector over the 16 experts,
then the cosine overlap between families. Rising overlap from r10 -> r11 is the interference; the
per-family entropy rising at the same time tells them apart from a plain accuracy regression.

  python diag_interfere.py <ckptA.npz> [<ckptB.npz>]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pretrain import load_ckpt                      # noqa: E402
from data import loader                                  # noqa: E402
from data.registry import FAMILIES, NUM_SLOTS                       # noqa: E402
from v3.data_ar import make_example_ar, stack_batch      # noqa: E402
from v3.model import V3Model                             # noqa: E402

# K=3 IS THE CONTESTED SPACE. Only five families exercise the 3D axial weights, and they split
# into two regimes that share one weight set: two steady single-channel SURFACE meshes that live
# or die on geometry (drivaernet 30.6, shapenet 14.1) against three transient five-channel
# periodic VOLUME turbulence sets with geom_kind=none (pdebench3d 10.2/15.9/17.2). Three-to-two,
# and six-to-four once the floor weighting doubles them. The 2D side has no comparable split.
FAMS = (os.environ.get("DI_FAMS", "").split(",") if os.environ.get("DI_FAMS") else
        ["drivaernet_pressure", "shapenet_car",
         "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08",
         "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08",
         "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08",
         "airfrans", "geofno_airfoil", "cfdbench", "pdearena_ns"])
N = int(os.environ.get("DI_N", "2"))
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "op_multihot", "cond", "fam_id", "sdf_box")


def loads_for(ckpt):
    model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=32, n_p=8, d_cond=128, dims2=(64, 64),
                    dims3=(32, 32, 32), t_in=10, transport=True, n_fams=len(FAMILIES),
                    n_slices=512, moe_experts=16, moe_topk=2)
    model.warm_build()
    V = model.trainable_variables
    load_ckpt(V, ckpt, d_w=32, names=[v.name for v in V])
    moes = [b[2] for b in model.blocks]
    out = {}
    rng = np.random.default_rng(1234)
    for fam in FAMS:
        if fam not in FAMILIES:
            continue
        acc, n = [], 0
        try:
            for s in loader.units_for(fam, "val"):
                K = int(s["K"])
                ex = make_example_ar(s, 10, 10, 4, 1024, 4096, rng,
                                     box_dims=(64, 64) if K == 2 else (32, 32, 32))
                b = stack_batch([ex])
                t = {k: tf.constant(b[k]) for k in KEYS}
                gd = tuple(b.get("dims", ())) or None
                pg = (tf.reshape(t["x_win"][:, -1], [1, gd[0], gd[1], model.S])
                      if b["dense2d"] else None)
                model.step(t["coords_node"], t["x_win"], t["fmask"], t["geom_node"], K,
                           tuple(int(r) for r in b["roles"]), t["cond"], t["op_multihot"],
                           tf.concat([t["coords_node"], t["coords_q"]], 1),
                           tf.concat([t["geom_node"], t["geom_q"]], 1),
                           tf.concat([t["x_win"][:, -1], t["ic_q"]], 1),
                           steady=bool(b["steady"]), prev_grid=pg, grid_dims=gd,
                           fam_id=t["fam_id"], sdf_box=t["sdf_box"])
                acc.append(np.stack([np.asarray(m.load) for m in moes]))     # (blocks,E)
                n += 1
                if n >= N:
                    break
        except Exception as e:
            print(f"  [skip] {fam}: {type(e).__name__}", flush=True)
            continue
        if n:
            out[fam] = np.mean(acc, 0)
    return out


def report(tag, L):
    fams = list(L)
    B = L[fams[0]].shape[0]
    # cosine overlap between families, averaged over blocks
    M = np.zeros((len(fams), len(fams)))
    for i, a in enumerate(fams):
        for j, b in enumerate(fams):
            v = [float(L[a][k] @ L[b][k] /
                       (np.linalg.norm(L[a][k]) * np.linalg.norm(L[b][k]) + 1e-9))
                 for k in range(B)]
            M[i, j] = np.mean(v)
    off = M[~np.eye(len(fams), dtype=bool)]
    ent = {}
    for f in fams:
        p = L[f] / np.maximum(L[f].sum(-1, keepdims=True), 1e-9)
        ent[f] = float(np.mean(-(p * np.log(p + 1e-12)).sum(-1) / np.log(p.shape[-1])))
    print(f"\n=== {tag} ===")
    print(f"mean cross-family expert overlap: {off.mean():.3f}   (0 = disjoint experts, "
          f"1 = every family on the same ones)")
    print(f"{'family':>22s} {'entropy':>8s}   overlap with the others")
    for i, f in enumerate(fams):
        o = np.delete(M[i], i)
        print(f"{f:>22s} {ent[f]:8.3f}   mean {o.mean():.3f}  max {o.max():.3f}")
    return off.mean()


def main():
    a = report(f"A: {os.path.basename(sys.argv[1])}", loads_for(sys.argv[1]))
    if len(sys.argv) > 2:
        b = report(f"B: {os.path.basename(sys.argv[2])}", loads_for(sys.argv[2]))
        print(f"\noverlap A -> B: {a:.3f} -> {b:.3f}  "
              f"({'MORE sharing = interference' if b > a else 'less sharing'})")
    print("DIAGINT_OK")


if __name__ == "__main__":
    main()
