"""Where does cfdbench's error live — at the solid/fluid boundary or everywhere?

cfdbench is the only 2D grid family with WALLS (every other one is periodic), it is the family
our previous generation solved best (0.128%) and the one we are now worst on relative to it
(~94x, and not explained by the loss definition or by physical-vs-normalized scoring). Two
mechanisms would both concentrate error at the interface:
  * the 128^2 field is scattered into a 64^2 latent box and read back by multilinear
    interpolation, which smears a step-shaped solid/fluid edge over a cell;
  * the boundary mask moved from a per-frame FIELD channel (what the previous generation fed,
    including through the AR feedback) to a static geom channel.
This bins the per-point error by distance to the mask edge. Error flat in distance means the
problem is global (capacity); error concentrated within a cell or two of the edge means it is
the interface representation.

  python diag_wall.py <ckpt.npz> [family]
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

FAM = sys.argv[2] if len(sys.argv) > 2 else "cfdbench"
N = int(os.environ.get("DW_N", "4"))
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "op_multihot", "cond", "fam_id", "sdf_box")


def edge_distance(mask, dims):
    """Cells' distance (in cells) to the nearest solid/fluid transition of a binary mask."""
    from scipy.ndimage import distance_transform_edt as edt
    m = mask.reshape(dims) > 0.5
    # boundary = fluid cells adjacent to solid (or vice versa)
    b = np.zeros_like(m, bool)
    for ax in range(m.ndim):
        b |= (m != np.roll(m, 1, ax)) | (m != np.roll(m, -1, ax))
    return edt(~b)


def main():
    model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=32, n_p=8, d_cond=128, dims2=(64, 64),
                    dims3=(32, 32, 32), t_in=10, transport=True, n_fams=len(FAMILIES),
                    n_slices=512, moe_experts=16, moe_topk=2)
    model.warm_build()
    V = model.trainable_variables
    load_ckpt(V, sys.argv[1], d_w=32, names=[v.name for v in V])

    rng = np.random.default_rng(1234)
    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 999)]
    acc = {b: [0.0, 0.0] for b in bins}                  # [sum sq err, sum sq ref]
    n = 0
    for s in loader.units_for(FAM, "val"):
        dims = tuple(int(x) for x in s["dims"])
        ex = make_example_ar(s, 10, 10, 4, 2048, 8192, rng, box_dims=(64, 64))
        b = stack_batch([ex])
        t = {k: tf.constant(b[k]) for k in KEYS}
        gd = dims
        B = 1
        Nn = int(b["coords_node"].shape[1])
        qall = tf.concat([t["coords_node"], t["coords_q"]], 1)
        gall = tf.concat([t["geom_node"], t["geom_q"]], 1)
        u_prev = tf.concat([t["x_win"][:, -1], t["ic_q"]], 1)
        win, fm = t["x_win"], t["fmask"]
        pg = tf.reshape(t["x_win"][:, -1], [B, gd[0], gd[1], model.S]) if b["dense2d"] else None
        pred = model.step(t["coords_node"], win, fm, t["geom_node"], 2, (0, 1), t["cond"],
                          t["op_multihot"], qall, gall, u_prev, steady=False,
                          prev_grid=pg, grid_dims=gd, fam_id=t["fam_id"], sdf_box=t["sdf_box"])
        y = tf.concat([t["y_node"][:, 0], t["y_q"][:, 0]], 1)
        cm = b["cmask"][0].astype(bool)
        # the node part is the full grid in raster order -> distance map applies directly
        mask = np.asarray(b["geom_node"][0][:, 7])
        if not np.any(mask):
            print(f"[{FAM}] geom ch7 (interior mask) is ALL ZERO — nothing to bin against")
            return
        dist = edge_distance(mask, dims).reshape(-1)
        e = (np.asarray(pred[0])[:Nn][:, cm] - np.asarray(y[0])[:Nn][:, cm]) ** 2
        r = np.asarray(y[0])[:Nn][:, cm] ** 2
        e, r = e.sum(-1), r.sum(-1)
        for lo, hi in bins:
            sel = (dist >= lo) & (dist < hi)
            acc[(lo, hi)][0] += float(e[sel].sum())
            acc[(lo, hi)][1] += float(r[sel].sum())
        n += 1
        if n >= N:
            break

    print(f"\n[{FAM}] error vs distance to the solid/fluid edge  ({n} val samples, node set)")
    print(f"{'cells from edge':>16s} {'rel-L2':>8s} {'share of total sq err':>22s}")
    tot = sum(v[0] for v in acc.values())
    for (lo, hi), (se, sr) in acc.items():
        if sr <= 0:
            continue
        lab = f"{lo}-{hi}" if hi < 999 else f">={lo}"
        print(f"{lab:>16s} {100*np.sqrt(se/sr):8.2f} {100*se/max(tot,1e-12):21.1f}%")
    print("DIAGWALL_OK")


if __name__ == "__main__":
    main()
