"""Same predictions, two metrics: normalized-space (what we log) vs physical-space (what
PROSE/BCAT report). Both are computed from ONE forward pass per sample, so sampling density
cancels and only the metric definition differs.

Why this matters: the loader divides every slot by its own std, and a channel-JOINT relative L2
is not invariant to that — it silently reweights the channels. Physically cfdbench's streamwise
velocity dwarfs the cross-flow, so in physical units the easy channel fills the denominator while
in normalized units the hard one gets equal footing. Until this is quantified our per-family
numbers cannot be placed next to published ones.

  python eval_phys.py <ckpt.npz>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pretrain import load_ckpt                      # noqa: E402
from data import loader                                  # noqa: E402
from data.registry import FAMILIES, NUM_SLOTS, pretrain_families    # noqa: E402
from data.splits import load_manifest                    # noqa: E402
from v3.data_ar import make_example_ar, stack_batch      # noqa: E402
from v3.model import V3Model                             # noqa: E402

N = int(os.environ.get("PH_N", "6"))
ENC = int(os.environ.get("PH_ENC", "4096"))
COL = int(os.environ.get("PH_COL", "1024"))
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "op_multihot", "cond", "fam_id", "sdf_box")


def both_metrics(model, b):
    """One free-running rollout; returns (normalized, physical) joint rel-L2 for this sample."""
    t = {k: tf.constant(b[k]) for k in KEYS}
    K, steady = int(b["K"]), bool(b["steady"])
    dense2d, k_seg = bool(b["dense2d"]), int(b["k_seg"])
    roles = tuple(int(r) for r in b["roles"])
    gd = tuple(b.get("dims", ())) or None
    sc = tf.constant(b["scale"])[:, None, :]
    cm = t["cmask"]
    B = tf.shape(t["coords_node"])[0]
    Nn = tf.shape(t["coords_node"])[1]
    qall = tf.concat([t["coords_node"], t["coords_q"]], 1)
    gall = tf.concat([t["geom_node"], t["geom_q"]], 1)
    u_prev = tf.concat([t["x_win"][:, -1], t["ic_q"]], 1)
    win, fm = t["x_win"], t["fmask"]
    pg = (tf.reshape(t["x_win"][:, -1], [B, gd[0], gd[1], model.S]) if dense2d else None)
    accn = accp = w = 0.0
    for k in range(k_seg):
        pred = model.step(t["coords_node"], win, fm, t["geom_node"], K, roles, t["cond"],
                          t["op_multihot"], qall, gall, u_prev, steady=steady,
                          prev_grid=pg, grid_dims=gd, fam_id=t["fam_id"], sdf_box=t["sdf_box"])
        kt = 0 if steady else k
        y = tf.concat([t["y_node"][:, kt], t["y_q"][:, kt]], 1)
        wk = (float(k + 1) if steady else float(t["tstep_mask"][0, k].numpy()))
        for tag, (p_, y_) in (("n", (pred, y)), ("p", (pred * sc, y * sc))):
            m = cm[:, None, :]
            num = tf.sqrt(tf.reduce_sum(m * tf.square(p_ - y_), axis=[1, 2]) + 1e-12)
            den = tf.sqrt(tf.reduce_sum(m * tf.square(y_), axis=[1, 2]) + 1e-12)
            v = float(tf.reduce_mean(num / den))
            if tag == "n":
                accn += wk * v
            else:
                accp += wk * v
        w += wk
        u_prev = pred
        pn = pred[:, :Nn]
        win = tf.concat([win[:, 1:], pn[:, None]], 1)
        fm = tf.concat([fm[:, 1:], tf.ones_like(fm[:, :1])], 1)
        if dense2d:
            pg = tf.reshape(pn, [B, gd[0], gd[1], model.S])
    w = max(w, 1e-6)
    return accn / w, accp / w


def main():
    model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=32, n_p=8, d_cond=128, dims2=(64, 64),
                    dims3=(32, 32, 32), t_in=10, transport=True, n_fams=len(FAMILIES),
                    n_slices=512, moe_experts=16, moe_topk=2)
    model.warm_build()
    V = model.trainable_variables
    load_ckpt(V, sys.argv[1], d_w=32, names=[v.name for v in V])

    print(f"\n{'family':40s} {'normalized':>11s} {'physical':>10s} {'ratio':>7s}")
    rows = []
    for f in pretrain_families():
        try:
            load_manifest(f.name)
        except Exception:
            continue
        rng = np.random.default_rng(1234)
        nn, pp, c = 0.0, 0.0, 0
        try:
            for s in loader.units_for(f.name, "val"):
                K = int(s["K"])
                ex = make_example_ar(s, 10, 10, 4, COL, ENC, rng,
                                     box_dims=(64, 64) if K == 2 else (32, 32, 32))
                a, b_ = both_metrics(model, stack_batch([ex]))
                nn += a; pp += b_; c += 1
                if c >= N:
                    break
        except Exception as e:
            print(f"  [skip] {f.name}: {type(e).__name__}", flush=True)
            continue
        if c:
            nn, pp = nn / c, pp / c
            rows.append((f.name, nn, pp))
            print(f"{f.name:40s} {100*nn:11.2f} {100*pp:10.2f} {pp/max(nn,1e-9):7.2f}", flush=True)
    if rows:
        mn = np.mean([r[1] for r in rows]); mp = np.mean([r[2] for r in rows])
        print(f"\n{'CLASS-AVG':40s} {100*mn:11.2f} {100*mp:10.2f} {mp/mn:7.2f}")
    print("PHYSEVAL_OK")


if __name__ == "__main__":
    main()
