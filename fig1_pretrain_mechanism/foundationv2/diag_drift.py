"""Is the error dynamics, or is it accumulated drift on top of a near-static field?

The wall-distance binning showed cfdbench's error sits entirely in the bulk (9.5%) while the
solid/fluid interface is fine (0.6-1.4%) — so it is not the geometry. cfdbench is developed
cavity/cylinder/tube flow, i.e. nearly steady, and our step is u_{k+1} = u_k + delta_k, so
copying the previous frame is the free default and any nonzero delta is a liability. Over a
10-step free-running rollout small deltas compound into drift, and the bulk is exactly where the
field is unconstrained.

Reports, per family: the PERSISTENCE baseline (predict no change at all) and our error at each
AR step. If persistence is already excellent and our error grows with step index, the problem is
drift, not dynamics — a different fix entirely from capacity or resolution.

  python diag_drift.py <ckpt.npz> [fam1,fam2,...]
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

FAMS = (sys.argv[2].split(",") if len(sys.argv) > 2
        else ["cfdbench", "incom_ns", "shallow_water", "pdearena_ns"])
N = int(os.environ.get("DD_N", "4"))
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "op_multihot", "cond", "fam_id", "sdf_box")


def rel(p, y, cm):
    m = cm[None, :]
    return float(np.sqrt((m * (p - y) ** 2).sum() + 1e-12) / np.sqrt((m * y ** 2).sum() + 1e-12))


def main():
    model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=32, n_p=8, d_cond=128, dims2=(64, 64),
                    dims3=(32, 32, 32), t_in=10, transport=True, n_fams=len(FAMILIES),
                    n_slices=512, moe_experts=16, moe_topk=2)
    model.warm_build()
    V = model.trainable_variables
    load_ckpt(V, sys.argv[1], d_w=32, names=[v.name for v in V])

    for fam in FAMS:
        if FAMILIES[fam].time == "steady":
            continue
        rng = np.random.default_rng(1234)
        pers, ours, cnt = [], [], 0
        for s in loader.units_for(fam, "val"):
            K = int(s["K"])
            ex = make_example_ar(s, 10, 10, 4, 1024, 4096, rng,
                                 box_dims=(64, 64) if K == 2 else (32, 32, 32))
            b = stack_batch([ex])
            t = {k: tf.constant(b[k]) for k in KEYS}
            cm = b["cmask"][0]
            gd = tuple(b.get("dims", ())) or None
            B = 1
            Nn = int(b["coords_node"].shape[1])
            qall = tf.concat([t["coords_node"], t["coords_q"]], 1)
            gall = tf.concat([t["geom_node"], t["geom_q"]], 1)
            anchor = np.concatenate([b["x_win"][0, -1], b["ic_q"][0]], 0)   # last observed frame
            u_prev = tf.constant(anchor[None])
            win, fm = t["x_win"], t["fmask"]
            pg = (tf.reshape(t["x_win"][:, -1], [B, gd[0], gd[1], model.S])
                  if b["dense2d"] else None)
            pk, ok = [], []
            for k in range(int(b["k_seg"])):
                if b["tstep_mask"][0, k] < 0.5:
                    break
                y = np.concatenate([b["y_node"][0, k], b["y_q"][0, k]], 0)
                pk.append(rel(anchor, y, cm))                        # persistence: never change
                pred = model.step(t["coords_node"], win, fm, t["geom_node"], K,
                                  tuple(int(r) for r in b["roles"]), t["cond"],
                                  t["op_multihot"], qall, gall, u_prev, steady=False,
                                  prev_grid=pg, grid_dims=gd, fam_id=t["fam_id"],
                                  sdf_box=t["sdf_box"])
                ok.append(rel(np.asarray(pred[0]), y, cm))
                u_prev = pred
                pn = pred[:, :Nn]
                win = tf.concat([win[:, 1:], pn[:, None]], 1)
                fm = tf.concat([fm[:, 1:], tf.ones_like(fm[:, :1])], 1)
                if b["dense2d"]:
                    pg = tf.reshape(pn, [B, gd[0], gd[1], model.S])
            pers.append(pk); ours.append(ok); cnt += 1
            if cnt >= N:
                break
        if not cnt:
            continue
        L = min(len(x) for x in pers)
        P = 100 * np.mean([x[:L] for x in pers], 0)
        O = 100 * np.mean([x[:L] for x in ours], 0)
        print(f"\n[{fam}]  {cnt} val samples, {L} scored AR steps")
        print("  step        " + " ".join(f"{i+1:>6d}" for i in range(L)))
        print("  persistence " + " ".join(f"{v:6.2f}" for v in P))
        print("  ours        " + " ".join(f"{v:6.2f}" for v in O))
        print(f"  mean: persistence {P.mean():.2f}%   ours {O.mean():.2f}%   "
              f"-> {'OURS WORSE THAN DOING NOTHING' if O.mean() > P.mean() else 'ours better'}")
    print("DIAGDRIFT_OK")


if __name__ == "__main__":
    main()
