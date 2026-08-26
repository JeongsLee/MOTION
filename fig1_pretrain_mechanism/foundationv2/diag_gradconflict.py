"""Do families actively FIGHT over the shared output head, or merely share it?

Slot occupancy alone explains almost nothing (contention vs headroom: Pearson +0.07, Spearman
+0.13 over 26 families; Wave-Layer has the second-least contended slot and the second-worst
headroom). Sharing a channel is only harmful if the families pull its weights in OPPOSING
directions — that is what "output splitting helps" actually means, and it is measurable.

For each family this takes the gradient of its own rollout loss with respect to the decoder's
shared output kernel, then reports the cosine between families. Negative cosine = every step one
family takes on that channel undoes part of another's; near-zero = they coexist; positive = they
want the same thing and sharing is free.

The case of interest is K=3: pdebench3d CNS writes the thermodynamic pressure of a periodic
turbulence volume to slot 4, shapenet_car / drivaernet_pressure write car-surface static pressure
to the SAME slot, and those five are the only families that exercise the 3D axial weights at all.
The report therefore breaks out the slot-4 column of the head separately from the whole kernel.

  python diag_gradconflict.py <ckpt.npz> [fam1,fam2,...]
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

FAMS = (sys.argv[2].split(",") if len(sys.argv) > 2 else
        ["drivaernet_pressure", "shapenet_car",
         "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08",
         "pdebench3d_cns_Turb_M1.0_Eta1e-08_Zeta1e-08",
         "pdebench3d_cns_Rand_M0.1_Eta1e-08_Zeta1e-08",
         "cfdbench", "pdearena_ns", "com_ns"])
N = int(os.environ.get("GC_N", "3"))
SLOT = int(os.environ.get("GC_SLOT", "4"))
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "op_multihot", "cond", "fam_id", "sdf_box")


def main():
    model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=32, n_p=8, d_cond=128, dims2=(64, 64),
                    dims3=(32, 32, 32), t_in=10, transport=True, n_fams=len(FAMILIES),
                    n_slices=512, moe_experts=16, moe_topk=2)
    model.warm_build()
    V = model.trainable_variables
    load_ckpt(V, sys.argv[1], d_w=32, names=[v.name for v in V])
    head = model.dec.o.kernel                                    # (hidden, S) shared output head

    G, Gs = {}, {}
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
                cm = t["cmask"][:, None, :]
                with tf.GradientTape() as tape:
                    pred = model.step(
                        t["coords_node"], t["x_win"], t["fmask"], t["geom_node"], K,
                        tuple(int(r) for r in b["roles"]), t["cond"], t["op_multihot"],
                        tf.concat([t["coords_node"], t["coords_q"]], 1),
                        tf.concat([t["geom_node"], t["geom_q"]], 1),
                        tf.concat([t["x_win"][:, -1], t["ic_q"]], 1),
                        steady=bool(b["steady"]), prev_grid=pg, grid_dims=gd,
                        fam_id=t["fam_id"], sdf_box=t["sdf_box"])
                    y = tf.concat([t["y_node"][:, 0], t["y_q"][:, 0]], 1)
                    num = tf.sqrt(tf.reduce_sum(cm * tf.square(pred - y), [1, 2]) + 1e-12)
                    den = tf.sqrt(tf.reduce_sum(cm * tf.square(y), [1, 2]) + 1e-12)
                    L = tf.reduce_mean(num / den)
                g = tape.gradient(L, head)
                if g is not None:
                    acc.append(np.asarray(g))
                n += 1
                if n >= N:
                    break
        except Exception as e:
            print(f"  [skip] {fam}: {type(e).__name__}: {e}", flush=True)
            continue
        if acc:
            m = np.mean(acc, 0)
            G[fam] = m.ravel()
            Gs[fam] = m[:, SLOT]

    def table(tag, D):
        fams = list(D)
        print(f"\n=== {tag} ===")
        print(f"{'':>26s} " + " ".join(f"{f[:11]:>12s}" for f in fams))
        for a in fams:
            row = []
            for b in fams:
                c = float(D[a] @ D[b] /
                          (np.linalg.norm(D[a]) * np.linalg.norm(D[b]) + 1e-12))
                row.append("      .     " if a == b else f"{c:12.3f}")
            print(f"{a[:26]:>26s} " + " ".join(row))

    table("whole shared output kernel", G)
    table(f"slot {SLOT} column only (contested 3D pressure channel)", Gs)
    fams = list(Gs)
    surf = [f for f in fams if f in ("drivaernet_pressure", "shapenet_car")]
    vol = [f for f in fams if f.startswith("pdebench3d")]
    if surf and vol:
        cs = [float(Gs[a] @ Gs[b] / (np.linalg.norm(Gs[a]) * np.linalg.norm(Gs[b]) + 1e-12))
              for a in surf for b in vol]
        print(f"\nsurface-mesh vs 3D-volume on slot {SLOT}: mean cosine {np.mean(cs):+.3f} "
              f"(min {np.min(cs):+.3f}, max {np.max(cs):+.3f})")
        print("negative => they undo each other and the channel should be split; "
              "~0 => sharing is not the problem.")
    print("DIAGGC_OK")


if __name__ == "__main__":
    main()
