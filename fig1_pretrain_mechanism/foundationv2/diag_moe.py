"""Is the E16 router actually using its experts, or has it collapsed?

The sparse upcycle seeds all 16 experts from ONE dense MLP and starts the router near zero, so at
load every expert is identical and the router has no signal. Two failure modes follow, and they
have opposite consequences:
  * COLLAPSE — the router concentrates on a couple of experts, so the 92.4M parameter count is a
    fiction (the extra experts never receive gradient) and the E16 gain we measured would have to
    come from somewhere else.
  * NO SPECIALISATION — the router stays flat and top-k picks are near-arbitrary among identical
    experts; the block is then a 2-way average of near-copies, i.e. still effectively dense.
Neither shows up in the loss curve. This also gates the sparse dispatch in axops: it gives each
expert a capacity of cf*k/E of the cells, so it is exact only while max(load) stays under that.

Reports per block: load per expert (should be ~k/E = 0.125), its max, the normalised entropy of
the load (1.0 = perfectly balanced, 0.0 = one expert), the top-1 gate share (0.5 = the two
experts blend equally, 1.0 = hard routing), and the pairwise divergence of the expert weights
from their mean (0 = still identical copies = no specialisation).

  python diag_moe.py <ckpt.npz> [fam1,fam2,...]
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
        else ["cfdbench", "pdearena_ns", "shallow_water", "drivaernet_pressure",
              "pdebench3d_cns_Rand_M1.0_Eta1e-08_Zeta1e-08"])
N = int(os.environ.get("DM_N", "2"))
E, KTOP, CF = 16, 2, 2.0
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "op_multihot", "cond", "fam_id", "sdf_box")


def main():
    model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=32, n_p=8, d_cond=128, dims2=(64, 64),
                    dims3=(32, 32, 32), t_in=10, transport=True, n_fams=len(FAMILIES),
                    n_slices=512, moe_experts=E, moe_topk=KTOP)
    model.warm_build()
    V = model.trainable_variables
    load_ckpt(V, sys.argv[1], d_w=32, names=[v.name for v in V])
    moes = [b[2] for b in model.blocks]

    # --- how far have the experts drifted from each other? (weights only, no data needed) -----
    print("\nEXPERT DIVERGENCE (upcycle seeds them identical; 0 = still copies)")
    print(f"{'block':>6s} {'rel spread of h kernels':>24s} {'o kernels':>12s}")
    for bi, m in enumerate(moes):
        for tag, layers in (("h", m.eh), ("o", m.eo)):
            W = np.stack([l.kernel.numpy() for l in layers])       # (E,in,out)
            mu = W.mean(0, keepdims=True)
            s = float(np.linalg.norm(W - mu) / (np.linalg.norm(mu) * np.sqrt(len(layers))))
            if tag == "h":
                hh = s
        print(f"{bi:6d} {hh:24.4f} {s:12.4f}")

    # --- and what does the router actually do on real data? ------------------------------------
    cap = CF * KTOP / E
    print(f"\nROUTER LOAD  (uniform = {KTOP/E:.3f};  sparse-dispatch capacity = {cap:.3f})")
    rng = np.random.default_rng(1234)
    for fam in FAMS:
        if fam not in FAMILIES:
            continue
        acc = [[] for _ in moes]
        gws, n = [[] for _ in moes], 0
        try:
            for s in loader.units_for(fam, "val"):
                K = int(s["K"])
                ex = make_example_ar(s, 10, 10, 4, 1024, 4096, rng,
                                     box_dims=(64, 64) if K == 2 else (32, 32, 32))
                b = stack_batch([ex])
                t = {k: tf.constant(b[k]) for k in KEYS}
                qall = tf.concat([t["coords_node"], t["coords_q"]], 1)
                gall = tf.concat([t["geom_node"], t["geom_q"]], 1)
                gd = tuple(b.get("dims", ())) or None
                pg = (tf.reshape(t["x_win"][:, -1], [1, gd[0], gd[1], model.S])
                      if b["dense2d"] else None)
                model.step(t["coords_node"], t["x_win"], t["fmask"], t["geom_node"], K,
                           tuple(int(r) for r in b["roles"]), t["cond"], t["op_multihot"],
                           qall, gall, tf.concat([t["x_win"][:, -1], t["ic_q"]], 1),
                           steady=bool(b["steady"]), prev_grid=pg, grid_dims=gd,
                           fam_id=t["fam_id"], sdf_box=t["sdf_box"])
                for i, m in enumerate(moes):
                    acc[i].append(np.asarray(m.load))
                    gws[i].append(float(m.gate_w))
                n += 1
                if n >= N:
                    break
        except Exception as e:
            print(f"  [skip] {fam}: {type(e).__name__}: {e}")
            continue
        if not n:
            continue
        print(f"\n[{fam}]  {n} samples")
        print(f"{'block':>6s} {'max load':>9s} {'min':>7s} {'entropy':>8s} {'top1 gate':>10s} "
              f"{'over cap':>9s}  load per expert")
        for i in range(len(moes)):
            L = np.mean(acc[i], 0)
            p = L / max(L.sum(), 1e-9)
            H = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
            bar = " ".join(f"{v:.2f}" for v in L)
            print(f"{i:6d} {L.max():9.3f} {L.min():7.3f} {H:8.3f} {np.mean(gws[i]):10.3f} "
                  f"{int((L > cap).sum()):9d}  {bar}")
    print("\nDIAGMOE_OK")


if __name__ == "__main__":
    main()
