"""Is the E16 MoE actually carrying capacity, or is it an expensive way to write one MLP?

diag_moe.py measured two things that sit badly together: the 16 experts are only 2.1-2.4% from
their own mean, and the top-1 gate share is 0.52 — the router picks two experts and then averages
them almost equally. If that is the whole story, the block is a dense MLP with jitter, the 92.4M
parameter count is largely fiction, and the capacity should be spent somewhere else.

Against that sits the measured gain from the upcycle itself (dense 18.83 -> E16 18.09 in v2;
Poisson 12.8 -> 4.5, pdearena_ns 28.1 -> 20.4, class-avg 13.9 -> 12.0 in v3), so the small
differences may still matter — a per-cell CHOICE of which two experts to average is a function a
single MLP cannot express at all, however close the experts are.

This decides it with no training. Three variants of the SAME checkpoint:
  as-is      top-2 blend, what we ship
  collapsed  every expert overwritten by the mean of the 16 -> exactly one dense MLP (the top-k
             blend of identical experts is that MLP, since the gate weights sum to 1)
  all-16     top_k = E, the full softmax blend over every expert at the same FLOPs we already pay
If `collapsed` matches `as-is`, the mixture is redundant. If `all-16` beats `as-is`, we are
throwing away capacity we are already computing.

  python diag_collapse.py <ckpt.npz> [fam1,fam2,...]
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

N = int(os.environ.get("DC_N", "6"))
# ONE VARIANT PER PROCESS. Running all three sweeps in a single process OOMs: the eager rollouts
# of the first sweep keep their allocations and the second one dies with ResourceExhausted on
# every family. The caller runs this three times and diffs the printed CLASS-AVG.
VARIANT = os.environ.get("DC_VARIANT", "as-is")
ENC = int(os.environ.get("DC_ENC", "4096"))     # eager E16 rollouts are memory-heavy: the first
COL = int(os.environ.get("DC_COL", "1024"))     # sweep OOMed every family after it at 8192/2048
KEYS = ("coords_node", "x_win", "fmask", "geom_node", "y_node", "ic_q", "coords_q",
        "geom_q", "y_q", "tstep_mask", "cmask", "op_multihot", "cond", "fam_id", "sdf_box")


def rollout(model, b):
    t = {k: tf.constant(b[k]) for k in KEYS}
    K, steady, dense2d = int(b["K"]), bool(b["steady"]), bool(b["dense2d"])
    roles = tuple(int(r) for r in b["roles"])
    gd = tuple(b.get("dims", ())) or None
    cm = t["cmask"]
    B, Nn = tf.shape(t["coords_node"])[0], tf.shape(t["coords_node"])[1]
    qall = tf.concat([t["coords_node"], t["coords_q"]], 1)
    gall = tf.concat([t["geom_node"], t["geom_q"]], 1)
    u_prev = tf.concat([t["x_win"][:, -1], t["ic_q"]], 1)
    win, fm = t["x_win"], t["fmask"]
    pg = tf.reshape(t["x_win"][:, -1], [B, gd[0], gd[1], model.S]) if dense2d else None
    acc = w = 0.0
    for k in range(int(b["k_seg"])):
        pred = model.step(t["coords_node"], win, fm, t["geom_node"], K, roles, t["cond"],
                          t["op_multihot"], qall, gall, u_prev, steady=steady,
                          prev_grid=pg, grid_dims=gd, fam_id=t["fam_id"], sdf_box=t["sdf_box"])
        kt = 0 if steady else k
        y = tf.concat([t["y_node"][:, kt], t["y_q"][:, kt]], 1)
        wk = float(k + 1) if steady else float(t["tstep_mask"][0, k].numpy())
        m = cm[:, None, :]
        num = tf.sqrt(tf.reduce_sum(m * tf.square(pred - y), axis=[1, 2]) + 1e-12)
        den = tf.sqrt(tf.reduce_sum(m * tf.square(y), axis=[1, 2]) + 1e-12)
        acc += wk * float(tf.reduce_mean(num / den))
        w += wk
        u_prev = pred
        pn = pred[:, :Nn]
        win = tf.concat([win[:, 1:], pn[:, None]], 1)
        fm = tf.concat([fm[:, 1:], tf.ones_like(fm[:, :1])], 1)
        if dense2d:
            pg = tf.reshape(pn, [B, gd[0], gd[1], model.S])
    return acc / max(w, 1e-6)


def sweep(model, fams):
    out = {}
    for name in fams:
        try:
            load_manifest(name)
        except Exception:
            continue
        rng = np.random.default_rng(1234)
        vals, n = [], 0
        try:
            for s in loader.units_for(name, "val"):
                K = int(s["K"])
                ex = make_example_ar(s, 10, 10, 4, COL, ENC, rng,
                                     box_dims=(64, 64) if K == 2 else (32, 32, 32))
                vals.append(rollout(model, stack_batch([ex])))
                n += 1
                if n >= N:
                    break
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}", flush=True)
            continue
        if vals:
            out[name] = float(np.mean(vals))
    return out


def main():
    fams = (sys.argv[2].split(",") if len(sys.argv) > 2
            else [f.name for f in pretrain_families()])
    model = V3Model(S=NUM_SLOTS, d=384, depth=8, d_w=32, n_p=8, d_cond=128, dims2=(64, 64),
                    dims3=(32, 32, 32), t_in=10, transport=True, n_fams=len(FAMILIES),
                    n_slices=512, moe_experts=16, moe_topk=2)
    model.warm_build()
    V = model.trainable_variables
    load_ckpt(V, sys.argv[1], d_w=32, names=[v.name for v in V])
    moes = [b[2] for b in model.blocks]
    pristine = {v.name: tf.identity(v) for v in V}

    del pristine
    if VARIANT == "all-16":                 # full softmax over every expert, same FLOPs as now
        for m in moes:
            m.k = m.E
    elif VARIANT == "collapsed":            # every expert := the mean -> exactly one dense MLP
        for m in moes:
            for layers in (m.eh, m.eo):
                for attr in ("kernel", "bias"):
                    ws = [getattr(l, attr) for l in layers
                          if getattr(l, attr, None) is not None]
                    if not ws:
                        continue
                    mu = tf.reduce_mean(tf.stack([tf.identity(w) for w in ws]), 0)
                    for w in ws:
                        w.assign(mu)

    r = sweep(model, fams)
    print(f"\n--- variant: {VARIANT} ---")
    for n in sorted(r):
        print(f"{n:42s} {100*r[n]:8.2f}")
    print(f"{'CLASS-AVG':42s} {100*np.mean(list(r.values())):8.2f}   [variant={VARIANT}]")
    print("\nif 'collapsed' ~ 'as-is', the 16 experts are one MLP and the capacity is elsewhere.")
    print("DIAGCOL_OK")


if __name__ == "__main__":
    main()
