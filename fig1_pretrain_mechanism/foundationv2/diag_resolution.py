"""What is the BEST error the latent box could ever produce, per family?

Every prediction leaves the core as `interp_box(z, coords_q, dims)` -- a multilinear read of a
dims-resolution grid (64^2 in 2D, 32^3 in 3D). That operator has a fixed image: fields whose
sub-cell variation is affine. If a family's one-step increment lives outside that image, no amount
of training, capacity or depth reaches it, and the residual is a hard floor under our reported
error. Families whose solution is smooth but must be right EVERYWHERE (cfdbench's developed
channel flow, drivaernet's surface pressure) are exactly the ones a coarse box would cap, while
families dominated by large-scale structure (shallow water) would not notice.

This measures that floor directly, with no checkpoint involved. For each family it takes the true
increment  D = y - u_prev  at native points and fits the box VALUES themselves by gradient descent
-- i.e. it hands the architecture a perfect oracle for the latent state and asks what comes out.
The reported number is  ||D - interp_box(box*)|| / ||y||, on the same channels and denominator our
eval uses, so it is directly comparable to the per-family numbers we log. Sweeping the resolution
shows what we would buy by making the box finer.

  python diag_resolution.py [fam1,fam2,...]
env: DR_N (samples/family, default 3), DR_STEPS (fit iterations, default 600)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import loader                                  # noqa: E402
from data.registry import FAMILIES, pretrain_families     # noqa: E402
from data.splits import load_manifest                     # noqa: E402
from v3.data_ar import make_example_ar, stack_batch        # noqa: E402
from v3.model import interp_box                            # noqa: E402

N = int(os.environ.get("DR_N", "3"))
STEPS = int(os.environ.get("DR_STEPS", "600"))
RES2 = tuple(int(x) for x in os.environ.get("DR_RES2", "32,64,128").split(","))
RES3 = tuple(int(x) for x in os.environ.get("DR_RES3", "16,32,64").split(","))
KEYS = ("coords_node", "x_win", "y_node", "ic_q", "coords_q", "y_q", "cmask")


def best_box_error(coords, target, ref, dims, steps=STEPS):
    """Fit box values to reproduce `target` at `coords`; return ||target-fit||/||ref||.

    The box is the only thing being optimised, so the result is the exact image of the multilinear
    readout -- an oracle upper bound on what the trained core could ever emit at this resolution.
    """
    C = int(target.shape[-1])
    box = tf.Variable(tf.zeros([1] + list(dims) + [C]), dtype=tf.float32)
    q = tf.constant(coords[None])
    y = tf.constant(target[None])
    opt = tf.keras.optimizers.Adam(0.05)
    den = float(np.sqrt((ref ** 2).sum()) + 1e-12)

    @tf.function(jit_compile=False)
    def one():
        with tf.GradientTape() as tp:
            r = interp_box(box, q, tuple(dims)) - y
            loss = tf.reduce_sum(tf.square(r))
        opt.apply_gradients([(tp.gradient(loss, box), box)])
        return loss

    prev = None
    for i in range(steps):
        L = float(one())
        if i > 50 and prev is not None and abs(prev - L) < 1e-9 * max(prev, 1e-9):
            break
        prev = L
    return float(np.sqrt(L)) / den


def main():
    fams = (sys.argv[1].split(",") if len(sys.argv) > 1
            else [f.name for f in pretrain_families()])
    print(f"\nORACLE box error: best possible interp_box output vs the true increment"
          f"   ({N} val samples, {STEPS} fit steps)\n")
    hdr2 = " ".join(f"{r}^2".rjust(8) for r in RES2)
    hdr3 = " ".join(f"{r}^3".rjust(8) for r in RES3)
    print(f"{'family':40s} {'K':>2s} {'native':>10s} {'persist':>8s}   "
          f"box-> {hdr2} / {hdr3}")
    rows = []
    for name in fams:
        if name not in FAMILIES:
            continue
        try:
            load_manifest(name)
        except Exception:
            continue
        rng = np.random.default_rng(1234)
        got, K = [], None
        native, pers = "", []
        try:
            for s in loader.units_for(name, "val"):
                K = int(s["K"])
                res = RES2 if K == 2 else RES3
                ex = make_example_ar(s, 10, 10, 4, 4096, 16384, rng,
                                     box_dims=(64, 64) if K == 2 else (32, 32, 32))
                b = stack_batch([ex])
                cm = b["cmask"][0].astype(bool)
                steady = bool(b["steady"])
                co = np.concatenate([b["coords_node"][0], b["coords_q"][0]], 0)
                y = np.concatenate([b["y_node"][0, 0], b["y_q"][0, 0]], 0)[:, cm]
                if steady:
                    tgt = y                                  # steady: the field IS the output
                else:
                    up = np.concatenate([b["x_win"][0, -1], b["ic_q"][0]], 0)[:, cm]
                    tgt = y - up                             # the increment the box must carry
                native = "x".join(str(int(v)) for v in b.get("dims", ())) or f"{co.shape[0]}pts"
                pers.append(float(np.sqrt((tgt ** 2).sum()) / (np.sqrt((y ** 2).sum()) + 1e-12)))
                got.append([best_box_error(co, tgt, y, (r,) * K) for r in res])
                if len(got) >= N:
                    break
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}", flush=True)
            continue
        if not got:
            continue
        m = 100 * np.mean(got, 0)
        pad = "" if K == 2 else " " * (9 * len(RES2))
        cells = " ".join(f"{v:8.2f}" for v in m)
        print(f"{name:40s} {K:2d} {native:>10s} {100*np.mean(pers):8.2f}   "
              f"box-> {pad}{cells}", flush=True)
        rows.append((name, K, m))
    print("\nreading: the box column is a FLOOR. our logged per-family error cannot go below it.")
    print("DIAGRES_OK")


if __name__ == "__main__":
    main()
