"""Physical-scale rel-L2 evaluation of a trained checkpoint (foundationv2).

The training/monitor metric is in NORMALIZED space (each channel / its own std), which weights
all channels equally and drops the DC magnitude — inflating rel-L2 for families with large-scale
or large-mean fields (mesh surface pressure, 3D CNS density/pressure). This script reports the
PHYSICAL rel-L2 used by the literature: predictions are denormalized (× per-channel std), so the
metric is dominated by the physically-dominant channels and includes the DC, matching how GINO/
Transolver (mesh) and DPOT/PDEBench (3D) report.

    python -m core.eval_physical --ckpt <npz> --families airfrans,shapenet_car,pdebench3d_... \
        --d 384 --dims2 48 --dims3 16 --encoder bin --rec_steps 2 --t_pad 1.1 --enc_n 16384 \
        --n_colloc 2048 --split test --n 32

Reports per-family PHYSICAL rel-L2 (joint over real channels) and, for reference, the normalized
rel-L2, so the gap between the two is explicit.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

from core.batching import bucketed_batches
from core.model import UniversalTPADA
from core.pretrain import load_ckpt
from data import loader
from data.registry import FAMILIES, NUM_SLOTS


def _relL2(pred, y, w):
    """pred,y (B,Q,S); w (B,1,S) channel mask/weight -> mean over batch of joint rel-L2."""
    num = tf.sqrt(tf.reduce_sum(w * tf.square(pred - y), axis=[1, 2]) + 1e-12)
    den = tf.sqrt(tf.reduce_sum(w * tf.square(y), axis=[1, 2]) + 1e-12)
    return float(tf.reduce_mean(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--d_w", type=int, default=16)
    ap.add_argument("--n_p", type=int, default=32)
    ap.add_argument("--dims2", type=int, default=48)
    ap.add_argument("--dims3", type=int, default=16)
    ap.add_argument("--encoder", default="bin")
    ap.add_argument("--rec_steps", type=int, default=2)
    ap.add_argument("--t_pad", type=float, default=1.1)
    ap.add_argument("--steady_mode", default="ada", choices=["ada", "field"])
    ap.add_argument("--enc_n", type=int, default=16384)
    ap.add_argument("--n_colloc", type=int, default=2048)
    a = ap.parse_args()

    box = {2: (a.dims2, a.dims2), 3: (a.dims3, a.dims3, a.dims3)}
    m = UniversalTPADA(box_cfgs=box, d=a.d, depth=a.depth, d_w=a.d_w, n_p=a.n_p, c_out=NUM_SLOTS,
                       t_final=a.t_pad, encoder=a.encoder, rec_steps=a.rec_steps)
    from core.train_step import GEOM_MAX
    F = 10 * NUM_SLOTS + 10 + GEOM_MAX
    for K in sorted(m.synth):                                          # build all vars
        cn = tf.zeros([1, a.enc_n, K]); ft = tf.zeros([1, a.enc_n, F]); cq = tf.zeros([1, 8, K])
        h = m.encode(cn, ft, K, roles=list(range(K))); _ = m.steady_field_at(h, K, cq)
        A, A_leg, _ = m.from_points(cn, ft, K, roles=list(range(K)), op_multihot=tf.ones([1, 13]))
        _ = m.point_traj_perquery_g(A, A_leg, K, cq, tf.zeros([1, 8, m.n_p]), tf.zeros([1, 8, NUM_SLOTS]))
    V = (m.core.trainable_weights + m.penc.trainable_weights + m.grid_proj.trainable_weights
         + m.banks.trainable_weights + m.steady.trainable_weights + m.dec.trainable_weights
         + m.rec_proj.trainable_weights + [g for s in m.synth.values() for g in (s.gains or [])]
         + [g for s in m.synth.values() for g in (s.gains_leg or [])])
    load_ckpt(V, a.ckpt)

    from core.train_step import build_feats  # noqa
    print(f"{'family':<40} {'physical':>10} {'normalized':>12}", flush=True)
    for fam in a.families.split(","):
        it = loader.stream([fam], a.split, t_in=10, n_colloc=a.n_colloc, seed=7, max_per_fam=a.n)
        phys, norm = [], []
        for b in bucketed_batches(it, 1, NUM_SLOTS, enc_n=a.enc_n, seed=7):
            K = int(b["K"]); roles = tuple(int(r) for r in b["roles"])
            cn = tf.constant(b["coords_node"]); ft = tf.constant(b["feats"]); cq = tf.constant(b["coords_q"])
            if b["steady"] and a.steady_mode == "field":
                hh = m.encode(cn, ft, K, roles=roles); pred = m.steady_field_at(hh, K, cq)
            else:                                                  # ADA relaxation (steady = t->inf endpoint)
                A, A_leg, _ = m.from_points(cn, ft, K, roles=roles, op_multihot=tf.constant(b["op_multihot"]))
                pred = m.point_traj_perquery(A, A_leg, K, cq, b["t_q"], tf.constant(b["ic_q"]))
            y = tf.constant(b["y_q"]); w = tf.constant(b["cmask"])[:, None]
            norm.append(_relL2(pred, y, w))
            s = tf.constant(b["scale"])[:, None]                      # (B,1,S) per-channel std
            phys.append(_relL2(pred * s, y * s, w))                   # denormalize -> physical
        if phys:
            print(f"{fam:<40} {np.mean(phys) * 100:>9.2f}% {np.mean(norm) * 100:>11.2f}%", flush=True)
    print("EVALPHYS_DONE", flush=True)


if __name__ == "__main__":
    main()
