"""Universal training step: unified loader example -> UniversalTPADA loss (foundationv2 P2/P4).

Everything goes through the POINT path (dimension-uniform, grid = its cell-centers): the
encoder cross-attends input nodes, TP-ADA synthesizes at the collocation queries, and the
loss is a masked rel-L2 at (coords_q, t_q) against y_q with the hard-IC anchor ic_q.

  feats_in = [x_frames flattened over observed time] (+ geom broadcast for steady/mesh)
  A        = model.from_points(coords_node, feats_in, K)
  pred     = model.point_traj(A, ic=None, K, coords_q, t_q, ic_at_pts=ic_q)   # (B,1,Q,S)
  loss     = cmask-weighted rel-L2( pred, y_q )

Batching: examples are grouped by (K, N_node, family) so shapes align — in practice one
family (fixed K) per micro-batch, which also matches the memory profile. This step takes a
single example (B=1) for the smoke; a bucketed sampler stacks same-shape examples for scale.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


T_IN_MAX = 10          # observed-frame slots (zero-padded, validity-masked)
GEOM_MAX = 8           # [SDF, BL mask, nx, ny, nz, mean curv, gauss curv, interior/material mask]


def build_feats(ex, S=None):
    """FIXED-width (1, N, F) encoder features so ONE weight set serves every family:
    F = T_IN_MAX*S + T_IN_MAX (frame-present mask) + GEOM_MAX. Observed frames are
    right-aligned into the last slots and zero-padded; steady -> all frames absent
    (mask 0) so coords + geom + conditioning carry the signal (no field leakage)."""
    x = ex["x_frames"]                                    # (ti, N, S)
    ti, N, Sx = x.shape
    S = Sx if S is None else S
    frames = np.zeros((T_IN_MAX, N, S), np.float32)
    fmask = np.zeros(T_IN_MAX, np.float32)
    use = min(ti, T_IN_MAX)
    if not ex["steady"] and use > 0:
        frames[T_IN_MAX - use:] = x[-use:]                # right-align newest frames
        fmask[T_IN_MAX - use:] = 1.0
    feats = np.transpose(frames, (1, 0, 2)).reshape(N, T_IN_MAX * S)
    feats = np.concatenate([feats, np.tile(fmask, (N, 1))], -1)          # (N, T*S + T)
    geom = np.zeros((N, GEOM_MAX), np.float32)
    if ex.get("geom") is not None:
        g = np.asarray(ex["geom"], np.float32)
        if g.ndim == 1:
            g = g[:, None]
        geom[:, :min(g.shape[1], GEOM_MAX)] = g[:, :GEOM_MAX]
    feats = np.concatenate([feats, geom], -1)
    return feats[None].astype(np.float32)                 # (1, N, F) fixed width


def masked_rel_l2(pred, y, cmask):
    """pred,y: (Q,S); cmask: (S,). rel-L2 over the real slots only."""
    w = cmask[None]
    num = tf.sqrt(tf.reduce_sum(w * tf.square(pred - y)) + 1e-12)
    den = tf.sqrt(tf.reduce_sum(w * tf.square(y)) + 1e-12)
    return num / den


def batched_loss(model, batch):
    """Masked rel-L2 over a stacked batch (B>1) of ONE (K, steady) bucket. Branches:
    steady -> direct elliptic field head (no time); unsteady -> TP-ADA tendency trajectory."""
    K = int(batch["K"])
    roles = batch["roles"]
    coords_node = tf.constant(batch["coords_node"]); feats = tf.constant(batch["feats"])
    coords_q = tf.constant(batch["coords_q"])
    if batch.get("steady"):
        h = model.encode(coords_node, feats, K, roles=roles)
        pred = model.steady_field_at(h, K, coords_q)                            # (B,Q,S)
    else:
        A, A_leg, _ = model.from_points(coords_node, feats, K, roles=roles,
                                        op_multihot=tf.constant(batch["op_multihot"]))
        pred = model.point_traj_perquery(A, A_leg, K, coords_q,
                                         batch["t_q"], tf.constant(batch["ic_q"]))
    y = tf.constant(batch["y_q"]); w = tf.constant(batch["cmask"])[:, None]      # (B,1,S)
    num = tf.sqrt(tf.reduce_sum(w * tf.square(pred - y), axis=[1, 2]) + 1e-12)
    den = tf.sqrt(tf.reduce_sum(w * tf.square(y), axis=[1, 2]) + 1e-12)
    return tf.reduce_mean(num / den)


def loss_on_example(model, ex):
    """Forward + scalar loss for one example (B=1). Returns (loss, pred)."""
    K = int(ex["K"])
    coords_node = tf.constant(ex["coords_node"][None])     # (1,N,K)
    feats = tf.constant(build_feats(ex))                   # (1,N,F)
    roles = [int(r) for r in ex["roles"]]
    A, A_leg, _ = model.from_points(coords_node, feats, K, roles=roles)

    coords_q = tf.constant(ex["coords_q"][None])           # (1,Q,K)
    pred = model.point_traj_perquery(A, A_leg, K, coords_q, ex["t_q"][None],
                                     ic_at_pts=tf.constant(ex["ic_q"][None]))  # (1,Q,S)
    y = tf.constant(ex["y_q"])
    loss = masked_rel_l2(pred[0], y, tf.constant(ex["cmask"]))
    return loss, pred
