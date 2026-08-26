"""Bucketed multi-family batching for the universal training loop.

The encoder/synthesis need aligned shapes within a micro-batch, but families differ
in K, node count N, and collocation Q. Strategy: bucket examples by (K, N_node) — the
two axes that must match to stack — and within a bucket stack B examples into arrays.
Q (collocation) is fixed by the loader (n_colloc), so it aligns automatically.

Grid families share N per (family, resolution); mesh families vary N per unit, so mesh
units are optionally resampled to a fixed N_node so they bucket together. This keeps one
weight set training across everything without ragged tensors.

Yields batch dicts of stacked numpy arrays consumed by core.train_step.batched_loss.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from core.train_step import build_feats


def _resample_nodes(ex, n_target, rng):
    """Subsample the ENCODER nodes to exactly n_target (grid OR mesh) so the encoder input is
    bounded regardless of native resolution — critical for 3D grids (128^3=2M cells). The
    collocation targets (coords_q, y_q) are UNTOUCHED, so the loss keeps native-resolution
    supervision. Sampling with replacement guarantees the fixed count even for small meshes."""
    N = ex["coords_node"].shape[0]
    if N == n_target:
        return ex
    idx = (rng.integers(0, N, size=n_target) if N < n_target
           else rng.choice(N, size=n_target, replace=False))
    ex = dict(ex)
    ex["coords_node"] = ex["coords_node"][idx]
    ex["x_frames"] = ex["x_frames"][:, idx]
    if ex.get("geom") is not None:
        ex["geom"] = np.asarray(ex["geom"])[idx]
    return ex


def _stack(examples, S):
    """Stack a list of same-(K,N) examples into a batch dict of arrays."""
    b = {}
    b["K"] = int(examples[0]["K"])
    b["roles"] = [int(r) for r in examples[0]["roles"]]
    b["coords_node"] = np.stack([e["coords_node"] for e in examples])          # (B,N,K)
    b["feats"] = np.concatenate([build_feats(e, S) for e in examples], 0)      # (B,N,F)
    b["coords_q"] = np.stack([e["coords_q"] for e in examples])                # (B,Q,K)
    b["geom_q"] = np.stack([e["geom_q"] for e in examples])                    # (B,Q,GEOM_W) per-query geom
    b["t_q"] = np.stack([e["t_q"] for e in examples])                          # (B,Q)
    b["ic_q"] = np.stack([e["ic_q"] for e in examples])                        # (B,Q,S)
    b["y_q"] = np.stack([e["y_q"] for e in examples])                          # (B,Q,S)
    b["cmask"] = np.stack([e["cmask"] for e in examples])                      # (B,S)
    b["op_multihot"] = np.stack([e["op_multihot"] for e in examples])          # (B, |vocab|)
    b["scale"] = np.stack([e["scale"] for e in examples])                      # (B,S) denorm -> physical
    b["families"] = [e["family"] for e in examples]
    b["steady"] = bool(examples[0]["steady"])                                  # homogeneous per bucket
    return b


def bucketed_batches(example_iter, batch_size, S, enc_n=8192, seed=0):
    """Subsample EVERY example's encoder nodes to `enc_n` (grid + mesh alike) and bucket by K.
    Bounded encoder input -> 3D grids (2M cells) and huge meshes (airfrans 177k) both fit;
    buckets reduce to (K,) so all same-K families stack together. Collocation targets untouched."""
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for ex in example_iter:
        ex = _resample_nodes(ex, enc_n, rng)
        key = (int(ex["K"]), bool(ex["steady"]))       # homogeneous mode per batch (steady/unsteady)
        buckets[key].append(ex)
        if len(buckets[key]) == batch_size:
            yield _stack(buckets[key], S)
            buckets[key] = []
    for key, rem in buckets.items():                                          # flush partials
        if rem:
            yield _stack(rem, S)
