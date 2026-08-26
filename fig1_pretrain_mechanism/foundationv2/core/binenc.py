"""Cell-binning point encoder (GINO-style local aggregation) — O(N), memory-flat, scalable.

The full cross-attention encoder is O(box_nodes x N) — it OOMs when the box or point count
grows, and it undersamples fine geometry (airfrans: 177k points -> 4096 sampled = 2.3%, the
airfoil boundary is lost). This encoder instead SCATTERS each input point into its latent-box
cell and mean-aggregates features there: cost is O(N), independent of box size, and local
geometry is preserved (points near a cell aggregate there). A log-count coverage channel tells
the core where information exists. Scales to enc_n=16k+ and box 24^3 on one GPU.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .pointio import CoordEncoder

GEOM_DIM = 8          # must match core.train_step.GEOM_MAX (last GEOM_DIM feat cols = geom channels)
# canonical geom layout: [0=SDF, 1=BL mask, 2=nx, 3=ny, 4=nz, 5=mean curvature, 6=gauss curvature,
# 7=interior/material mask]. Adapters fill what they have (zeros elsewhere); loader builds ch1 from SDF.


class BinEncoder(tf.keras.layers.Layer):
    def __init__(self, d, n_slices=0, name="binenc", **kw):
        super().__init__(name=name, **kw)
        self.d = d
        self.coord = CoordEncoder(d, name=f"{name}_coord")
        self.fproj = tf.keras.layers.Dense(d, name=f"{name}_f")
        self.out = tf.keras.layers.Dense(d, name=f"{name}_o")
        # SLICE ATTENTION (Transolver-style physics attention, O(N*M)): the cell-binning
        # scatter gives points NO point<->point interaction before the coarse box — surface
        # physics (drivaernet/shapenet pressure: curvature x tangential-flow coupling) lives
        # exactly there. Points soft-assign to M learnable slices, slices self-attend, the
        # result redistributes through a ZERO-INIT gate (inert at load -> warm-safe).
        self.n_slices = int(n_slices)
        if self.n_slices:
            M = self.n_slices
            self.sl_assign = tf.keras.layers.Dense(M, name=f"{name}_slw")
            self.sl_mha = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=max(d // 4, 8), output_shape=d, name=f"{name}_slmha")
            self.sl_gate = tf.keras.layers.Dense(d, use_bias=False, kernel_initializer="zeros",
                                                 name=f"{name}_slgate")
        # UNIVERSAL geometry-conditioning branch (SDF / boundary-layer mask / surface normal+area /
        # elliptic source — whatever a family populates in its geom slots; zeros where absent). A
        # dedicated MLP processes the geom vector and enters via a ZERO-INIT gate so it is inert at
        # load (clean warm-start) and only families WITH geometry (mesh / walled grids) are affected.
        self.geom_mlp = tf.keras.layers.Dense(d, activation="gelu", name=f"{name}_gm")
        self.geom_gate = tf.keras.layers.Dense(d, use_bias=False, kernel_initializer="zeros",
                                               name=f"{name}_gg")

    def call(self, coords, feats, dims, roles=None):
        """coords (B,N,K) in [0,1]^K, feats (B,N,F) -> box (B, *dims, d). Scatter-mean per cell."""
        B = tf.shape(coords)[0]
        N = tf.shape(coords)[1]
        K = len(dims)
        ncell = int(np.prod(dims))
        # per-point token (+ zero-init geometry branch on the last GEOM_DIM feat cols)
        tok = self.coord(coords, roles=roles) + self.fproj(feats)          # (B,N,d)
        tok = tok + self.geom_gate(self.geom_mlp(feats[..., -GEOM_DIM:]))  # 0 at load -> warm-safe
        slices = None
        if self.n_slices:
            w = tf.nn.softmax(self.sl_assign(tok), axis=-1)                # (B,N,M) point->slice
            den = tf.reduce_sum(w, axis=1, keepdims=True) + 1e-6           # (B,1,M)
            sl = tf.einsum("bnm,bnd->bmd", w, tok) / tf.transpose(den, [0, 2, 1])  # (B,M,d)
            sl = self.sl_mha(sl, sl)                                       # slice<->slice physics attn
            tok = tok + self.sl_gate(tf.einsum("bnm,bmd->bnd", w, sl))     # zero-init redistribute
            # SURFACE-MANIFOLD LATENT: the slice tokens are ALSO returned so the decoder can read
            # them directly. Scattering them into the volumetric box would undo the point-level
            # resolution they carry — a 2-manifold fills only ~3% of a 3D box, so the box path
            # cannot represent surface detail no matter how good the tokens are. Their mean
            # position (slice centroid) lets the decoder do distance-aware attention.
            cen = tf.einsum("bnm,bnk->bmk", w, coords) / tf.transpose(den, [0, 2, 1])  # (B,M,K)
            slices = (sl, cen)
        # cell index per point (clamp to grid)
        idx = tf.zeros([B, N], tf.int32)
        stride = 1
        for a in reversed(range(K)):
            ci = tf.cast(tf.floor(coords[..., a] * dims[a]), tf.int32)
            ci = tf.clip_by_value(ci, 0, dims[a] - 1)
            idx = idx + ci * stride
            stride *= dims[a]
        boff = tf.range(B)[:, None] * ncell                                # flatten batch into cells
        flat_idx = tf.reshape(idx + boff, [-1])                            # (B*N,)
        flat_tok = tf.reshape(tok, [-1, self.d])
        summ = tf.math.unsorted_segment_sum(flat_tok, flat_idx, B * ncell)  # (B*ncell, d)
        cnt = tf.math.unsorted_segment_sum(                                 # (B*ncell,1); dtype
            tf.ones([tf.shape(flat_idx)[0], 1], summ.dtype), flat_idx, B * ncell)  # follows summ

        mean = summ / tf.maximum(cnt, 1.0)
        cov = tf.math.log1p(cnt)                                           # coverage: log point count
        h = self.out(tf.concat([mean, cov], -1))                          # (B*ncell, d)
        # geom box (scatter-mean of the GEOM_DIM geom cols onto the box) for the boundary-layer bank:
        # ch0 = SDF -> wall-normal n = grad(SDF), near-wall weight = exp(-|SDF|). Zeros where no wall.
        gsum = tf.math.unsorted_segment_sum(tf.reshape(feats[..., -GEOM_DIM:], [-1, GEOM_DIM]),
                                            flat_idx, B * ncell)
        gcnt = tf.maximum(tf.cast(cnt, gsum.dtype), 1.0)     # geometry keeps its own dtype: the
        geom_box = tf.reshape(gsum / gcnt,                   # BL bank differences the SDF, and the
                              tf.concat([[B], dims, [GEOM_DIM]], 0))   # token count is bf16 above
        return tf.reshape(h, tf.concat([[B], dims, [self.d]], 0)), geom_box, slices
