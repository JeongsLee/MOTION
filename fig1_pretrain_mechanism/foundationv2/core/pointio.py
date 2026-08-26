"""Point I/O: mesh/point-cloud encoder onto the latent box + continuous decoder.

Encoder (GINO/GNOT-style, prototype): latent-box nodes are learned queries that
cross-attend to the input points; coordinates enter through a SHARED per-coordinate
Fourier feature encoder + axis-role embedding (K-agnostic weights). A point-density
"coverage" channel tells the core where information exists. Grid inputs bypass this
(frames-as-channels on the box, the degenerate case).

Decoder: bias-free pointwise MLP on the analytic latent trajectory value z(y, t)
(hard IC preserved: z(·, 0) = 0 and MLP(0) = 0), + the IC anchor interpolated
through the same band-limited representation (synthesis.field_at).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

MAX_ROLES = 4


class CoordEncoder(tf.keras.layers.Layer):
    """Shared per-coordinate Fourier features + axis-role embedding, summed over
    axes -> (B, Q, d). Same weights for any K."""

    def __init__(self, d, n_freq=16, name="coord", **kw):
        super().__init__(name=name, **kw)
        self.freq = tf.constant(2.0 ** np.arange(n_freq), tf.float32) * np.pi
        self.proj = tf.keras.layers.Dense(d, name=f"{name}_p")
        self.role = self.add_weight(name=f"{name}_role", shape=(MAX_ROLES, 2 * n_freq),
                                    initializer="zeros", trainable=True)

    def call(self, pts, roles=None):
        K = int(pts.shape[-1])
        roles = roles or list(range(K))
        acc = None
        for a in range(K):
            ph = tf.cast(pts[..., a:a + 1], tf.float32) * self.freq          # (B,Q,F) in f32:
            ff = tf.cast(tf.concat([tf.sin(ph), tf.cos(ph)], -1),             # 2^15*pi*x needs it
                         self.compute_dtype) + self.role[roles[a] % MAX_ROLES]
            acc = ff if acc is None else acc + ff
        return self.proj(acc)


class PointEncoder(tf.keras.layers.Layer):
    """Points (coords (B,N,K) in [0,1]^K, feats (B,N,F)) -> latent box (B, *dims, d)."""

    def __init__(self, d, heads=4, sigma=0.05, name="penc", **kw):
        super().__init__(name=name, **kw)
        self.d = d
        self.sigma = sigma
        self.coord = CoordEncoder(d, name=f"{name}_coord")
        self.fproj = tf.keras.layers.Dense(d, name=f"{name}_f")
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=heads,
                                                      key_dim=max(d // heads, 8),
                                                      output_shape=d, name=f"{name}_mha")
        self.out = tf.keras.layers.Dense(d, name=f"{name}_o")

    def call(self, coords, feats, dims, roles=None):
        B = tf.shape(coords)[0]
        nodes = _box_nodes(dims)                                   # (P, K) in [0,1]^K
        nodes_b = tf.tile(nodes[None], [B, 1, 1])
        q = self.coord(nodes_b, roles=roles)                             # (B, P, d)
        kv = self.coord(coords, roles=roles) + self.fproj(feats)         # (B, N, d)
        h = self.mha(q, kv)                                        # (B, P, d)
        # coverage: soft point density at each node (information-existence map)
        d2 = tf.reduce_sum(tf.square(nodes_b[:, :, None] - coords[:, None]), -1)  # (B,P,N)
        cov = tf.reduce_sum(tf.exp(-d2 / (2 * self.sigma ** 2)), -1, keepdims=True)
        cov = tf.math.log1p(cov)
        h = self.out(tf.concat([h, q, cov], -1))
        return tf.reshape(h, tf.concat([[B], dims, [self.d]], 0)), None   # (no geom box for attn encoder)


class ContinuousDecoder(tf.keras.layers.Layer):
    """Bias-free pointwise decode of the latent trajectory: MLP(0) = 0 => hard IC.

    Optional Fourier-feature branch on the QUERY COORDINATES (MARIO lever) gives the decoder a
    high-frequency spatial basis the band-limited latent box cannot resolve — sharpening near-wall
    (mesh), small-scale (turbulence) and discontinuous (shock) fields. It enters as a MULTIPLICATIVE
    FiLM scale on the hidden features: hid * (1 + gate(fourier(coords))). This keeps BOTH invariants:
    (a) hard IC — at z=0, h(z)=0 so the scaled hidden is 0 regardless of coords; (b) clean warm-start
    — gate is zero-init so the factor is exactly 1 at load (output unchanged)."""

    def __init__(self, d_out, hidden=64, n_freq=16, n_fams=0, n_dict=0, n_dict_out=0,
                 fam_only=False, name="pdec", **kw):
        super().__init__(name=name, **kw)
        # ITEM G (2026-08-05): the box carries COEFFICIENTS, not field content. On top of the shared
        # delta head the decoder emits d_out*n_dict coefficients that multiply the NATIVE anchor
        # dictionary of core/dictdec.py, so the increment's fine structure never has to pass through
        # the band-limited box. The constant slot of the dictionary is deliberately absent: the
        # existing head `o` already plays that role. Zero-init => identical output at load, so every
        # existing checkpoint warm-starts unchanged; the input `hid` is not itself zero-gated, so
        # this is not a dead saddle (cf. the note on `o` below).
        # The dictionary covers the SEMANTIC slots only. The memory slots are recurrent scratch,
        # not a physical field, so they have no anchor whose derivatives could mean anything; they
        # keep the shared delta head alone. This also cuts the dfeat tensor by a quarter.
        self.n_dict = int(n_dict)
        self.n_dict_out = int(n_dict_out) or d_out
        if self.n_dict:
            self.dict_c = tf.keras.layers.Dense(self.n_dict_out * self.n_dict, use_bias=False,
                                                kernel_initializer="zeros", name=f"{name}_dc")
        # bias-free => MLP(0)=0 (hard IC). NOT zero-init: the tendency head is the single
        # zero-init gate of the path — two zero-inits in series is a gradient-dead saddle.
        self.h = tf.keras.layers.Dense(hidden, activation="gelu", use_bias=False, name=f"{name}_h")
        self.o = tf.keras.layers.Dense(d_out, use_bias=False, name=f"{name}_o")
        # SHARED-SLOT INTERFERENCE FIX (2026-07-31): the final latent->slot map used to be ONE
        # unconditioned Dense for every family, so families sharing a slot with DIFFERENT physics
        # (slot4: compressible p / Poisson solution / steady surface p / elastic stress; slot6:
        # wave field / reaction scalar / advected smoke) had to share output directions.
        # (a) cond-FiLM: equation conditioning modulates the decode hidden (zero-init -> inert);
        # (b) per-family output delta: out += hid @ W_fam (zero-init table) — independent output
        # channels per problem on top of the shared head. Both warm-safe. n_fams=0 disables (v2).
        self.cond_gate = tf.keras.layers.Dense(hidden, use_bias=False, kernel_initializer="zeros",
                                               name=f"{name}_cgate")
        self.n_fams, self.hidden, self.d_out = int(n_fams), hidden, d_out
        # UNSHARED OUTPUT CHANNELS (MOTION-NCS, 2026-08-06). fam_only=True removes the shared
        # latent->slot map entirely: the ONLY route from the hidden features to output channels
        # is this family's own weight block, so no two families share any output direction and
        # the slot layout degrades to a container. This is what makes the mechanism-knockout
        # table clean — any cross-family effect must flow through the shared trunk/banks, never
        # through a shared output head. Init is 1/sqrt(hidden) normal (it IS the output map now,
        # not a warm-safe delta); still bias-free, so MLP(0)=0 and the hard IC survives.
        # fam_only=False keeps the v2/v3 behavior: shared head + zero-init per-family delta.
        self.fam_only = bool(fam_only) and self.n_fams > 0
        if self.n_fams:
            self.fam_table = self.add_weight(
                name=f"{name}_famW", shape=(self.n_fams, hidden * d_out),
                initializer=(tf.keras.initializers.RandomNormal(stddev=hidden ** -0.5)
                             if self.fam_only else "zeros"),
                trainable=True)
        self.ff_freq = tf.constant(2.0 ** np.arange(n_freq), tf.float32) * np.pi
        self.ff_role = self.add_weight(name=f"{name}_ffrole", shape=(MAX_ROLES, 2 * n_freq),
                                       initializer="zeros", trainable=True)
        self.ff_gate = tf.keras.layers.Dense(hidden, use_bias=False, kernel_initializer="zeros",
                                             name=f"{name}_ffgate")            # zero-init -> factor 1 at load
        # FULL-RESOLUTION per-query GEOMETRY FiLM (mesh lever): the coarse latent box scatter-means
        # the surface geometry into cells, blurring the normal/curvature that determine surface
        # pressure (ablation: box-path geometry ~0 effect). Read the geometry AT THE QUERY POINT
        # (native resolution, no box averaging) and modulate the decoder hidden — same multiplicative
        # FiLM form as the coord branch, so it preserves the hard IC (h(0)=0) and is warm-safe
        # (geom_gate zero-init -> factor 1 at load). Off for families with no geometry (geom_q=0).
        self.geom_feat = tf.keras.layers.Dense(hidden, activation="gelu", name=f"{name}_gf")
        self.geom_gate = tf.keras.layers.Dense(hidden, use_bias=False, kernel_initializer="zeros",
                                               name=f"{name}_gg")

    def _fourier(self, coords, roles=None):
        K = int(coords.shape[-1]); roles = roles or list(range(K))
        acc = None
        for a in range(K):
            ph = tf.cast(coords[..., a:a + 1], tf.float32) * self.ff_freq     # f32 phase: 2^15*pi*x
            ff = tf.cast(tf.concat([tf.sin(ph), tf.cos(ph)], -1),              # loses the high
                         self.compute_dtype) + self.ff_role[roles[a] % MAX_ROLES]   # bands in bf16
            acc = ff if acc is None else acc + ff
        return acc                                                            # (B,Q,2*n_freq)

    def call(self, z, coords=None, roles=None, geom=None, cond=None, fam_id=None, dfeat=None):
        hid = self.h(z)
        if coords is not None:                                                # FiLM scale (1 at init, 0 at z=0)
            hid = hid * (1.0 + self.ff_gate(self._fourier(coords, roles)))
        if geom is not None:                                                  # full-res per-query geometry FiLM
            hid = hid * (1.0 + self.geom_gate(self.geom_feat(tf.cast(geom, self.compute_dtype))))
        if cond is not None:                                                  # equation-conditioned decode
            hid = hid * (1.0 + self.cond_gate(cond)[:, None, :])
        if self.fam_only and fam_id is not None:                              # unshared output head
            Wf = tf.reshape(tf.gather(self.fam_table, fam_id),
                            [-1, self.hidden, self.d_out])                    # (B,hidden,d_out)
            out = tf.einsum("bqh,bho->bqo", hid, Wf)
        else:
            out = self.o(hid)
            if self.n_fams and fam_id is not None:                            # per-family output delta
                Wf = tf.reshape(tf.gather(self.fam_table, fam_id),
                                [-1, self.hidden, self.d_out])                # (B,hidden,d_out)
                out = out + tf.einsum("bqh,bho->bqo", hid, Wf)
        if self.n_dict and dfeat is not None:                                 # item G: coefficients
            c = tf.reshape(self.dict_c(hid),
                           [tf.shape(hid)[0], -1, self.n_dict_out, self.n_dict])
            dd = tf.reduce_sum(c * tf.cast(dfeat, c.dtype), -1)               # (B,Q,n_dict_out)
            if self.n_dict_out < self.d_out:                                  # memory slots: no dict
                dd = tf.pad(dd, [[0, 0], [0, 0], [0, self.d_out - self.n_dict_out]])
            out = out + dd
        return out


def _box_nodes(dims):
    """Cell-center lattice coordinates in [0,1]^K -> (prod(dims), K)."""
    axes = [(np.arange(n) + 0.5) / n for n in dims]
    mesh = np.meshgrid(*axes, indexing="ij")
    return tf.constant(np.stack([m.ravel() for m in mesh], -1), tf.float32)
