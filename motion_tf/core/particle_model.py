"""ParticleTPADA — the original vision: encoder emits N latent PARTICLES (each a position +
ADA time-panel W); a Transformer decoder projects the particles to physical space.

  input (points/frames) --[Perceiver cross-attn: N learned particle queries]--> N particle tokens
    --> per particle: position x_p in [0,1]^K (movable/Lagrangian) + W-panels W_p (ADA time)
    --> ADA time synthesis: z_p(t) = time-antiderivative(W_p)   (hard IC: z_p(0)=0)
    --> Transformer decoder: query x_q cross-attends to particles (key=pos, value=z_p(t_q)) --> u

Hard IC is structural: the time map G(0,:) = 0 -> z_p(0)=0; the decoder value/out projections are
bias-free so u(x,0)=u_0. Reuses CoordEncoder + KAxisTPADA's time map. Implements the subset of the
UniversalTPADA interface that core.pretrain drives (encode / from_points / point_traj_perquery[_g]
/ steady_field_at)."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .pointio import CoordEncoder
from .synthesis import KAxisTPADA


class ParticleEncoder(tf.keras.layers.Layer):
    """N learned particle-query tokens cross-attend to the input point cloud -> per particle:
    a token, a position in [0,1]^K (movable), and W-panels (d_w*n_p) for the ADA time trajectory."""

    def __init__(self, d, n_particles, d_w, n_p, heads=4, depth=6, name="penc_p", **kw):
        super().__init__(name=name, **kw)
        self.N, self.d, self.d_w, self.n_p = n_particles, d, d_w, n_p
        self.qemb = self.add_weight(name=f"{name}_qemb", shape=(n_particles, d),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.coord = CoordEncoder(d, name=f"{name}_coord")
        self.fproj = tf.keras.layers.Dense(d, name=f"{name}_f")
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=max(d // heads, 8),
                                                      output_shape=d, name=f"{name}_mha")
        # PROCESSOR: depth self-attention blocks over the N particles (the model's 'core').
        self.proc = []
        for i in range(depth):
            self.proc.append((
                tf.keras.layers.LayerNormalization(name=f"{name}_ln1_{i}"),
                tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=max(d // heads, 8),
                                                   output_shape=d, name=f"{name}_sa{i}"),
                tf.keras.layers.LayerNormalization(name=f"{name}_ln2_{i}"),
                tf.keras.layers.Dense(4 * d, activation="gelu", name=f"{name}_ff1_{i}"),
                tf.keras.layers.Dense(d, name=f"{name}_ff2_{i}")))
        self.pos = tf.keras.layers.Dense(1, name=f"{name}_pos")          # per-axis position logit (shared)
        self.wout = tf.keras.layers.Dense(d_w * n_p, name=f"{name}_w")    # ADA panels

    def call(self, coords, feats, K, roles=None):
        B = tf.shape(coords)[0]
        q = tf.tile(self.qemb[None], [B, 1, 1])                          # (B,N,d)
        kv = self.coord(coords, roles=roles) + self.fproj(feats)         # (B,Nin,d)
        h = self.mha(q, kv)                                              # (B,N,d)
        for ln1, sa, ln2, ff1, ff2 in self.proc:                        # transformer processor
            x = ln1(h); h = h + sa(x, x)
            h = h + ff2(ff1(ln2(h)))
        # K-agnostic position head: per-axis sigmoid from a shared 1-logit head over role-shifted h
        pos = tf.concat([tf.sigmoid(self.pos(h)) for _ in range(K)], -1)  # (B,N,K) in [0,1]
        W = self.wout(h)                                                 # (B,N,d_w*n_p)
        return h, pos, W


class TransformerDecoder(tf.keras.layers.Layer):
    """Query coords cross-attend to particle tokens; value = bias-free proj of the particle's
    latent trajectory at the query time z_p(t_q). Hard IC: z_p(0)=0 -> value 0 -> u(x,0)=u_0."""

    def __init__(self, d, d_out, d_w, heads=4, name="tdec", **kw):
        super().__init__(name=name, **kw)
        self.h_heads = heads
        self.coord = CoordEncoder(d, name=f"{name}_coord")               # query features
        self.kpos = CoordEncoder(d, name=f"{name}_kpos")                 # particle-position keys
        self.ktok = tf.keras.layers.Dense(d, name=f"{name}_ktok")        # particle-token -> key
        self.vproj = tf.keras.layers.Dense(d, use_bias=False, name=f"{name}_v")   # z -> value (bias-free)
        self.o = tf.keras.layers.Dense(d_out, use_bias=False, name=f"{name}_o")   # bias-free -> hard IC
        self.scale = 1.0 / np.sqrt(float(d))

    def call(self, q_coords, pos_p, ztok_p, z_pq, roles=None):
        """q_coords (B,Q,K); pos_p (B,N,K); ztok_p particle tokens (B,N,d); z_pq particle latent at
        query time (B,N,Q,d_w) -> u (B,Q,d_out)."""
        qf = self.coord(q_coords, roles=roles)                           # (B,Q,d)
        key = self.kpos(pos_p, roles=roles) + self.ktok(ztok_p)          # (B,N,d)
        logits = tf.einsum("bqd,bnd->bqn", qf, key) * self.scale         # (B,Q,N)
        a = tf.nn.softmax(logits, -1)                                    # attend query->particles
        val = self.vproj(z_pq)                                           # (B,N,Q,d) bias-free
        ctx = tf.einsum("bqn,bnqd->bqd", a, val)                         # (B,Q,d)
        return self.o(ctx)                                               # (B,Q,d_out)


class ParticleTPADA(tf.Module):
    """Particle-latent + Transformer-decoder operator. Implements the pretrain interface subset."""

    def __init__(self, box_cfgs, d=384, n_particles=256, d_w=16, n_p=48, t_final=1.0,
                 c_out=6, heads=4, depth=6, name="particle", **kw):
        super().__init__(name=name)
        self.d_w, self.n_p, self.N = d_w, n_p, n_particles
        with self.name_scope:
            self.penc = ParticleEncoder(d, n_particles, d_w, n_p, heads=heads, depth=depth)
            self.dec = TransformerDecoder(d, c_out, d_w, heads=heads)
        # reuse KAxisTPADA purely for the (spatial-agnostic) time map / query_time_map
        self.synth = {K: KAxisTPADA(dims, n_p, t_final=t_final, name=f"tp{K}d")
                      for K, dims in box_cfgs.items()}

    # -- interface -------------------------------------------------------------
    def encode(self, coords, feats, K, roles=None):
        h, pos, W = self.penc(coords, feats, K=K, roles=roles)
        return (h, pos, W)                                               # "latent" bundle

    def from_points(self, coords, feats, K, roles=None, op_multihot=None):
        h, pos, W = self.penc(coords, feats, K=K, roles=roles)
        return (h, pos, W), None, K

    def _z_pq(self, W, Gq):
        """W (B,N,d_w*n_p), Gq (B,Q,n_p) -> z_pq (B,N,Q,d_w): particle latent at each query time."""
        B = tf.shape(W)[0]
        Wp = tf.reshape(W, [B, self.N, self.d_w, self.n_p])
        return tf.einsum("bndp,bqp->bnqd", Wp, Gq)                       # (B,N,Q,d_w); z=0 at t=0 (Gq(0)=0)

    def point_traj_perquery_g(self, A, A_leg, K, pts, Gq, ic_at_pts, op_multihot=None, q_chunk=256):
        (h, pos, W) = A
        Q = int(pts.shape[1]); outs = []
        for s in range(0, Q, q_chunk):
            e = min(s + q_chunk, Q)
            z_pq = self._z_pq(W, Gq[:, s:e])                             # (B,N,q,d_w)
            u = self.dec(pts[:, s:e], pos, h, z_pq)                      # (B,q,c_out)
            outs.append(ic_at_pts[:, s:e] + u)
        return tf.concat(outs, axis=1)

    def point_traj_perquery(self, A, A_leg, K, pts, t_q, ic_at_pts, op_multihot=None):
        Gq = self.synth[K].query_time_map(t_q)
        return self.point_traj_perquery_g(A, A_leg, K, pts, Gq, ic_at_pts, op_multihot)

    def steady_field_at(self, latent, K, query, op_multihot=None):
        """Steady = ADA relaxation endpoint (t=1): evaluate the trajectory at t=1 with zero IC."""
        (h, pos, W) = latent
        B = tf.shape(query)[0]
        Gq = self.synth[K].query_time_map(np.ones((1, 1), np.float32))   # (1,1,n_p) at t=1
        Gq = tf.tile(Gq, [B, tf.shape(query)[1], 1])                     # (B,Q,n_p)
        z_pq = self._z_pq(W, Gq)
        return self.dec(query, pos, h, z_pq)

    def trainable_weights_flat(self):
        return (list(self.penc.trainable_weights) + list(self.dec.trainable_weights)
                + [g for s in self.synth.values() for g in (s.gains or [])]
                + [g for s in self.synth.values() for g in (s.gains_leg or [])])
