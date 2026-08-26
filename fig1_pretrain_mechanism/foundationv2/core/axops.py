"""Axis-factorized primitives: the SAME weights process any number of spatial axes.

State layout: (B, n_1, ..., n_K, D) with K in {2, 3} (works for any K >= 1).
Each op loops over the spatial axes, moves axis a to the sequence position,
folds all other dims into batch, applies ONE shared 1D layer, and unfolds.
An axis-role embedding (learned, tiny) is added before each application so
anisotropic physics (gravity axis vs horizontal) stays expressible; roles are
dataset metadata mapping axes -> role ids (default: axis index, capped).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

MAX_ROLES = 4  # x, y, z, spare


class RankFreeLN(tf.keras.layers.Layer):
    """LayerNorm over the LAST axis, rank-agnostic. tf.keras.LayerNormalization caches a
    build-time-rank reshape (Keras 2), so sharing one instance across K=2 (rank 4) and K=3
    (rank 5) inputs crashes. This normalizes over axis=-1 with broadcast gamma/beta (shape
    (d,)) that work at ANY rank — the whole point of the dimension-agnostic core."""

    def __init__(self, d, eps=1e-5, name="ln", **kw):
        super().__init__(name=name, **kw)
        self.g = self.add_weight(name=f"{name}_g", shape=(d,), initializer="ones", trainable=True)
        self.b = self.add_weight(name=f"{name}_b", shape=(d,), initializer="zeros", trainable=True)
        self.eps = eps

    def call(self, x):
        m = tf.reduce_mean(x, axis=-1, keepdims=True)
        v = tf.reduce_mean(tf.square(x - m), axis=-1, keepdims=True)
        return self.g * (x - m) * tf.math.rsqrt(v + self.eps) + self.b


def _to_axis_major(x, axis):
    """(B, n1..nK, D), spatial axis index a (0-based) -> (B*, n_a, D) + restore fn."""
    K = x.shape.rank - 2
    perm = [0] + [1 + j for j in range(K) if j != axis] + [1 + axis, K + 1]
    xt = tf.transpose(x, perm)
    shp = tf.shape(xt)
    static = xt.shape
    n_a = static[-2] if static[-2] is not None else shp[-2]
    D = static[-1]
    flat = tf.reshape(xt, [-1, n_a, D])

    def restore(y):
        y = tf.reshape(y, shp)                    # ops here preserve length and channel count
        inv = [0] * (K + 2)
        for i, p in enumerate(perm):
            inv[p] = i
        return tf.transpose(y, inv)

    return flat, restore


class AxialConv(tf.keras.layers.Layer):
    """Shared periodic 1D conv applied along every spatial axis + pointwise mix.
    Parameters are independent of K. Residual, zero-init output (safe to stack)."""

    def __init__(self, d, kernel=5, hidden=None, name="axconv", **kw):
        super().__init__(name=name, **kw)
        self.d = d
        self.k = kernel
        self.conv = tf.keras.layers.Conv1D(d, kernel, padding="valid", name=f"{name}_c1d")
        self.role = self.add_weight(name=f"{name}_role", shape=(MAX_ROLES, d),
                                    initializer="zeros", trainable=True)
        self.mix = tf.keras.layers.Dense(d, name=f"{name}_mix",
                                         kernel_initializer="zeros")   # zero-init residual
        self.act = tf.keras.activations.gelu

    def call(self, x, roles=None):
        K = x.shape.rank - 2
        roles = roles or list(range(K))
        acc = []
        p = self.k // 2
        for a in range(K):
            xa = x + self.role[roles[a] % MAX_ROLES]        # (d,) broadcasts on last axis
            flat, restore = _to_axis_major(xa, a)
            flat = tf.concat([flat[:, -p:], flat, flat[:, :p]], axis=1)   # periodic wrap
            acc.append(restore(self.conv(flat)))
        h = self.act(tf.add_n(acc) / float(K))
        return x + self.mix(h)


class AxialAttention(tf.keras.layers.Layer):
    """Shared MHSA applied along every spatial axis (rows of the other axes = batch).
    QKV/MLP weights are token-set ops -> dimension-blind; sharing across axes makes
    the whole layer K-agnostic. Residual; output projection zero-init."""

    def __init__(self, d, heads=4, name="axattn", **kw):
        super().__init__(name=name, **kw)
        self.d = d
        self.ln = RankFreeLN(d, name=f"{name}_ln")
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=max(d // heads, 8), output_shape=d, name=f"{name}_mha")
        self.role = self.add_weight(name=f"{name}_role", shape=(MAX_ROLES, d),
                                    initializer="zeros", trainable=True)
        self.gate = self.add_weight(name=f"{name}_gate", shape=(), initializer="zeros",
                                    trainable=True)                     # zero-init residual gate

    def call(self, x, roles=None):
        K = x.shape.rank - 2
        roles = roles or list(range(K))
        acc = []
        for a in range(K):
            xa = self.ln(x) + self.role[roles[a] % MAX_ROLES]   # (d,) broadcasts on last axis
            flat, restore = _to_axis_major(xa, a)
            acc.append(restore(self.mha(flat, flat)))
        return x + self.gate * (tf.add_n(acc) / float(K))


class PointwiseMLP(tf.keras.layers.Layer):
    """Per-node MLP — inherently dimension-blind. Residual, zero-init out."""

    def __init__(self, d, ratio=2, name="pmlp", **kw):
        super().__init__(name=name, **kw)
        self.h = tf.keras.layers.Dense(d * ratio, activation="gelu", name=f"{name}_h")
        self.o = tf.keras.layers.Dense(d, kernel_initializer="zeros", name=f"{name}_o")
        self.ln = RankFreeLN(d, name=f"{name}_ln")

    def call(self, x):
        return x + self.o(self.h(self.ln(x)))


class MoEPointwiseMLP(tf.keras.layers.Layer):
    """Sparse top-k Mixture-of-Experts replacing PointwiseMLP (LLM sparse-upcycling style).
    E experts (each = an {h(gelu), o} MLP); a per-cell router picks top_k. Dense compute (all
    experts run), top-k softmax gating -> only k contribute per cell. Upcycle: copy a trained
    dense MLP's h/o into every expert (+ tiny noise) and start the router near-zero -> at load
    the k identical experts average back to the dense block (warm-safe). `self.aux` carries the
    Switch-Transformer load-balance loss (E * sum_e f_e * P_e) for the training objective."""

    def __init__(self, d, ratio=2, n_experts=8, top_k=2, capacity=0.0, name="moe", **kw):
        super().__init__(name=name, **kw)
        self.E, self.k = int(n_experts), int(top_k)
        self.cf = float(capacity)          # >0: sparse dispatch at this capacity factor
        self.ln = RankFreeLN(d, name=f"{name}_ln")
        self.router = tf.keras.layers.Dense(
            self.E, use_bias=False, name=f"{name}_router",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-2))
        self.eh = [tf.keras.layers.Dense(d * ratio, activation="gelu", name=f"{name}_e{e}_h")
                   for e in range(self.E)]
        self.eo = [tf.keras.layers.Dense(d, name=f"{name}_e{e}_o") for e in range(self.E)]
        self.aux = tf.constant(0.0)

    def call(self, x):
        xn = self.ln(x)
        logits = self.router(xn)                                  # (B,*dims,E)
        topv, topi = tf.math.top_k(logits, self.k)               # (B,*dims,k)
        w = tf.nn.softmax(topv, -1)                              # (B,*dims,k)
        gate = tf.reduce_sum(tf.one_hot(topi, self.E, dtype=w.dtype)          # (B,*dims,E)
                             * w[..., None], axis=-2)
        out = (self._sparse(xn, gate) if self._can_sparse(xn) else
               tf.reduce_sum(gate[..., None]
                             * tf.stack([self.eo[e](self.eh[e](xn)) for e in range(self.E)], -2),
                             axis=-2))                           # (B,*dims,d)
        ax = list(range(len(xn.shape) - 1))                      # all but channel
        probs = tf.nn.softmax(logits, -1)
        P = tf.cast(tf.reduce_mean(probs, ax), tf.float32)       # (E,) mean router prob; the aux
        f = tf.reduce_mean(tf.cast(gate > 0.0, tf.float32), ax)  # term stays f32 (it is a loss)
        self.aux = tf.cast(self.E, tf.float32) * tf.reduce_sum(f * P)
        # Router telemetry, for diag_moe.py: `load` is the fraction of cells that routed to each
        # expert (uniform = k/E; collapse = one entry near 1 and the rest near 0) and `pmean` is
        # the mean softmax mass. `load` also decides whether the sparse dispatch above is exact:
        # its capacity is cf*k/E of the cells, so max(load) must stay below cf*k/E.
        self.load, self.pmean = f, P
        self.gate_w = tf.cast(tf.reduce_mean(tf.reduce_max(w, -1)), tf.float32)  # .5=blend, 1=hard
        return x + out

    # ------------------------------------------------------------------ sparse dispatch
    # Dense compute runs all E experts on every cell and then multiplies E-2 of them by a zero
    # gate: with E=16, k=2 that is 8x the expert FLOPs actually used, and it scales linearly with
    # box cells -- exactly the term that a 64^2 -> 128^2 resolution bump multiplies by 4.
    # Here each expert instead runs on a FIXED capacity C of cells, selected by top_k over its own
    # gate column so every shape stays static and XLA can compile it (tf.where would not).
    # Exact, not approximate, in both directions: cells the router did not assign to e arrive with
    # gate 0 and contribute nothing, and C is sized so assignment never overflows in practice
    # (expected load is T*k/E; cf=2.0 leaves 2x slack). Overflow, if it ever happened, would drop
    # the LOWEST-gate cells for that expert only.
    def _can_sparse(self, xn):
        return self.cf > 0.0 and xn.shape.is_fully_defined()

    def _capacity(self, T):
        return int(min(T, max(1, int(round(self.cf * T * self.k / self.E)))))

    def _sparse(self, xn, gate):
        shp = tf.shape(xn)
        d = xn.shape[-1]
        T = int(np.prod(xn.shape[:-1]))
        C = self._capacity(T)
        x2 = tf.reshape(xn, [T, d])
        g2 = tf.reshape(gate, [T, self.E])
        acc = tf.zeros([T, d], xn.dtype)
        for e in range(self.E):
            v, idx = tf.math.top_k(g2[:, e], C)                   # (C,) weights and cell ids
            ye = self.eo[e](self.eh[e](tf.gather(x2, idx)))        # (C,d) -- only C cells, not T
            acc = tf.tensor_scatter_nd_add(acc, idx[:, None], ye * tf.cast(v, ye.dtype)[:, None])
        return tf.reshape(acc, shp)
