"""K-agnostic latent-box core: a stack of axis-factorized blocks.

The SAME instance (same variables) processes (B, n1, n2, D) and (B, n1, n2, n3, D).
This is the parameter-sharing backbone for the universal 2D/3D foundation model;
tendency heads / TP-ADA synthesis / decoders attach around it.
"""
from __future__ import annotations

import tensorflow as tf

from .axops import AxialAttention, AxialConv, MoEPointwiseMLP, PointwiseMLP


class LatentBoxCore(tf.keras.layers.Layer):
    def __init__(self, d, depth=4, heads=4, kernel=5, moe_experts=0, moe_topk=2, name="lbcore", **kw):
        super().__init__(name=name, **kw)
        self.inp = tf.keras.layers.Dense(d, name=f"{name}_in")
        self.blocks = []
        self.moe = []                                  # MoE MLP layers (for load-balance aux)
        for i in range(depth):
            if moe_experts > 0:
                mlp = MoEPointwiseMLP(d, n_experts=moe_experts, top_k=moe_topk, name=f"{name}_ml{i}")
                self.moe.append(mlp)
            else:
                mlp = PointwiseMLP(d, name=f"{name}_ml{i}")
            self.blocks.append((AxialConv(d, kernel, name=f"{name}_cv{i}"),
                                AxialAttention(d, heads, name=f"{name}_at{i}"), mlp))

    def aux(self):
        """Summed MoE load-balance loss across blocks (0 if not MoE)."""
        import tensorflow as _tf
        return _tf.add_n([m.aux for m in self.moe]) if self.moe else _tf.constant(0.0)

    def call(self, x, roles=None):
        """x: (B, n_1..n_K, C_in) -> (B, n_1..n_K, D), any K >= 1."""
        h = self.inp(x)
        for cv, at, ml in self.blocks:
            h = cv(h, roles=roles)
            h = at(h, roles=roles)
            h = ml(h)
        return h
