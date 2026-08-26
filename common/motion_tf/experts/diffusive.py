"""Diffusive expert (DESIGN.md §4.1).

Parabolic dissipation. Structural bias: the **only** spatial-coupling pathway is a fixed
symmetric periodic Laplacian (zero-sum → pure dissipation); a pointwise (RF-0) MLP shapes
the response but cannot introduce transport. So this expert can diffuse but not advect.
Operator magnitude is carried by the family gate (DESIGN.md §6.4).
"""
from __future__ import annotations
import tensorflow as tf
from .base import Branch, laplacian


class DiffusiveExpert(Branch):
    name_tag = "diffusive"

    def __init__(self, d_geom, hidden=24, out_ch=1, name="diffusive"):
        super().__init__(name=name)
        self.l1 = tf.keras.layers.Conv2D(hidden, 1, activation="gelu", name="d_l1")
        self.out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="d_out")

    def contribution(self, z, ctx):
        dx = ctx["metric"][..., 0:1]
        dy = ctx["metric"][..., 1:2]
        lap = laplacian(z, dx, dy)                       # symmetric → dissipative only
        x = tf.concat([lap, z, ctx["h_geom"]], axis=-1)  # MLP is pointwise (RF 0)
        return self.out(self.l1(x))
