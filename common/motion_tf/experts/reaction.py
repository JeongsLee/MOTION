"""Reaction-Source expert (DESIGN.md §4.1).

Pointwise nonlinearity R(u) + source f. Structural bias: **zero receptive field** —
1×1 convs only, so it CANNOT represent any spatial coupling. This is the constraint that
prevents it from absorbing advective/diffusive physics (§6.4). Operator magnitude is
carried by the family gate, not by the expert.
"""
from __future__ import annotations
import tensorflow as tf
from .base import Branch


class ReactionExpert(Branch):
    name_tag = "reaction"

    def __init__(self, d_geom, hidden=32, out_ch=1, name="reaction"):
        super().__init__(name=name)
        self.l1 = tf.keras.layers.Conv2D(hidden, 1, activation="gelu", name="r_l1")
        self.l2 = tf.keras.layers.Conv2D(hidden, 1, activation="gelu", name="r_l2")
        self.out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="r_out")

    def contribution(self, z, ctx):
        # pure pointwise: state + shared geom feature (for spatial source f(x)).
        x = tf.concat([z, ctx["h_geom"]], axis=-1)
        return self.out(self.l2(self.l1(x)))
