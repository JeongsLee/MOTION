"""Forcing expert — external (input-inferred) body-force source.

The reaction expert emits a POINTWISE source f (zero receptive field); convective/diffusive/helmholtz model
STATE-operators. NONE captures a spatially-structured EXTERNAL forcing f(x) that drives the flow independent
of the instantaneous state — e.g. PDEBench incom_ns's external force field, or PDEArena NS-cond's buoyancy
(force ∝ a per-trajectory VARYING coefficient × the advected scalar). Such forcing is quasi-static over a
trajectory and inferable from the INPUT WINDOW (encoded in h_geom). This expert injects that bias: a
dilated-conv force field from h_geom (spatial receptive field, unlike pointwise reaction), scaled by a global
coefficient read from the input statistics (captures the varying buoyancy/force magnitude). Zero-init out →
starts at 0 (stable); for incompressible families the emitted body force is made divergence-free downstream
by the elliptic/helmholtz projection (cross-expert W-feedback)."""
from __future__ import annotations
import tensorflow as tf
from .base import Branch, periodic_pad_2d


class ForcingExpert(Branch):
    name_tag = "forcing"

    def __init__(self, d_geom, hidden=24, dilations=(1, 2, 4, 8), out_ch=1, name="forcing"):
        super().__init__(name=name)
        self.dilations = tuple(dilations)
        self.convs = [tf.keras.layers.Conv2D(hidden, 3, padding="valid", dilation_rate=d,
                                             activation="gelu", name=f"f_d{d}")
                      for d in self.dilations]
        # per-trajectory force MAGNITUDE (e.g. NS-cond buoyancy coefficient) from global input statistics
        self.coef = tf.keras.layers.Dense(hidden, activation="sigmoid", name="f_coef")
        self.out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="f_out")

    def contribution(self, z, ctx):
        hg = ctx["h_geom"]                                   # input-window encoding carries the forcing signature
        h = tf.concat([z, hg], axis=-1)
        for conv, d in zip(self.convs, self.dilations):
            h = conv(periodic_pad_2d(h, d))                  # dilated → spatial RF (unlike pointwise reaction)
        gmean = tf.reduce_mean(hg, axis=[1, 2], keepdims=True)
        h = h * self.coef(gmean)                             # scale the force field by inferred global magnitude
        return self.out(h)
