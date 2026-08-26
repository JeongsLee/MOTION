"""Wave / restoring expert (DESIGN.md §4.1 extension).

The operator experts so far are either single-channel (convective/diffusive/reaction process each
channel independently) or NS-vorticity-specific (helmholtz). NONE captures the CROSS-CHANNEL
GRADIENT coupling that defines WAVE dynamics — where the gradient of one field is the restoring
force on another:
    • shallow water / gravity waves:   ∂_t(h u) = … − g·h·∇h        (height gradient → momentum)
    • compressible / acoustic waves:   ∂_t(ρu) = … − ∇p ,  ∂_t ρ = −∇·(ρu)  (pressure↔velocity↔density)
These are first-order-in-time cross-channel ∇ couplings — exactly the structure the panel framework
(dz/dt = W) needs, but which no single-channel operator can form.

Structural bias: the spatial-coupling pathway is the per-channel 1st derivatives (∂_x, ∂_y of every
channel); a 1×1 mix then forms ANY linear cross-channel gradient coupling (the wave operator), and an
MLP shapes the nonlinear (e.g. h·∇h) response. zero-init out → 0 contribution at start. Distinct from
convective (self-advection u·∇u, same-channel transport) and helmholtz (vorticity/divergence, NS-form).
"""
from __future__ import annotations
import tensorflow as tf
from .base import Branch, d_dx, d_dy


class WaveExpert(Branch):
    name_tag = "wave"

    def __init__(self, d_geom, hidden=32, out_ch=1, name="wave"):
        super().__init__(name=name)
        self.l1 = tf.keras.layers.Conv2D(hidden, 1, activation="gelu", name="w_l1")
        self.out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="w_out")

    def contribution(self, z, ctx):
        dx = ctx["metric"][..., 0:1]
        dy = ctx["metric"][..., 1:2]
        gx = d_dx(z, dx)                                       # (B,H,W,C) ∂_x of every channel
        gy = d_dy(z, dy)                                       # (B,H,W,C) ∂_y of every channel
        # gradients of ALL channels + state → 1×1 MLP forms cross-channel ∇ coupling (wave/restoring).
        x = tf.concat([gx, gy, z, ctx["h_geom"]], axis=-1)
        return self.out(self.l1(x))
