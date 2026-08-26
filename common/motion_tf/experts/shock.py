"""Shock expert (DESIGN.md §4.1, compressible-Euler gate) — hyperbolic conservation law with
shock capturing. Generalizes the proven NTO-ADA Burgers `UpwindConv1D` (learned state-gated
upwind) to 2-D systems, in CONSERVATION form.

    ∂_t u = −∇·F̂(u),   F̂ = state-gated upwind blend of left/right-biased flux kernels.

Key principle (from Burgers, proven on shocks): NO hardcoded Riemann solver — a sigmoid gate on
the local state + a shock indicator (|∇u|) selects the upwind direction, recovering symmetric
(central) flux where smooth and upwind where a shock forms. Conservation form (divergence of a
learned numerical flux) gives the Rankine–Hugoniot-consistent structure needed for strong Euler
shocks. Structural bias = directional + flux-divergence (distinct from smooth convective / global
elliptic / local diffusive).
"""
from __future__ import annotations
import tensorflow as tf
from .base import Branch, periodic_pad_2d, d_dx, d_dy


class ShockExpert(Branch):
    name_tag = "shock"

    def __init__(self, d_geom, hidden=32, out_ch=1, name="shock"):
        super().__init__(name=name)
        # left/right-biased flux kernels per axis (learned numerical flux)
        self.fxL = tf.keras.layers.Conv2D(hidden, 3, padding="valid", name="s_fxL")
        self.fxR = tf.keras.layers.Conv2D(hidden, 3, padding="valid", name="s_fxR")
        self.fyL = tf.keras.layers.Conv2D(hidden, 3, padding="valid", name="s_fyL")
        self.fyR = tf.keras.layers.Conv2D(hidden, 3, padding="valid", name="s_fyR")
        # state + shock-indicator gate → (g_x, g_y); zero-init → 0.5 (symmetric/central start)
        self.gate = tf.keras.layers.Conv2D(2, 1, kernel_initializer="zeros",
                                           bias_initializer="zeros", name="s_gate")
        self.out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="s_out")

    def contribution(self, z, ctx):
        dx = ctx["metric"][..., 0:1]
        dy = ctx["metric"][..., 1:2]
        grad = tf.abs(d_dx(z, dx)) + tf.abs(d_dy(z, dy))         # shock indicator (B,H,W,C)
        g = tf.sigmoid(self.gate(tf.concat([z, grad], axis=-1)))  # (B,H,W,2)
        gx, gy = g[..., 0:1], g[..., 1:2]

        xp = periodic_pad_2d(tf.concat([z, ctx["h_geom"]], axis=-1), 1)
        Fx = gx * self.fxR(xp) + (1.0 - gx) * self.fxL(xp)        # numerical x-flux (hidden)
        Fy = gy * self.fyR(xp) + (1.0 - gy) * self.fyL(xp)
        div = d_dx(Fx, dx) + d_dy(Fy, dy)                         # ∇·F̂ (conservation form)
        return self.out(-div)
