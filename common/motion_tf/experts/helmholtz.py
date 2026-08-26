"""Helmholtz / vector-decomposition expert.

The operator-type experts (convective/diffusive/…) process each channel independently — they never
form the CROSS-CHANNEL quantities that define vector-field (Navier–Stokes) dynamics. 2-D NS lives in
its Helmholtz decomposition: the SOLENOIDAL part = vorticity ω = ∂_y vx − ∂_x vy (rotation) and the
DILATATIONAL part = divergence δ = ∂_x vx + ∂_y vy (compressibility). NTO-ADA solved NS in vorticity
form (ω→ψ=Δ⁻¹(−ω)→u) — this expert injects that bias: compute (ω, δ) from the velocity slots, mix
them with a dilated-conv + global-mean stack (the Δ⁻¹/elliptic global coupling, non-spectral per
feedback_no_spectral_encoder), and emit a contribution to every channel.

Convention: velocity = slots 0 (vx), 1 (vy) (data.prose SPEC). Non-velocity families (SWE height,
diff-react concentrations) have zeros there → ω=δ=0 → this expert contributes ~0 (zero-init out)."""
from __future__ import annotations
import tensorflow as tf
from .base import Branch, periodic_pad_2d, d_dx, d_dy


class HelmholtzExpert(Branch):
    name_tag = "helmholtz"

    def __init__(self, d_geom, hidden=24, dilations=(1, 2, 4, 8), out_ch=1, name="helmholtz"):
        super().__init__(name=name)
        self.dilations = tuple(dilations)
        self.convs = [tf.keras.layers.Conv2D(hidden, 3, padding="valid", dilation_rate=d,
                                             activation="gelu", name=f"hz_d{d}")
                      for d in self.dilations]
        self.global_dense = tf.keras.layers.Dense(hidden, name="hz_glob")
        self.out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="hz_out")

    def contribution(self, z, ctx):
        dx = ctx["metric"][..., 0:1]
        dy = ctx["metric"][..., 1:2]
        vx = z[..., 0:1]
        vy = z[..., 1:2]
        omega = d_dy(vx, dy) - d_dx(vy, dx)          # vorticity (solenoidal feature)
        div = d_dx(vx, dx) + d_dy(vy, dy)            # divergence (dilatational feature)
        h = tf.concat([omega, div, z, ctx["h_geom"]], axis=-1)
        for conv, d in zip(self.convs, self.dilations):
            h = conv(periodic_pad_2d(h, d))          # dilated → global RF for the Δ⁻¹ coupling
        gmean = tf.reduce_mean(h, axis=[1, 2], keepdims=True)
        h = h + self.global_dense(gmean)             # global-mean broadcast (non-spectral elliptic)
        return self.out(h)
