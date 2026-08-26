"""Elliptic expert (DESIGN.md §4.1) — the global / non-local operator (pressure projection,
incompressibility, Δ⁻¹-like coupling, e.g. NS streamfunction ψ=Δ⁻¹(−ω)).

IMPORTANT (see memory feedback_no_spectral_encoder): NO spatial spectral/FFT transform here.
ADA already provides the spectral representation IN TIME; a spatial Fourier transform would be
a redundant "double sin" and worsens conditioning. NTO-ADA solved 2-D NS (incl. the elliptic
streamfunction coupling) to SOTA *without* a spectral encoder — using a multi-dilation
g-feedback conv stack. We follow that: large effective receptive field via dilated convs
(1,2,4,8) + a global-mean pool/broadcast for the truly-global component. Structural bias =
GLOBAL coupling, distinct from the local diffusive (Laplacian) / convective (gradient) experts.
"""
from __future__ import annotations
import tensorflow as tf
from .base import Branch, periodic_pad_2d


class EllipticExpert(Branch):
    name_tag = "elliptic"

    def __init__(self, d_geom, hidden=24, dilations=(1, 2, 4, 8), rounds=1, out_ch=1,
                 multigrid=False, mg_levels=3, mg_hidden=None, name="elliptic"):
        super().__init__(name=name)
        self.multigrid = bool(multigrid)
        if self.multigrid:
            # MULTIGRID V-CYCLE global coupling — the actual numerical method for the elliptic Poisson
            # (pressure) operator. Non-spectral (real-space hierarchy → orthogonal to ADA's TIME-Fourier,
            # no "double sin"), O(N), parameter-efficient. Restrict avg-pool L→L/2 ... →coarsest (near-
            # global RF), prolong bilinear + fine-level skip (residual correction), periodic-pad smoothers.
            # Gives STRUCTURED, spatially-varying global coupling vs the dilated+global-MEAN scalar proxy.
            self.mg_levels = int(mg_levels)
            # internal V-cycle width. Default LEAN (≤384) for a pure STRUCTURE test; set mg_hidden higher
            # (combo runs) to give the global-coupling operator real capacity on top of the structure.
            mgh = int(mg_hidden) if mg_hidden else min(int(hidden), 384)
            self.pre = tf.keras.layers.Conv2D(mgh, 3, padding="valid", activation="gelu", name="mg_pre")
            self.smooth_d = [tf.keras.layers.Conv2D(mgh, 3, padding="valid", activation="gelu", name=f"mg_sd{l}")
                             for l in range(self.mg_levels)]
            self.smooth_u = [tf.keras.layers.Conv2D(mgh, 3, padding="valid", activation="gelu", name=f"mg_su{l}")
                             for l in range(self.mg_levels)]
        else:
            self.dilations = tuple(dilations)
            self.rounds = int(rounds)                # >1 = ITERATE the dilated+global block (Poisson-
            #   iteration-like deeper global coupling for the incompressible-NS pressure/Δ⁻¹ term)
            self.blocks = []
            for r in range(self.rounds):
                convs = [tf.keras.layers.Conv2D(hidden, 3, padding="valid", dilation_rate=d,
                                                activation="gelu", name=f"e_r{r}_d{d}")
                         for d in self.dilations]
                gdense = tf.keras.layers.Dense(hidden, name=f"e_r{r}_glob")
                self.blocks.append((convs, gdense))
        self.out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="e_out")

    def contribution(self, z, ctx):
        h = tf.concat([z, ctx["h_geom"]], axis=-1)
        if self.multigrid:
            x = self.pre(periodic_pad_2d(h, 1))          # (B,L,L,hidden)
            skips = []
            for l in range(self.mg_levels):
                skips.append(x)
                x = tf.nn.avg_pool2d(x, 2, 2, "VALID")   # RESTRICT: L→L/2 (full-weighting ≈ averaging)
                x = self.smooth_d[l](periodic_pad_2d(x, 1))   # smooth at coarse level (near-global at coarsest)
            for l in reversed(range(self.mg_levels)):
                # PROLONG L/2→L. tf.image.resize upcasts to float32 → cast back to the skip's (bf16) dtype
                # so the residual add matches under mixed_bfloat16.
                x = tf.cast(tf.image.resize(x, tf.shape(skips[l])[1:3], method="bilinear"), skips[l].dtype)
                x = x + skips[l]                          # fine-level residual correction (V-cycle)
                x = self.smooth_u[l](periodic_pad_2d(x, 1))
            return self.out(x)
        feat = None
        for r, (convs, gdense) in enumerate(self.blocks):
            x = h if r == 0 else feat                # round 0: raw input; later rounds refine the field
            for conv, d in zip(convs, self.dilations):
                x = conv(periodic_pad_2d(x, d))      # dilation d, periodic → grows eff. RF
            gmean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)   # (B,1,1,hidden)
            x = x + gdense(gmean)                    # broadcast global context (non-spectral)
            feat = x if r == 0 else feat + x         # residual across iterated global-coupling rounds
        return self.out(feat)
