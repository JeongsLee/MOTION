"""Convective expert (DESIGN.md §4.1).

Hyperbolic transport −a·∇u. Structural bias: the **only** spatial-coupling pathway is a
fixed antisymmetric 1st-derivative (directional derivative along the advection direction);
a pointwise (RF-0) MLP shapes the response but cannot introduce dissipation. So this expert
can advect but not diffuse.

Two advection modes:
  • coeffs (default): direction read from coeffs[:, :2] (prescribed-advection toy PDEs). NOTE: in the
    multi-family pipeline coeffs ≡ 0 → this term VANISHES (the expert degenerates to a pointwise MLP).
  • self_advect=True: NONLINEAR SELF-ADVECTION u·∇z with u = the STATE velocity (channels 0,1) — the
    transport that actually drives SWE/Euler/NS. Velocity-less families (SWE height-only, diff-react)
    have v=0 → transport=0 (correctly inactive). This is the physically-grounded fluid transport that
    the coeffs path could never represent (it advected along a fixed EXTERNAL direction, not the state).
The operator *magnitude* is carried by the family gate (DESIGN.md §6.4).
"""
from __future__ import annotations
import tensorflow as tf
from .base import Branch, d_dx, d_dy, upwind_dx, upwind_dy, minmod_dx, minmod_dy


class ConvectiveExpert(Branch):
    name_tag = "convective"

    def __init__(self, d_geom, hidden=24, out_ch=1, self_advect=False, upwind=False, muscl=False,
                 vel_slots=(0, 1), name="convective"):
        super().__init__(name=name)
        self.self_advect = bool(self_advect)
        # advection scheme for self_advect (precedence muscl > upwind > central):
        #  muscl=True  → minmod-limited 2nd-order TVD (low numerical diffusion, stable) — the accurate choice
        #  upwind=True → 1st-order upwind (stable but over-diffusive, esp at the coarse latent grid)
        #  else        → central (unstable for advection; diverged incom 311% 2026-06-11)
        self.upwind = bool(upwind)
        self.muscl = bool(muscl)
        self.vel_slots = tuple(vel_slots)
        self.l1 = tf.keras.layers.Conv2D(hidden, 1, activation="gelu", name="c_l1")
        self.out = tf.keras.layers.Conv2D(out_ch, 1, kernel_initializer="zeros", name="c_out")

    def contribution(self, z, ctx):
        dx = ctx["metric"][..., 0:1]
        dy = ctx["metric"][..., 1:2]
        if self.self_advect:
            # nonlinear self-advection u·∇z, u = state velocity (channels vel_slots). Per channel:
            # −(v_x ∂_x z_c + v_y ∂_y z_c). v=0 for velocity-less families → transport=0.
            sx, sy = self.vel_slots
            ux = z[..., sx:sx + 1]
            uy = z[..., sy:sy + 1]
            if self.muscl:                                       # 2nd-order TVD (minmod): low diffusion + stable
                transport = -(ux * minmod_dx(z, dx) + uy * minmod_dy(z, dy))
            elif self.upwind:                                    # STABLE 1st-order upwind −u·∇z (sign(u)-biased)
                transport = -(ux * upwind_dx(z, ux, dx) + uy * upwind_dy(z, uy, dy))
            else:                                                # central (legacy; unstable for advection)
                transport = -(ux * d_dx(z, dx) + uy * d_dy(z, dy))   # broadcast over channels → −u·∇z
        else:
            # unit advection direction from coeffs[:, :2] = (a_x, a_y); magnitude → gate.
            a = ctx["coeffs"][:, :2]
            adir = a / (tf.norm(a, axis=-1, keepdims=True) + 1e-6)
            ux = adir[:, 0][:, None, None, None]
            uy = adir[:, 1][:, None, None, None]
            transport = -(ux * d_dx(z, dx) + uy * d_dy(z, dy))   # −a·∇z structure
        x = tf.concat([transport, z, ctx["h_geom"]], axis=-1)
        return self.out(self.l1(x))
