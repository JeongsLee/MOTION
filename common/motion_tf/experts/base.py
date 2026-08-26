"""Common interface for physics-operator experts (DESIGN.md §4, §5).

Every expert is a relational spatial operator module (conv / graph stencil — NOT a
pointwise coord→W INR; see DESIGN.md §4.5) that, given the current latent state z_i and
the problem instance, emits its own panel-velocity contribution ΔW_i:

    ΔW_i = B_k(z_i, ctx)        # same field shape as z_i (B, H, W, 1)

The model (model.py) sums these by a family-conditioned gate (operator splitting):

    W_i = Π_{BC,Ω}[ Σ_k α_k(family) · B_k(z_i, ctx) ]

`ctx` is a dict carrying everything an expert may need:
    h_geom : (B, H, W, d_geom)  shared geometry/coord feature (computed once)
    coeffs : (B, n_coeff)       per-sample PDE coefficients (advection vel, ν, rate, ...)
    metric : (B, H, W, 2)       (Δx, Δy) cell sizes (computational-coordinate metric)

Structural specialization (DESIGN.md §6.4) is enforced by each expert's receptive field:
Reaction = 1×1 (RF 0), Diffusive = local 3×3, Convective = directional 3×3, Elliptic =
global. The structural limit is what keeps a flexible expert from absorbing all physics.
"""
from __future__ import annotations
import tensorflow as tf


def periodic_pad_2d(x, pad=1):
    """Wrap-pad along H, W (axes 1, 2) of (B, H, W, C). Matches NS ADO convention."""
    if pad == 0:
        return x
    x = tf.concat([x[:, :, -pad:, :], x, x[:, :, :pad, :]], axis=2)
    x = tf.concat([x[:, -pad:, :, :], x, x[:, :pad, :, :]], axis=1)
    return x


# --------------------------------------------------------------------------- #
# Fixed periodic finite-difference operators.
#
# These hard-code the *spatial differential character* of each expert so that
# an expert's only spatial-coupling pathway is its characteristic operator
# (advection = antisymmetric 1st derivative, diffusion = symmetric Laplacian).
# A pointwise (RF-0) MLP on top cannot re-introduce other couplings, so the
# experts are structurally specialized and the family gate must track the
# active operators (DESIGN.md §6.4).
# --------------------------------------------------------------------------- #
def d_dx(z, dx):
    """Central ∂/∂x on (B,H,W,1), periodic. dx broadcastable to (B,1,1,1)."""
    dx = tf.cast(dx, z.dtype)                                         # match z (bf16 under AMP)
    return (tf.roll(z, -1, axis=1) - tf.roll(z, 1, axis=1)) / (2.0 * dx)


def d_dy(z, dy):
    dy = tf.cast(dy, z.dtype)
    return (tf.roll(z, -1, axis=2) - tf.roll(z, 1, axis=2)) / (2.0 * dy)


def upwind_dx(z, v, dx):
    """First-order UPWIND ∂z/∂x for advection by velocity v, periodic. Backward difference where v>0,
    forward where v<0 → the stable/monotone scheme for transport (central-diff self-advection blew up:
    incom 311%). z:(B,H,W,C), v:(B,H,W,1) broadcasts over channels; dx:(B,1,1,1)."""
    dx = tf.cast(dx, z.dtype); v = tf.cast(v, z.dtype)
    back = (z - tf.roll(z, 1, axis=1)) / dx                  # (z_i - z_{i-1})/dx   (v>0: info from upstream −x)
    fwd = (tf.roll(z, -1, axis=1) - z) / dx                  # (z_{i+1} - z_i)/dx   (v<0)
    return tf.where(v > 0, back, fwd)


def upwind_dy(z, v, dy):
    """First-order UPWIND ∂z/∂y for advection by velocity v (see upwind_dx)."""
    dy = tf.cast(dy, z.dtype); v = tf.cast(v, z.dtype)
    back = (z - tf.roll(z, 1, axis=2)) / dy
    fwd = (tf.roll(z, -1, axis=2) - z) / dy
    return tf.where(v > 0, back, fwd)


def _minmod(a, b):
    """minmod limiter: 0 if a,b opposite-signed, else the smaller-magnitude (TVD slope limiter)."""
    return 0.5 * (tf.sign(a) + tf.sign(b)) * tf.minimum(tf.abs(a), tf.abs(b))


def minmod_dx(z, dx):
    """minmod-limited (2nd-order TVD) ∂z/∂x, periodic — 2nd-order accurate where smooth, falls to 0 at
    extrema (no spurious overshoot). MUCH less numerical diffusion than 1st-order upwind, still bounded/
    stable. Use for self-advection to avoid the upwind over-smoothing. z:(B,H,W,C), dx:(B,1,1,1)."""
    dx = tf.cast(dx, z.dtype)
    dm = (z - tf.roll(z, 1, axis=1)) / dx                    # backward (left) slope
    dp = (tf.roll(z, -1, axis=1) - z) / dx                   # forward (right) slope
    return _minmod(dm, dp)


def minmod_dy(z, dy):
    """minmod-limited (2nd-order TVD) ∂z/∂y, periodic (see minmod_dx)."""
    dy = tf.cast(dy, z.dtype)
    dm = (z - tf.roll(z, 1, axis=2)) / dy
    dp = (tf.roll(z, -1, axis=2) - z) / dy
    return _minmod(dm, dp)


def laplacian(z, dx, dy):
    """5-point periodic Laplacian on (B,H,W,1) — symmetric, zero-sum (pure dissipation)."""
    dx = tf.cast(dx, z.dtype); dy = tf.cast(dy, z.dtype)
    lap_x = (tf.roll(z, -1, axis=1) - 2.0 * z + tf.roll(z, 1, axis=1)) / (dx ** 2)
    lap_y = (tf.roll(z, -1, axis=2) - 2.0 * z + tf.roll(z, 1, axis=2)) / (dy ** 2)
    return lap_x + lap_y


class Branch(tf.Module):
    """Abstract physics-operator expert. Subclasses implement ``contribution``."""

    #: short physical name, used for gate indexing / interpretability read-out
    name_tag: str = "branch"

    def contribution(self, z, ctx):
        """z: (B, H, W, 1) current state. Returns ΔW: (B, H, W, 1)."""
        raise NotImplementedError

    def __call__(self, z, ctx):
        return self.contribution(z, ctx)

    @property
    def trainable_variables(self):
        # default: gather from any tf.keras layers held as attributes
        out = []
        for v in vars(self).values():
            if isinstance(v, tf.Variable):                 # bare learnable scalars/vectors (e.g. learnable-FD stencils)
                if getattr(v, "trainable", True): out.append(v)
                continue
            tv = getattr(v, "trainable_variables", None)
            if tv:
                out.extend(list(tv))
        # dedup
        seen, uniq = set(), []
        for t in out:
            if id(t) not in seen:
                seen.add(id(t)); uniq.append(t)
        return uniq
