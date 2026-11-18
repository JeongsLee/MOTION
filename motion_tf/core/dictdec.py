"""Native anchor dictionary for the decode (design items G and H).

WHY (measured 2026-08-05, `paper2_universal/manuscript/supplement.tex`).
The band limit of the latent box is not this architecture's floor, and two independent measurements
say so: paper 1 runs its panel dynamics at an effective box of R = 32 and attains 7.73% on
pdearena_ns against a band-limited increment floor of 13.37% there, and diff_react attains 1.7%
against a band-limited field floor of 16.5% at R = 64. Both are possible because a native-resolution
path reconstructs content the box never carried. r19's decode does the opposite: it interpolates box
*content*, so the box has to carry the increment field itself, and the floor becomes real.

ITEM G. The box carries COEFFICIENTS; the structure they multiply is evaluated at native resolution
from the anchor,

    delta(x) = sum_j c_j(x) * B_j[u_anchor](x),          c_j read from the box, B_j native,

with B the local Taylor stencil of the anchor. That form is a spatially varying semi-Lagrangian
displacement plus a diffusion and a reaction term: the coarse field supplies the displacement, the
native anchor supplies the structure being displaced. Measured floor of exactly this decode at
R = 32, per-frame anchoring: ACE 0.05, shallow_water 0.09, com_ns 0.10, NS-Gauss 0.27,
Wave-Layer 0.47, CE-Gauss 0.67, CE-KH 0.84, incom_ns 0.86 (eight families under the 1% requirement),
against 1.80 and 0.71 for the band-limited decode on the first two of those. It is worth about one
octave of resolution, and it is worth most on transport-dominated families, which is what the form
predicts.

ITEM H. With one future frame per segment the anchor is re-imposed every frame, and a SINGLE frame
is not a state for a second-order-in-time system -- the failure paper 1 measured on the wave family.
Slot `dt` carries u(t-1) - u(t-2), so du/dt enters the dictionary and the re-anchored state is
complete. This is what makes L = 1 legitimate for wave rather than a trade-off.

SLOT LAYOUT is K-agnostic by design: the axis-specific slots exist for three axes and are exactly
zero in 2D, so one weight set serves K = 2 and K = 3 (the axops convention). Their coefficients then
multiply zero and receive no gradient, which is inert rather than harmful.

    0            value                     u
    1,2,3        first derivative          d/dx_a          (a = 2 zero in 2D)
    4,5,6        second derivative         d2/dx_a2        (a = 2 zero in 2D)
    7,8,9        mixed second derivative   d2/dx_a dx_b    for (0,1), (0,2), (1,2)
    10           time derivative (item H)  u - u_prev2     (zero when u_prev2 is absent)
    11..22       axis shifts, radius 1-2   u(x +- r e_a) - u(x)      (shift dictionary, optional)

The shift slots exist because the increment of a translating discontinuity is a two-sided jump of
FINITE WIDTH, which a derivative stencil represents only as a one-pixel spike; they are enabled for
the fine and native box buckets (the shock-carrying and turbulent families). The probe used a full
radius-1 ring; the axis-aligned form here keeps the layout K-agnostic at the same intent.

Stencils are periodic (tf.roll). Every family in the field corpus is periodic, and the walled
families were excluded from this study, so no boundary case arises here.
"""
from __future__ import annotations

import tensorflow as tf

J_TAYLOR = 11                  # slots 0..10
J_SHIFT = 12                   # slots 11..22 (3 axes x {+-1, +-2})
MIXED_PAIRS = ((0, 1), (0, 2), (1, 2))


def n_dict_slots(shift=False):
    return J_TAYLOR + (J_SHIFT if shift else 0)


def anchor_features(u, u_prev2=None, K=2, shift=False, normalize=True):
    """Native-resolution dictionary of the anchor.

    u        (B, *grid, S)  anchor frame on its native grid
    u_prev2  (B, *grid, S)  the frame before it, or None (item H off -> slot 10 is zero)
    returns  (B, *grid, S, J)

    Axis `a` of the grid is tensor axis `1 + a`; the channel axis is last. Slots that do not exist
    for this K are filled with zeros so that the coefficient head has one fixed layout.
    """
    zero = tf.zeros_like(u)
    ax = [1 + a for a in range(K)]

    def d1(a):
        return 0.5 * (tf.roll(u, -1, ax[a]) - tf.roll(u, 1, ax[a]))

    def d2(a):
        return tf.roll(u, -1, ax[a]) - 2.0 * u + tf.roll(u, 1, ax[a])

    first = [d1(a) if a < K else zero for a in range(3)]
    slots = [u]
    slots += first
    slots += [d2(a) if a < K else zero for a in range(3)]
    for (a, b) in MIXED_PAIRS:
        if a < K and b < K:
            slots.append(0.5 * (tf.roll(first[a], -1, ax[b]) - tf.roll(first[a], 1, ax[b])))
        else:
            slots.append(zero)
    slots.append(u - tf.cast(u_prev2, u.dtype) if u_prev2 is not None else zero)   # item H
    assert len(slots) == J_TAYLOR

    if shift:
        for a in range(3):
            for r in (1, 2):
                if a < K:
                    slots.append(tf.roll(u, -r, ax[a]) - u)
                    slots.append(tf.roll(u, r, ax[a]) - u)
                else:
                    slots += [zero, zero]
        assert len(slots) == J_TAYLOR + J_SHIFT

    F = tf.stack(slots, -1)                                        # (B, *grid, S, J)
    if normalize:
        # The slots differ by orders of magnitude (u ~ 1 after std-normalization, d2 u ~ 1e-2), and a
        # LEARNED linear head is not scale invariant the way the per-cell least squares of the probe
        # was. Dividing each slot by its own spatial RMS puts the coefficients on a common scale;
        # stop_gradient keeps this a normalization and not an extra path.
        red = list(range(1, 1 + K))
        rms = tf.sqrt(tf.reduce_mean(tf.square(F), axis=red, keepdims=True) + 1e-8)
        F = F / tf.stop_gradient(rms)
    return F
