"""Shared utilities for panel-based analytic basis modules.

The panel-based pattern is:
    1. The network outputs piecewise-constant samples  W (shape (..., N_p))
       of a physical quantity  f(t)  on a uniform partition of t in [0, T_seg].
    2. W is projected onto an analytic basis (Legendre / Fourier / ...) via a
       *fixed* (precomputed) projection matrix.
    3. Anti-derivatives of the basis functions, also precomputed analytically,
       are evaluated against the projection coefficients to recover smooth
       k-th anti-derivatives  g_k(t)  of f, with hard initial conditions
       enforced via integration constants.

The DOF the network controls is W.  Basis coefficients are *derived*
from W, not learned independently — this is what distinguishes a
panel-based LPA / ADAF from a plain spectral-coefficient Galerkin.

This file collects helpers shared between the LPA (Legendre) and ADAF
(Fourier) backends.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import tensorflow as tf


def horner_eval(poly_mat, x):
    """Evaluate a matrix of polynomials at points x via Horner's method.

    poly_mat : (n_basis, deg+1) ascending-power coefficients
    x        : (Nq,)
    returns  : (n_basis, Nq)
    """
    poly_mat = tf.convert_to_tensor(poly_mat)
    x = tf.cast(x, poly_mat.dtype)
    coeffs_rev = tf.reverse(poly_mat, axis=[1])
    y = tf.zeros((tf.shape(poly_mat)[0], tf.shape(x)[0]), dtype=poly_mat.dtype)
    for c in tf.unstack(coeffs_rev, axis=1):
        y = y * x[None, :] + c[:, None]
    return y


def differentiate_poly_matrix(poly_mat):
    """Coefficient-level differentiation: rows = bases, cols = ascending powers."""
    poly_mat = np.asarray(poly_mat)
    _, width = poly_mat.shape
    out = np.zeros_like(poly_mat)
    for k in range(1, width):
        out[:, k - 1] = k * poly_mat[:, k]
    return out


def polynomial_ic_correction(t, ics, k):
    """Polynomial IC term for the k-th anti-derivative.

        Σ_{j=0}^{k-1}  ics[j] · t^{k-1-j} / (k-1-j)!

    With this, if G_k(t) is the analytic anti-derivative built so that
    G_k(0) = 0, the full k-th anti-derivative is
        g_k(t) = G_k(t) + poly(t)
    and satisfies  g_k(0) = ics[k-1],  g_k'(0) = ics[k-2], ...,
    g_k^{(k-1)}(0) = ics[0].

    Parameters
    ----------
    t   : (Nt,) tensor
    ics : sequence of length >= k.  ics[j] has any leading shape (...,).
    k   : int >= 1
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(ics) < k:
        raise ValueError(f"need >= {k} initial conditions, got {len(ics)}")
    pieces = []
    for j in range(k):
        ic_j = tf.cast(ics[j], t.dtype)
        power = k - 1 - j
        fact = float(math.factorial(power))
        pieces.append(ic_j[..., None] * (t ** power) / fact)
    return tf.add_n(pieces)


def zeros_like_batch(reference, dtype):
    """Shape (...,) zeros taking leading batch dims from `reference` (which is (..., N))."""
    return tf.zeros(tf.shape(reference)[:-1], dtype=dtype)


def normalize_ics(ics, n_integrations, like_tensor, dtype):
    """Validate / pad the IC list. Empty or None -> all-zero ICs of correct shape."""
    if ics is None or (hasattr(ics, "__len__") and len(ics) == 0):
        z = zeros_like_batch(like_tensor, dtype)
        return [z for _ in range(n_integrations)]
    if len(ics) != n_integrations:
        raise ValueError(
            f"expected {n_integrations} IC tensors, got {len(ics)}"
        )
    return [tf.cast(ic, dtype) for ic in ics]
