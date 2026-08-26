"""Legendre Panel Analytic basis (LPA).

A drop-in module for any problem (ODE / PDE latent dynamics / control /
trajectory generation) that wants the panel-based LPA construction.

Network output:
    W : (..., N_p) panel values (piecewise-constant samples) of a function
                   f(t) on a uniform partition of t in [0, T_seg].

Returned by __call__:
    {
      "f"     : (..., Nt)  smooth basis reconstruction of f
      "df"    : (..., Nt)  df/dt              (if return_derivative=True)
      "g1"    : (..., Nt)  ∫₀^t f ds                with g1(0) = ics[0]
      "g2"    : (..., Nt)  ∫₀^t∫₀^s f dr ds         with g2(0) = ics[1]
      ...
      "g{n}"  : (..., Nt)  n-th anti-derivative     with gn(0) = ics[n-1]
      "t"     : (Nt,)
    }

Construction:
    f(t) = a0·P_0(ξ(t)) + Σ_{n=1}^{max_order} A[n]·P_n(ξ(t))
    where
        ξ(t) = -1 + 2γ · t/T_seg                  (linear time-to-ξ map)
        a0   = mean(W)                            (panel mean = DC)
        A[n] = Σ_i coef_mat[n, i] · W[i]          (panel→Legendre projection,
                                                   coef_mat is sympy-precomputed)

    Anti-derivatives use sympy-precomputed integrated Legendre polynomials
    I^(k)_n(ξ) chosen so that I^(k)_n(-1) = 0; chained with the appropriate
    1/scale factors (scale = dξ/dt = 2γ/T_seg) and the polynomial IC
    correction in t.

Notes
-----
* The DOF is W (panel values).  Legendre coefficients are *derived*
  from W via a fixed projection — they are NOT learned independently.
  This is the whole point of LPA and what distinguishes it from a
  plain Legendre-Galerkin ansatz.
* The default γ=1 covers all of ξ ∈ [-1, +1].  γ < 1 concentrates
  panels into ξ ∈ [-1, 2γ-1] and pads the tail; useful for stiff
  initial transients.
* For most problems N_p = max_order + 1 is a good choice.
"""
from __future__ import annotations

from typing import Optional, Sequence, Dict

import numpy as np
import sympy as sp
import tensorflow as tf

from .panel_basis_common import (
    horner_eval,
    differentiate_poly_matrix,
    polynomial_ic_correction,
    normalize_ics,
)


# ===================================================================== #
# symbolic helpers
# ===================================================================== #
def _build_legendre_anti_derivative_tables(max_order, n_integrations, common_width):
    """Numeric coefficient tables for P_n, dP_n, and I^(k)_n
    (k = 1..n_integrations), with I^(k)_n(-1) = 0 convention."""
    x = sp.symbols("x", real=True)

    def to_row(expr):
        poly = sp.Poly(sp.expand(expr), x)
        out = np.zeros(common_width, dtype=np.float64)
        for pwr, coeff in poly.terms():
            if pwr[0] < common_width:
                out[pwr[0]] = float(coeff)
        return out

    P_rows = []
    Ik_rows = [[] for _ in range(n_integrations)]
    for n in range(max_order + 1):
        Pn = sp.legendre(n, x)
        P_rows.append(to_row(Pn))
        cur = Pn
        for k in range(n_integrations):
            cur = sp.integrate(cur, x)
            cur = sp.expand(cur - cur.subs(x, -1))   # ensure I^(k)_n(-1) = 0
            Ik_rows[k].append(to_row(cur))

    P_mat = np.stack(P_rows, axis=0)
    Ik_mats = [np.stack(rows, axis=0) for rows in Ik_rows]
    dP_mat = differentiate_poly_matrix(P_mat)
    return P_mat, dP_mat, Ik_mats


def _legendre_panel_projection(max_order, panel_edges):
    """coef_mat[n-1, i] = n-th Legendre coefficient of indicator on panel i,
    for n = 1..max_order.  Shape (max_order, N_p).

    For n = 0 (DC), the projection is simply 1/N_p (panel mean) and is
    handled separately as a0 = mean(W)."""
    x = sp.symbols("x", real=True)
    rows = []
    for n in range(1, max_order + 1):
        P = sp.legendre(n, x)
        Pint = sp.integrate(P, x)
        vals = np.array([float(Pint.subs(x, s)) for s in panel_edges],
                        dtype=np.float64)
        coefs = (vals[1:] - vals[:-1]) * (2.0 * n + 1.0) / 2.0
        rows.append(coefs)
    return np.stack(rows, axis=0)


# ===================================================================== #
# main module
# ===================================================================== #
class LegendrePanelBasis(tf.Module):
    """Panel-projection Legendre basis with analytic anti-derivatives."""

    def __init__(
        self,
        T_seg: float = 1.0,
        Nt: int = 257,
        N_p: int = 11,
        max_order: Optional[int] = None,
        n_integrations: int = 1,
        gamma: float = 1.0,
        return_derivative: bool = True,
        dtype=tf.float32,
        name=None,
    ):
        super().__init__(name=name)
        if max_order is None:
            max_order = N_p - 1
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        if not (-1.0 < gamma <= 1.0):
            raise ValueError("gamma must satisfy -1 < gamma <= 1")
        if n_integrations < 0:
            raise ValueError("n_integrations must be >= 0")

        self.dtype_ = dtype
        self.T_seg = float(T_seg)
        self.Nt = int(Nt)
        self.N_p = int(N_p)
        self.max_order = int(max_order)
        self.n_integrations = int(n_integrations)
        self.gamma = float(gamma)
        self.return_derivative = bool(return_derivative)

        # time grid + ξ(t)
        t_np = np.linspace(0.0, self.T_seg, self.Nt).astype(np.float64)
        xi_np = -1.0 + 2.0 * self.gamma * (t_np / self.T_seg)
        self._t = tf.constant(t_np, dtype=dtype)
        self.time_scale = tf.constant(2.0 * self.gamma / self.T_seg, dtype=dtype)

        # panel edges in ξ
        if np.isclose(self.gamma, 1.0):
            panel_edges = np.linspace(-1.0, 1.0, self.N_p + 1)
        else:
            xi_active = -1.0 + 2.0 * self.gamma
            panel_edges = np.concatenate(
                [np.linspace(-1.0, xi_active, self.N_p), [1.0]]
            )

        # symbolic basis tables
        n_int_eff = max(self.n_integrations, 1)   # build at least 1 for derivative
        common_width = self.max_order + n_int_eff + 2
        P_mat, dP_mat, Ik_mats = _build_legendre_anti_derivative_tables(
            self.max_order, n_int_eff, common_width
        )

        # basis caches on the time grid (split DC from higher orders)
        self.P0_cache = tf.cast(horner_eval(P_mat[0:1], xi_np)[0], dtype)
        self.P_cache = tf.cast(horner_eval(P_mat[1:], xi_np), dtype)
        self.dP0_cache = tf.cast(horner_eval(dP_mat[0:1], xi_np)[0], dtype)
        self.dP_cache = tf.cast(horner_eval(dP_mat[1:], xi_np), dtype)

        self.Ik_0_caches = []
        self.Ik_caches = []
        for k in range(self.n_integrations):
            Imat = Ik_mats[k]
            self.Ik_0_caches.append(tf.cast(horner_eval(Imat[0:1], xi_np)[0], dtype))
            self.Ik_caches.append(tf.cast(horner_eval(Imat[1:], xi_np), dtype))

        # panel→Legendre projection
        coef_mat_np = _legendre_panel_projection(self.max_order, panel_edges)
        self.coef_mat = tf.constant(coef_mat_np, dtype=dtype)
        self.mean_vec = tf.constant(np.ones(self.N_p) / float(self.N_p), dtype=dtype)
        self.panel_edges_np = panel_edges

    @property
    def t(self) -> tf.Tensor:
        return self._t

    def __call__(
        self,
        W: tf.Tensor,
        ics: Optional[Sequence[tf.Tensor]] = None,
    ) -> Dict[str, tf.Tensor]:
        W = tf.cast(W, self.dtype_)

        # project W → (a0, A)
        a0 = tf.reduce_sum(W * self.mean_vec, axis=-1)                  # (...,)
        A = tf.einsum("op,...p->...o", self.coef_mat, W)                 # (..., max_order)

        out: Dict[str, tf.Tensor] = {"t": self._t}

        # f(t)
        f = (a0[..., None] * self.P0_cache
             + tf.einsum("...o,ot->...t", A, self.P_cache))
        out["f"] = f

        # df/dt
        if self.return_derivative:
            df_xi = (a0[..., None] * self.dP0_cache
                     + tf.einsum("...o,ot->...t", A, self.dP_cache))
            out["df"] = self.time_scale * df_xi

        # k-th anti-derivatives
        ics_list = normalize_ics(ics, self.n_integrations, W, self.dtype_)
        scale_inv = 1.0 / self.time_scale
        for k in range(self.n_integrations):
            kth = k + 1
            Ik_xi = (a0[..., None] * self.Ik_0_caches[k]
                     + tf.einsum("...o,ot->...t", A, self.Ik_caches[k]))
            ana = Ik_xi * (scale_inv ** kth)
            poly = polynomial_ic_correction(self._t, ics_list, kth)
            out[f"g{kth}"] = ana + poly

        return out
