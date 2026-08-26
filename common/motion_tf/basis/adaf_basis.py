"""Analytic Decomposition Anti-derivative Fourier basis (ADAF).

A drop-in module with the SAME API as ``lpa_basis.LegendrePanelBasis``,
but using a Fourier (sin / cos) basis instead of Legendre.  The two
backends are intended to be interchangeable in any problem.

Network output:
    W : (..., N_p) panel values (piecewise-constant samples) of a function
                   f(t) on a uniform partition of t in [0, T_seg].

Returned by __call__:
    {
      "f"     : (..., Nt)
      "df"    : (..., Nt)              (if return_derivative=True)
      "g1"    : (..., Nt)               g1(0) = ics[0]
      "g2"    : (..., Nt)               g2(0) = ics[1]
      ...
      "t"     : (Nt,)
    }

Construction:
    Let  k_n = 2πn / L,  L = T_seg.  Full Fourier expansion of period L
    on [0, L]  (basis is orthogonal on [0, L]):

        f(t) = a0  +  Σ_{n=1}^{n_modes} [ a_n cos(k_n t) + b_n sin(k_n t) ]

    Coefficients are derived from W as exact L² projections of the
    indicator-on-panel functions:

        a0   = mean(W)
        a_n  = (2/L) ∫₀^L W(s) cos(k_n s) ds   = (W · M_a^T)[n]
        b_n  = (2/L) ∫₀^L W(s) sin(k_n s) ds   = (W · M_b^T)[n]

    Anti-derivatives of cos(k_n t) and sin(k_n t) are closed-form (with
    integration constants chosen so each I^(k)(0) = 0); together with
    a0·t^k/k! for the DC, they give the smooth analytic g_k(t) before
    the polynomial IC correction.

Notes
-----
* As with LegendrePanelBasis, the DOF is W (panel samples) — Fourier
  coefficients are *derived*, not learned.
* Default n_modes = N_p // 2.  With period-L basis, n_modes much above
  N_p / 2 (Nyquist) brings little benefit and may amplify panel-jump
  noise.
* **Periodicity caveat:** the period-L basis implicitly enforces
  f(0) = f(T_seg) (it represents f as an L-periodic function).  For
  problems where this is unnatural (e.g. trajectory generation that does
  not close), LPA is a better default — the two backends are not
  equivalent for non-periodic signals.  ADAF excels for inherently
  periodic phenomena (oscillatory systems, gust loads, etc.).
* If your downstream physics expects the steady51-style half-range
  (period 2L, k_n = nπ/L) reconstruction, that basis is non-orthogonal
  on [0, L] and would need normal-equation solving — wrap that in a
  separate file rather than this one.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from .panel_basis_common import polynomial_ic_correction, normalize_ics


# ===================================================================== #
# closed-form integrals of cos(k t) and sin(k t)
# ===================================================================== #
def _cos_sin_anti_derivatives(k_arr, t, n_integrations):
    """Return lists I_cos[k-1], I_sin[k-1] of shape (n_modes, Nt) for
    k = 1..n_integrations, where each I^(k)(0) = 0.

    Closed forms used:
        I1_cos = sin(kt)/k
        I1_sin = (1 - cos(kt))/k
        I2_cos = (1 - cos(kt))/k²
        I2_sin = t/k - sin(kt)/k²
        I3_cos = t/k² - sin(kt)/k³
        I3_sin = t²/(2k) + (cos(kt) - 1)/k³
        ...
    Beyond k=3 we recurse with sympy.
    """
    k = k_arr[:, None]                 # (n_modes, 1)
    arg = k * t[None, :]               # (n_modes, Nt)
    sin_g = np.sin(arg)
    cos_g = np.cos(arg)

    Ic: List[np.ndarray] = []
    Is: List[np.ndarray] = []
    if n_integrations >= 1:
        Ic.append(sin_g / k)
        Is.append((1.0 - cos_g) / k)
    if n_integrations >= 2:
        Ic.append((1.0 - cos_g) / (k ** 2))
        Is.append(t[None, :] / k - sin_g / (k ** 2))
    if n_integrations >= 3:
        Ic.append(t[None, :] / (k ** 2) - sin_g / (k ** 3))
        Is.append((t[None, :] ** 2) / (2.0 * k) + (cos_g - 1.0) / (k ** 3))

    if n_integrations > 3:
        import sympy as sp
        s = sp.Symbol("s", real=True)
        for kth in range(4, n_integrations + 1):
            ic_rows = []
            is_rows = []
            for kn in k_arr:
                f_c = sp.cos(float(kn) * s)
                f_s = sp.sin(float(kn) * s)
                for _ in range(kth):
                    f_c = sp.integrate(f_c, s)
                    f_c = sp.expand(f_c - f_c.subs(s, 0))
                    f_s = sp.integrate(f_s, s)
                    f_s = sp.expand(f_s - f_s.subs(s, 0))
                fn_c = sp.lambdify(s, f_c, "numpy")
                fn_s = sp.lambdify(s, f_s, "numpy")
                ic_rows.append(np.asarray(fn_c(t), dtype=np.float64))
                is_rows.append(np.asarray(fn_s(t), dtype=np.float64))
            Ic.append(np.stack(ic_rows, axis=0))
            Is.append(np.stack(is_rows, axis=0))

    return Ic, Is


def _fourier_panel_projection(n_modes, T_seg, panel_edges):
    """Closed-form Fourier coefficients of indicator-on-panel functions.

    For panel j with edges [t2_j, t1_j]:
        ∫sin(k s) ds = (1/k)·(cos(k t2) - cos(k t1))
        ∫cos(k s) ds = (1/k)·(sin(k t1) - sin(k t2))
    So:
        a_n = (2/L) Σ_j W_j ∫panel cos(k_n s) ds = W · M_a^T   with
        M_a[n, j] = (2/L) · (sin(k_n t1_j) - sin(k_n t2_j)) / k_n
        M_b[n, j] = (2/L) · (cos(k_n t2_j) - cos(k_n t1_j)) / k_n
    """
    L = float(T_seg)
    n_arr = np.arange(1, n_modes + 1, dtype=np.float64)
    k = 2.0 * np.pi * n_arr / L                    # period-L: k_n = 2πn/L
    t1 = panel_edges[1:]                           # right edges
    t2 = panel_edges[:-1]                          # left edges
    Ma = (2.0 / L) * (np.sin(k[:, None] * t1[None, :])
                      - np.sin(k[:, None] * t2[None, :])) / k[:, None]
    Mb = (2.0 / L) * (np.cos(k[:, None] * t2[None, :])
                      - np.cos(k[:, None] * t1[None, :])) / k[:, None]
    return Ma, Mb, k


# ===================================================================== #
# main module
# ===================================================================== #
class FourierPanelBasis(tf.Module):
    """Panel-projection Fourier (cos/sin) basis with analytic anti-derivatives."""

    def __init__(
        self,
        T_seg: float = 1.0,
        Nt: int = 257,
        N_p: int = 11,
        n_modes: Optional[int] = None,
        n_integrations: int = 1,
        return_derivative: bool = True,
        dtype=tf.float32,
        name=None,
    ):
        super().__init__(name=name)
        if n_modes is None:
            n_modes = max(N_p // 2, 1)
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if n_integrations < 0:
            raise ValueError("n_integrations must be >= 0")

        self.dtype_ = dtype
        self.T_seg = float(T_seg)
        self.Nt = int(Nt)
        self.N_p = int(N_p)
        self.n_modes = int(n_modes)
        self.n_integrations = int(n_integrations)
        self.return_derivative = bool(return_derivative)

        t_np = np.linspace(0.0, self.T_seg, self.Nt).astype(np.float64)
        self._t = tf.constant(t_np, dtype=dtype)

        panel_edges = np.linspace(0.0, self.T_seg, self.N_p + 1)
        self.panel_edges_np = panel_edges

        # projection matrices and modal frequencies
        Ma_np, Mb_np, k_arr = _fourier_panel_projection(
            self.n_modes, self.T_seg, panel_edges
        )
        self.Ma = tf.constant(Ma_np, dtype=dtype)
        self.Mb = tf.constant(Mb_np, dtype=dtype)

        # basis values + first derivative on grid
        arg = k_arr[:, None] * t_np[None, :]
        cos_g = np.cos(arg)
        sin_g = np.sin(arg)
        self.cos_cache = tf.constant(cos_g, dtype=dtype)
        self.sin_cache = tf.constant(sin_g, dtype=dtype)
        # d/dt cos(kt) = -k sin(kt) ; d/dt sin(kt) = k cos(kt)
        self.dcos_cache = tf.constant(-k_arr[:, None] * sin_g, dtype=dtype)
        self.dsin_cache = tf.constant(k_arr[:, None] * cos_g, dtype=dtype)

        # anti-derivatives
        Ic, Is = _cos_sin_anti_derivatives(k_arr, t_np, self.n_integrations)
        self.Ik_cos_caches = [tf.constant(c, dtype=dtype) for c in Ic]
        self.Ik_sin_caches = [tf.constant(c, dtype=dtype) for c in Is]

        # DC: a0 = mean(W)
        self.mean_vec = tf.constant(np.ones(self.N_p) / float(self.N_p), dtype=dtype)

        # factorial table for DC anti-derivative coefficients (a0·t^k/k!)
        self._fact = [float(math.factorial(k + 1))
                      for k in range(self.n_integrations)]

    @property
    def t(self) -> tf.Tensor:
        return self._t

    def __call__(
        self,
        W: tf.Tensor,
        ics: Optional[Sequence[tf.Tensor]] = None,
    ) -> Dict[str, tf.Tensor]:
        W = tf.cast(W, self.dtype_)

        # project W → (a0, a_n, b_n)
        a0 = tf.reduce_sum(W * self.mean_vec, axis=-1)                  # (...,)
        a_n = tf.einsum("np,...p->...n", self.Ma, W)                     # (..., n_modes)
        b_n = tf.einsum("np,...p->...n", self.Mb, W)                     # (..., n_modes)

        out: Dict[str, tf.Tensor] = {"t": self._t}

        # f(t) = a0 + Σ_n [ a_n cos(k_n t) + b_n sin(k_n t) ]
        f = (a0[..., None] * tf.ones_like(self._t)
             + tf.einsum("...n,nt->...t", a_n, self.cos_cache)
             + tf.einsum("...n,nt->...t", b_n, self.sin_cache))
        out["f"] = f

        # df/dt = Σ_n [ -k_n a_n sin + k_n b_n cos ]
        if self.return_derivative:
            df = (tf.einsum("...n,nt->...t", a_n, self.dcos_cache)
                  + tf.einsum("...n,nt->...t", b_n, self.dsin_cache))
            out["df"] = df

        # k-th anti-derivatives
        ics_list = normalize_ics(ics, self.n_integrations, W, self.dtype_)
        for k in range(self.n_integrations):
            kth = k + 1
            dc_term = a0[..., None] * (self._t ** kth) / self._fact[k]
            cos_term = tf.einsum("...n,nt->...t", a_n, self.Ik_cos_caches[k])
            sin_term = tf.einsum("...n,nt->...t", b_n, self.Ik_sin_caches[k])
            ana = dc_term + cos_term + sin_term
            poly = polynomial_ic_correction(self._t, ics_list, kth)
            out[f"g{kth}"] = ana + poly

        return out
