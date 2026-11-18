"""Per-axis closed-form tables for TP-ADA.

One `Axis` = one spatial or temporal direction, carrying
  * a panel family (piecewise-linear "hat" nodes, or piecewise-constant cells)
    on a uniform partition of the reference interval,
  * an orthogonal mode family (Legendre P_n, or Fourier {1, cos, sin}),
  * the exact L2 projection matrix  C[m, i] = <panel_i, mode_m>/<mode_m, mode_m>,
  * closed-form evaluation of every mode after `k` anti-derivatives and `d`
    derivatives, at arbitrary points of the physical domain,
  * boundary-constraint rows (linear functionals of the panel DOF) and the
    null-space projector that enforces them exactly (ADA-L construction).

Conventions
  * Reference coordinate xi in [-1, 1]; physical x in [lo, hi];
    xi = -1 + 2 (x - lo)/(hi - lo),  scale = d xi/d x = 2/(hi - lo).
  * Anti-derivatives are taken from the axis base point xi = -1 (x = lo) and
    vanish there together with all lower-order ones: this is what makes hard
    IC/BC-at-lo structural.
  * A quantity obtained after k anti-derivatives and d derivatives in physical
    coordinates carries the factor scale**(d - k).
  * Legendre modes are handled through their Legendre-coefficient vectors
    (numpy.polynomial.legendre legint/legder/legval): exact modulo rounding,
    stable at order ~10^2. Fourier modes use elementary closed forms in
    tau = xi + 1 in [0, 2], modal frequency k_n = pi * n (period 2 = the full
    reference interval, matching NTO-ADA's period-L convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy.polynomial import legendre as L

__all__ = ["AxisSpec", "Axis", "null_projector"]


# ---------------------------------------------------------------------------
# closed-form segment integrals (Fourier projection with PL/PC panels)
# ---------------------------------------------------------------------------

def _seg_trig(a: float, b: float, k: float) -> tuple[float, float, float, float]:
    """(int cos, int tau*cos, int sin, int tau*sin) over tau in [a, b], k > 0."""
    sa, sb, ca, cb = np.sin(k * a), np.sin(k * b), np.cos(k * a), np.cos(k * b)
    i_c = (sb - sa) / k
    i_s = (ca - cb) / k
    i_tc = (b * sb - a * sa) / k + (cb - ca) / k**2
    i_ts = (a * ca - b * cb) / k + (sb - sa) / k**2
    return i_c, i_tc, i_s, i_ts


def _hat_pieces(nodes: np.ndarray, i: int):
    """Linear pieces (a, b, alpha, beta) with h_i(tau) = alpha + beta*tau on [a,b]."""
    out = []
    if i > 0:
        a, b = nodes[i - 1], nodes[i]
        out.append((a, b, -a / (b - a), 1.0 / (b - a)))      # ascending
    if i < nodes.size - 1:
        a, b = nodes[i], nodes[i + 1]
        out.append((a, b, b / (b - a), -1.0 / (b - a)))      # descending
    return out


def _cell_pieces(nodes: np.ndarray, i: int):
    """Indicator of cell i: constant 1 on [nodes[i], nodes[i+1]]."""
    return [(nodes[i], nodes[i + 1], 1.0, 0.0)]


# ---------------------------------------------------------------------------
# axis definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisSpec:
    """Static description of one TP-ADA axis.

    kind      : 'legendre' (non-periodic; supports hard BC) | 'fourier' (periodic).
    n_panels  : number of uniform panels on the axis.
    n_modes   : legendre -> max polynomial order N (n_coef = N+1);
                fourier  -> number of harmonics H (n_coef = 1 + 2H).
    panel     : 'linear' (nodal hats, n_dof = n_panels+1)
              | 'constant' (cells, n_dof = n_panels; NTO-ADA-compatible).
    k         : anti-derivative order applied in synthesis (0, 1, or 2).
    domain    : physical interval (lo, hi).
    """
    kind: str
    n_panels: int
    n_modes: int
    panel: str = "linear"
    k: int = 0
    domain: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self):
        if self.kind not in ("legendre", "fourier"):
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.panel not in ("linear", "constant"):
            raise ValueError(f"unknown panel {self.panel!r}")
        if not 0 <= self.k <= 2:
            raise ValueError("k (anti-derivative order) must be 0, 1, or 2")
        if self.domain[1] <= self.domain[0]:
            raise ValueError("domain must be increasing")


@dataclass(frozen=True)
class Axis:
    spec: AxisSpec
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # -- basic sizes -------------------------------------------------------
    @property
    def n_dof(self) -> int:
        return self.spec.n_panels + (1 if self.spec.panel == "linear" else 0)

    @property
    def n_coef(self) -> int:
        return (self.spec.n_modes + 1 if self.spec.kind == "legendre"
                else 1 + 2 * self.spec.n_modes)

    @property
    def scale(self) -> float:
        lo, hi = self.spec.domain
        return 2.0 / (hi - lo)

    def to_ref(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.spec.domain
        return -1.0 + 2.0 * (np.asarray(x, dtype=float) - lo) / (hi - lo)

    def _pieces(self, nodes: np.ndarray, i: int):
        return (_hat_pieces if self.spec.panel == "linear" else _cell_pieces)(nodes, i)

    # -- projection matrix C (n_coef x n_dof) ------------------------------
    @cached_property
    def C(self) -> np.ndarray:
        if self.spec.kind == "legendre":
            return self._legendre_C()
        return self._fourier_C()

    def _legendre_C(self) -> np.ndarray:
        """C[n, i] = (2n+1)/2 * int h_i(xi) P_n(xi) dxi, exact via Legendre algebra."""
        N, nd = self.spec.n_modes, self.n_dof
        nodes = np.linspace(-1.0, 1.0, self.spec.n_panels + 1)
        C = np.zeros((N + 1, nd))
        for n in range(N + 1):
            pn = np.zeros(n + 1)
            pn[n] = 1.0
            I_pn = L.legint(pn)              # antiderivative of P_n (constant cancels
            I_xpn = L.legint(L.legmulx(pn))  # in definite integrals below)
            for i in range(nd):
                acc = 0.0
                for a, b, al, be in self._pieces(nodes, i):
                    acc += al * (L.legval(b, I_pn) - L.legval(a, I_pn))
                    acc += be * (L.legval(b, I_xpn) - L.legval(a, I_xpn))
                C[n, i] = (2 * n + 1) / 2.0 * acc
        return C

    def _fourier_C(self) -> np.ndarray:
        """Rows: [a0; a_1..a_H; b_1..b_H] with f = a0 + sum a_n cos(k_n tau) + b_n sin(k_n tau),
        tau = xi + 1 in [0, 2], k_n = pi n. a0 = (1/2) int f; a_n, b_n = int f * trig (2/L, L=2)."""
        H, nd = self.spec.n_modes, self.n_dof
        nodes = np.linspace(0.0, 2.0, self.spec.n_panels + 1)   # tau nodes
        C = np.zeros((1 + 2 * H, nd))
        for i in range(nd):
            for a, b, al, be in self._pieces(nodes, i):
                C[0, i] += 0.5 * (al * (b - a) + be * (b * b - a * a) / 2.0)
                for n in range(1, H + 1):
                    i_c, i_tc, i_s, i_ts = _seg_trig(a, b, np.pi * n)
                    C[n, i] += al * i_c + be * i_tc
                    C[H + n, i] += al * i_s + be * i_ts
        return C

    # -- mode evaluation ----------------------------------------------------
    def eval(self, x: np.ndarray, deriv: int = 0) -> np.ndarray:
        """E[p, m]: mode m after `spec.k` anti-derivatives and `deriv` derivatives,
        at physical points x. Physical scaling included. Field values on x are
        then `A @ E.T` (or the synthesis einsums for multi-axis tensors)."""
        key = ("E", tuple(np.atleast_1d(np.asarray(x, float)).tolist()), deriv)
        if key in self._cache:
            return self._cache[key]
        xi = np.atleast_1d(self.to_ref(x))
        if self.spec.kind == "legendre":
            E = self._legendre_eval(xi, self.spec.k, deriv)
        else:
            E = self._fourier_eval(xi, self.spec.k, deriv)
        E = E * self.scale ** (deriv - self.spec.k)
        self._cache[key] = E
        return E

    def _legendre_eval(self, xi: np.ndarray, k: int, d: int) -> np.ndarray:
        """Exact: P_n -> legint^k (vanishing at -1 with all lower orders) -> legder^d."""
        N = self.spec.n_modes
        E = np.empty((xi.size, N + 1))
        for n in range(N + 1):
            c = np.zeros(n + 1)
            c[n] = 1.0
            if k:
                c = L.legint(c, m=k, lbnd=-1.0)   # I^(j)(-1) = 0 for all j <= k
            if d:
                c = L.legder(c, m=d)
            E[:, n] = L.legval(xi, c)
        return E

    def _fourier_eval(self, xi: np.ndarray, k: int, d: int) -> np.ndarray:
        """Modes in tau = xi+1: [1, cos(k_n tau), sin(k_n tau)]; anti-derivatives
        from tau=0 (vanishing there), then derivatives — all elementary."""
        H = self.spec.n_modes
        tau = xi + 1.0
        order = k - d  # net anti-derivative order per trig mode (see closed forms)
        if not -2 <= order <= 2:
            raise ValueError("fourier axis supports k - deriv in [-2, 2]")
        E = np.empty((tau.size, 1 + 2 * H))
        # DC mode: tau^k / k!  differentiated d times
        p = k - d
        if p < 0:
            E[:, 0] = 0.0
        else:
            from math import factorial
            E[:, 0] = tau**p / factorial(p)
        for n in range(1, H + 1):
            kn = np.pi * n
            c, s = np.cos(kn * tau), np.sin(kn * tau)
            if order == 0:
                ec, es = c, s
            elif order == 1:
                ec, es = s / kn, (1.0 - c) / kn
            elif order == 2:
                ec, es = (1.0 - c) / kn**2, (tau - s / kn) / kn
            elif order == -1:
                ec, es = -kn * s, kn * c
            else:  # order == -2
                ec, es = -(kn**2) * c, -(kn**2) * s
            E[:, n] = ec
            E[:, H + n] = es
        return E

    # -- boundary functionals & hard-BC projection --------------------------
    def constraint_row(self, side: str, deriv: int = 0) -> np.ndarray:
        """v with  (d^deriv u / dx^deriv)(endpoint) = v^T W  for this axis's DOF
        (after the axis's k anti-derivatives). side: 'lo' | 'hi'."""
        if self.spec.kind != "legendre":
            raise ValueError("hard BC rows only on legendre axes (fourier = periodic)")
        x = self.spec.domain[0] if side == "lo" else self.spec.domain[1]
        m = self.eval(np.array([x]), deriv=deriv)[0]      # functional on coefficients
        return self.C.T @ m                                # -> functional on panel DOF

    def bc(self, conditions: list[tuple[str, int]]):
        """Build (B, pinv, Pi) for conditions = [(side, deriv), ...]:
        B W = c  enforced by  W = pinv @ c + Pi @ W_tilde  (exact, penalty-free)."""
        B = np.stack([self.constraint_row(s, d) for s, d in conditions])
        return null_projector(B)


def null_projector(B: np.ndarray):
    """(B, B^+, I - B^+ B) with B^+ = B^T (B B^T)^{-1} (full row rank assumed)."""
    Bp = B.T @ np.linalg.inv(B @ B.T)
    return B, Bp, np.eye(B.shape[1]) - Bp @ B
