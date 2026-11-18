"""Differentiable Jacobi anti-derivative (ADA) spatial basis, trainable in (alpha, beta).

Second spatial basis to run in PARALLEL with the existing Fourier basis (the encoder learns,
via a zero-init gate, which one a family uses). Jacobi P_n^{(a,b)} generalizes Legendre (a=b=0)
and its two shape parameters control the weight (1-x)^a (1+x)^b — i.e. resolution emphasis near
each boundary (walls / boundary layers). Sturm-Liouville eigenfunctions => PDE-natural.

KEY TRICK (differentiability + stability at a=b=0):
  represent each Jacobi mode P_n^{(a,b)} in the LEGENDRE-coefficient basis via the three-term
  recurrence, where `x*` is the fixed linear map `legmulx` (a constant matrix M) and the
  recurrence scalars c1..c4(a,b) are differentiable in (a,b). Then
    * anti-derivative (vanishing at xi=-1, hard BC) = fixed linear map J = legint(lbnd=-1),
    * evaluation at query points = Legendre Vandermonde V(xi) @ coeffs.
  This avoids the param-shift integral formula (which is singular at a=b=0) and is EXACT
  Legendre at a=b=0 (validated: P_n^{(0,0)} -> e_n, err ~1e-15).

Public: JacobiADA(n_modes, n_panels).build_tables(alpha, beta) -> (C_J, E_centers) TF tensors,
and eval(alpha, beta, xi) -> mode values (after 1 anti-derivative) at arbitrary xi, for the
parallel Jacobi path in synthesis. alpha,beta are tf scalars (trainable Variables upstream).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from numpy.polynomial import legendre as L


def _legmulx_matrix(N: int) -> np.ndarray:
    """Constant matrix M (N+2, N+1): maps Legendre coeffs c -> coeffs of (x * sum c_k L_k).
    Uses the exact Legendre recurrence x L_n = ((n+1)L_{n+1} + n L_{n-1})/(2n+1)."""
    M = np.zeros((N + 2, N + 1))
    for n in range(N + 1):
        e = np.zeros(n + 1); e[n] = 1.0
        xe = L.legmulx(e)                       # coeffs of x*L_n (length n+2)
        M[:len(xe), n] = xe
    return M


def _legint_matrix(N: int) -> np.ndarray:
    """Constant matrix J (N+2, N+1): Legendre coeffs -> coeffs of the anti-derivative that
    VANISHES at xi=-1 (hard BC at the low boundary). legint(c, m=1, lbnd=-1)."""
    J = np.zeros((N + 2, N + 1))
    for n in range(N + 1):
        e = np.zeros(n + 1); e[n] = 1.0
        Ie = L.legint(e, m=1, lbnd=-1.0)        # length n+2
        J[:len(Ie), n] = Ie
    return J


def _leg_vandermonde(xi: np.ndarray, N: int) -> np.ndarray:
    """V (len(xi), N+1): Legendre polynomials L_0..L_N evaluated at xi (for coeffs -> values)."""
    xi = np.asarray(xi, np.float64)
    V = np.empty((xi.size, N + 1))
    for n in range(N + 1):
        e = np.zeros(n + 1); e[n] = 1.0
        V[:, n] = L.legval(xi, e)
    return V


class JacobiADA:
    """Fixed panel/mode geometry for one axis; produces differentiable (in alpha,beta) Jacobi
    anti-derivative tables. n_modes = highest Jacobi order N (n_coef = N+1)."""

    def __init__(self, n_modes: int, n_panels: int, panel: str = "constant"):
        self.N = int(n_modes)
        self.n_panels = int(n_panels)
        self.panel = panel
        # fixed constant maps in Legendre-coeff space (padded to degree N+1 headroom)
        Np = self.N + 1
        self._M = tf.constant(_legmulx_matrix(Np)[: Np + 1, : Np], tf.float32)      # x*
        self._J = tf.constant(_legint_matrix(self.N)[: self.N + 2, : self.N + 1], tf.float32)  # int
        # panel->legendre projection of the PANEL functions (constant, basis-independent):
        # reuse the existing exact Legendre C (panels -> Legendre coeffs) at order N.
        from tpada.basis import Axis, AxisSpec
        self._axis = Axis(AxisSpec("legendre", n_panels, self.N, panel=panel, k=0))
        self._C_leg = tf.constant(self._axis.C, tf.float32)                          # (N+1, n_dof)

    # ---- Jacobi modes as Legendre-coeff matrix P(a,b): row n = P_n^{(a,b)} in Legendre basis ----
    def _leg_coeffs(self, a, b):
        """Return (N+1, N+1) lower-triangular-ish Legendre-coeff matrix, differentiable in a,b."""
        N = self.N
        rows = []
        p0 = tf.scatter_nd([[0]], [1.0], [N + 1])                     # L-coeffs of P_0 = 1
        rows.append(p0)
        if N >= 1:
            p1 = tf.scatter_nd([[0], [1]], [0.5 * (a - b), 0.5 * (a + b + 2.0)], [N + 1])
            rows.append(p1)
        for k in range(2, N + 1):
            n = k - 1
            c1 = 2.0 * (n + 1) * (n + a + b + 1) * (2 * n + a + b)
            c2 = (2 * n + a + b + 1) * (a * a - b * b)
            c3 = (2 * n + a + b) * (2 * n + a + b + 1) * (2 * n + a + b + 2)
            c4 = 2.0 * (n + a) * (n + b) * (2 * n + a + b + 2)
            xPk1 = tf.linalg.matvec(self._M[: N + 1, : N + 1], rows[k - 1])   # x * P_{k-1}
            term = (c2 * rows[k - 1] + c3 * xPk1 - c4 * rows[k - 2]) / c1
            rows.append(term)
        return tf.stack(rows, 0)                                       # (N+1, N+1)

    def tables(self, alpha, beta, centers_xi):
        """Differentiable tables for synthesis, given trainable scalars alpha,beta.
        Returns:
          C_J (N+1, n_dof): panels -> Jacobi-mode coefficients (project panel DOF onto Jacobi modes),
          E_c (n_centers, N+1): Jacobi modes AFTER one anti-derivative, evaluated at box centers.
        The parallel Jacobi path then mirrors the Fourier path: A_J = W x C_J; field += E @ A_J."""
        P = self._leg_coeffs(alpha, beta)                             # (N+1,N+1) Legendre coeffs
        # panels -> Jacobi modes: express panel-Legendre-coeffs in the Jacobi mode set.
        # C_leg: panels -> Legendre coeffs (N+1, n_dof). Jacobi modes P (N+1 Legendre coeffs).
        # coefficients of panels in Jacobi basis = P^{-1}^T @ C_leg  (P maps Jacobi->Legendre).
        Pinv = tf.linalg.inv(P + 1e-6 * tf.eye(self.N + 1))           # Jacobi<-Legendre (a=b=0 -> I)
        # change of basis: A = P^{-T} g (g=Legendre coeffs). C_J = P^{-T} @ C_leg (NOT Pinv@C_leg —
        # validated: Pinv@C round-trips wrong for a,b!=0; the transpose is required).
        C_J = tf.linalg.matmul(tf.transpose(Pinv), self._C_leg)      # (N+1, n_dof)
        # anti-derivative eval at centers: J @ (Legendre coeffs of each Jacobi mode), then legval
        Vc = tf.constant(_leg_vandermonde(centers_xi, self.N + 1), tf.float32)   # (nc, N+2)
        Icoef = tf.linalg.matmul(self._J, tf.transpose(P))           # (N+2, N+1): anti-deriv coeffs
        E_c = tf.linalg.matmul(Vc, Icoef)                            # (nc, N+1)
        return C_J, E_c

    def antideriv_legcoeffs(self, alpha, beta):
        """(N+2, N+1) : column n = the LEGENDRE coeffs of the anti-derivative (vanishing at -1) of
        the n-th Jacobi mode P_n^{(a,b)}. Differentiable in (alpha,beta). Synthesis evaluates the
        continuous point query as  leg_vandermonde_tf(xi) @ this  (Legendre Vandermonde built in TF
        by recurrence at the arbitrary query coords)."""
        P = self._leg_coeffs(alpha, beta)                            # (N+1,N+1)
        return tf.linalg.matmul(self._J, tf.transpose(P))            # (N+2, N+1)


# --------------------------------------------------------------------------- #
# numpy reference (validation only — mirrors the TF math for local checks)
# --------------------------------------------------------------------------- #
def _np_jacobi_synth_ref(alpha, beta, N, n_panels, W, xi_q, panel="constant"):
    """Reference: project panel DOF W onto Jacobi modes, reconstruct the ANTI-DERIVATIVE field at
    xi_q. Pure numpy; used to check the TF synthesis wiring at a=b=0 (== Legendre-ADA) and a,b!=0."""
    from tpada.basis import Axis, AxisSpec
    ax = Axis(AxisSpec("legendre", n_panels, N, panel=panel, k=0))
    C_leg = ax.C                                                     # (N+1, n_dof)
    # Jacobi modes in Legendre coeffs
    Pm = [np.eye(N + 1)[0]]
    if N >= 1:
        r = np.zeros(N + 1); r[0] = 0.5 * (alpha - beta); r[1] = 0.5 * (alpha + beta + 2); Pm.append(r)
    for k in range(2, N + 1):
        n = k - 1
        c1 = 2 * (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta)
        c2 = (2 * n + alpha + beta + 1) * (alpha * alpha - beta * beta)
        c3 = (2 * n + alpha + beta) * (2 * n + alpha + beta + 1) * (2 * n + alpha + beta + 2)
        c4 = 2 * (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2)
        xk = np.zeros(N + 1); xe = L.legmulx(Pm[k - 1]); xk[:len(xe)] = xe[:N + 1]
        Pm.append((c2 * Pm[k - 1] + c3 * xk - c4 * Pm[k - 2]) / c1)
    P = np.stack(Pm, 0)                                              # (N+1,N+1)
    C_J = np.linalg.inv(P).T @ C_leg                                 # panels -> Jacobi coeffs
    A = C_J @ W                                                     # (N+1,)
    Icoef = np.stack([L.legint(P[n], m=1, lbnd=-1.0) for n in range(N + 1)], -1)  # (N+2,N+1)
    V = np.stack([L.legval(xi_q, np.eye(N + 2)[m]) for m in range(N + 2)], -1)    # (Q,N+2)
    return (V @ Icoef) @ A                                          # field (anti-derivative) at xi_q
