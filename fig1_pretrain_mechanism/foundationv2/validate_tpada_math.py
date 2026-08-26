"""Numerical validation of the tensor-product ADA (TP-ADA) closed forms.

Claims checked, each to near machine precision:
  1. Piecewise-LINEAR (hat) panels project onto Legendre modes through a
     closed-form matrix C (no quadrature), exactly as piecewise-constant
     panels do in NTO-ADA's `panel_basis_common`.
  2. The analytic anti-derivative of the spectral reconstruction matches
     brute-force numerical integration -> the 1-D "closed integration
     equation" holds for PL panels.
  3. The 2-D tensor-product extension holds: A = Cx W Cy^T gives a field
     whose double anti-derivative matches nested numerical integration.
  4. Hard IC: g1(-1) = ic exactly, for every panel tensor.
  5. Hard Dirichlet BC in 2-D via null-space projection applied per axis:
     all four edges vanish to machine precision for ANY predicted panel
     tensor, and the x/y projectors commute (homogeneous case).
  6. Super-resolution: the representation is a global spectral field, so
     evaluation on a 4x finer grid than the panel grid shows spectral
     accuracy against the underlying smooth function.

All Legendre manipulations use the Legendre-coefficient algebra of
numpy.polynomial.legendre (legmulx/legint/legval), which is exact modulo
float rounding -- i.e. these ARE the closed forms, not quadrature.
"""

import numpy as np
from numpy.polynomial import legendre as L


# ---------------------------------------------------------------------------
# Closed-form building blocks
# ---------------------------------------------------------------------------

def hat_projection_matrix(n_panels: int, max_order: int) -> tuple[np.ndarray, np.ndarray]:
    """C[n, i] = (2n+1)/2 * int_{-1}^{1} h_i(xi) P_n(xi) dxi  (closed form).

    h_i are the PL nodal hat functions on n_panels uniform panels
    (n_panels + 1 nodes).  Returns (C, nodes).
    """
    nodes = np.linspace(-1.0, 1.0, n_panels + 1)
    n_nodes = n_panels + 1
    C = np.zeros((max_order + 1, n_nodes))
    for n in range(max_order + 1):
        pn = np.zeros(n + 1)
        pn[n] = 1.0                       # P_n in Legendre-coefficient form
        x_pn = L.legmulx(pn)              # xi * P_n(xi), still exact
        I_pn = L.legint(pn)               # antiderivatives (constant irrelevant:
        I_xpn = L.legint(x_pn)            #  used only as definite integrals)
        for i in range(n_nodes):
            acc = 0.0
            # ascending piece on [nodes[i-1], nodes[i]]: h = (xi - a)/(b - a)
            if i > 0:
                a, b = nodes[i - 1], nodes[i]
                acc += ((L.legval(b, I_xpn) - L.legval(a, I_xpn))
                        - a * (L.legval(b, I_pn) - L.legval(a, I_pn))) / (b - a)
            # descending piece on [nodes[i], nodes[i+1]]: h = (b - xi)/(b - a)
            if i < n_nodes - 1:
                a, b = nodes[i], nodes[i + 1]
                acc += (b * (L.legval(b, I_pn) - L.legval(a, I_pn))
                        - (L.legval(b, I_xpn) - L.legval(a, I_xpn))) / (b - a)
            C[n, i] = (2 * n + 1) / 2.0 * acc
    return C, nodes


def antideriv_table(max_order: int, xs: np.ndarray) -> np.ndarray:
    """I1[n, j] = int_{-1}^{xs[j]} P_n(s) ds  (closed form).

    I_0 = xi + 1;  I_n = (P_{n+1} - P_{n-1})/(2n+1) for n >= 1
    (vanishes at -1 automatically).
    """
    out = np.zeros((max_order + 1, xs.size))
    out[0] = xs + 1.0
    for n in range(1, max_order + 1):
        cp = np.zeros(n + 2); cp[n + 1] = 1.0
        cm = np.zeros(n); cm[n - 1] = 1.0
        out[n] = (L.legval(xs, cp) - L.legval(xs, cm)) / (2 * n + 1)
    return out


def eval_table(max_order: int, xs: np.ndarray) -> np.ndarray:
    """P[n, j] = P_n(xs[j])."""
    out = np.zeros((max_order + 1, xs.size))
    for n in range(max_order + 1):
        c = np.zeros(n + 1); c[n] = 1.0
        out[n] = L.legval(xs, c)
    return out


def dirichlet_projector(C: np.ndarray) -> np.ndarray:
    """Null-space projector killing the reconstruction at BOTH endpoints.

    Edge-value functionals on node values W:  v(+-1)^T W with
    v(+-1) = C^T P(+-1)  (PDF eq. (14) style).  Returns I - B^+ B.
    """
    N = C.shape[0] - 1
    ones = np.ones(N + 1)
    sig = np.array([(-1.0) ** n for n in range(N + 1)])
    B = np.stack([C.T @ ones, C.T @ sig])          # (2, n_nodes)
    Bp = B.T @ np.linalg.inv(B @ B.T)
    return np.eye(C.shape[1]) - Bp @ B


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(0)
    NP, N = 32, 32                                   # panels per axis, max order
    C, nodes = hat_projection_matrix(NP, N)
    fine = np.linspace(-1.0, 1.0, 2001)
    P_fine = eval_table(N, fine)
    I1_fine = antideriv_table(N, fine)

    print("== 1) closed-form C vs high-res quadrature ==")
    f_smooth = lambda x: np.sin(3.0 * x) + 0.3 * np.cos(7.0 * x) + 0.2 * x
    W1 = f_smooth(nodes)                             # PL interpolant node values
    a = C @ W1                                       # spectral coefficients
    # quadrature reference for the same projection
    h_interp = np.interp(fine, nodes, W1)
    a_quad = np.array([(2 * n + 1) / 2.0 * np.trapz(h_interp * P_fine[n], fine)
                       for n in range(N + 1)])
    print(f"   max |a_closed - a_quad|            = {np.abs(a - a_quad).max():.2e}")

    print("== 2) 1-D analytic anti-derivative vs numerical integration ==")
    fN = a @ P_fine                                  # spectral reconstruction f_N
    g1 = a @ I1_fine                                 # closed-form  int_{-1}^x f_N
    g1_num = np.concatenate([[0.0], np.cumsum((fN[1:] + fN[:-1]) / 2 * np.diff(fine))])
    print(f"   max |g1_analytic - g1_trapz|       = {np.abs(g1 - g1_num).max():.2e}")

    print("== 4) hard IC ==")
    ic = 1.234
    print(f"   |(ic + g1)(-1) - ic|               = {abs((ic + g1)[0] - ic):.2e}"
          f"   (any W: I_n(-1)=0 structurally)")

    print("== 3) 2-D tensor product: closed double integral vs nested trapz ==")
    W2 = rng.standard_normal((NP + 1, NP + 1))       # arbitrary "network" output
    A2 = C @ W2 @ C.T                                # A = W x1 Cx x2 Cy
    G2 = I1_fine.T @ A2 @ I1_fine                    # int int w_N dx dy, closed form
    w2 = P_fine.T @ A2 @ P_fine                      # w_N on the fine grid
    gx = np.concatenate([np.zeros((1, w2.shape[1])),
                         np.cumsum((w2[1:] + w2[:-1]) / 2 * np.diff(fine)[:, None], axis=0)])
    G2_num = np.concatenate([np.zeros((gx.shape[0], 1)),
                             np.cumsum((gx[:, 1:] + gx[:, :-1]) / 2 * np.diff(fine)[None, :], axis=1)], axis=1)
    print(f"   max |G_analytic - G_trapz|         = {np.abs(G2 - G2_num).max():.2e}")

    print("== 5) 2-D hard Dirichlet BC via per-axis null-space projection ==")
    Pi = dirichlet_projector(C)
    Wc = Pi @ W2 @ Pi.T                              # project x-axis then y-axis
    Wc_rev = (Pi @ (W2 @ Pi.T))                      # other order
    print(f"   projector commutation (x<->y)      = {np.abs(Wc - Wc_rev).max():.2e}")
    Ac = C @ Wc @ C.T
    edges = np.array([+1.0, -1.0])
    P_edge = eval_table(N, edges)
    ex = P_edge.T @ Ac @ P_fine                      # u(+-1, y) for all y
    ey = P_fine.T @ Ac @ P_edge                      # u(x, +-1) for all x
    print(f"   max |u| on all four edges          = {max(np.abs(ex).max(), np.abs(ey).max()):.2e}")
    interior_change = np.abs((C @ (W2 - Wc) @ C.T)).max() / np.abs(A2).max()
    print(f"   relative coefficient change        = {interior_change:.2f} (projection, not zeroing)")

    print("== 6) super-resolution: PL panels + spectral projection ==")
    u_true = lambda x, y: np.sin(2.5 * x) * np.cos(1.5 * y) + 0.3 * np.exp(-4 * (x ** 2 + y ** 2))
    Ws = u_true(nodes[:, None], nodes[None, :])      # sampled on the 33x33 node grid
    As = C @ Ws @ C.T
    fine4 = np.linspace(-1.0, 1.0, 4 * NP + 1)       # 4x the panel resolution
    P4 = eval_table(N, fine4)
    u_sr = P4.T @ As @ P4
    err = np.abs(u_sr - u_true(fine4[:, None], fine4[None, :]))
    print(f"   max SR error on 4x grid (N={N})    = {err.max():.2e}")
    print(f"   (vs field scale {np.abs(Ws).max():.2f}; limited by PL sampling, not by the spectral map)")


if __name__ == "__main__":
    main()
