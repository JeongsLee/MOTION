"""Machine-precision tests for the TP-ADA numerics module (Phase 0).

Runnable as  `pytest tests/`  or  `python tests/test_tpada.py`.
References are exact wherever possible (per-panel Gauss quadrature exact for
polynomial integrands, closed-form identities); quadrature-limited checks are
labeled as such and use accordingly looser tolerances.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tpada import Axis, AxisSpec, apply_bc, eval_grid, eval_points, project  # noqa: E402

RNG = np.random.default_rng(7)


def _gauss_panel_quad(f, lo, hi, n_sub, n_gauss=64):
    """Composite Gauss-Legendre quadrature: exact for per-panel polynomials."""
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    edges = np.linspace(lo, hi, n_sub + 1)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        x = (a + b) / 2 + (b - a) / 2 * xg
        total += (b - a) / 2 * np.sum(wg * f(x))
    return total


def _panel_fn(ax: Axis, W: np.ndarray):
    """The piecewise panel field w(xi) the DOF vector W represents (reference coords)."""
    if ax.spec.panel == "linear":
        nodes = np.linspace(-1, 1, ax.spec.n_panels + 1)
        return lambda xi: np.interp(xi, nodes, W)
    edges = np.linspace(-1, 1, ax.spec.n_panels + 1)
    idx = lambda xi: np.clip(np.searchsorted(edges, xi, side="right") - 1, 0, ax.spec.n_panels - 1)
    return lambda xi: W[idx(xi)]


# ---------------------------------------------------------------------------
# projection matrices are the exact closed forms
# ---------------------------------------------------------------------------

def test_legendre_projection_exact():
    for panel in ("linear", "constant"):
        ax = Axis(AxisSpec("legendre", n_panels=16, n_modes=16, panel=panel))
        W = RNG.standard_normal(ax.n_dof)
        w = _panel_fn(ax, W)
        for n in [0, 1, 5, 16]:
            pn = np.zeros(n + 1)
            pn[n] = 1.0
            ref = (2 * n + 1) / 2 * _gauss_panel_quad(
                lambda xi: w(xi) * np.polynomial.legendre.legval(xi, pn), -1, 1, 16)
            assert abs((ax.C @ W)[n] - ref) < 1e-12, (panel, n)


def test_fourier_projection_exact():
    for panel in ("linear", "constant"):
        ax = Axis(AxisSpec("fourier", n_panels=16, n_modes=6, panel=panel))
        W = RNG.standard_normal(ax.n_dof)
        w = _panel_fn(ax, W)
        coef = ax.C @ W
        assert abs(coef[0] - 0.5 * _gauss_panel_quad(w, -1, 1, 16)) < 1e-12
        for n in [1, 3, 6]:
            kn = np.pi * n
            ref_a = _gauss_panel_quad(lambda xi: w(xi) * np.cos(kn * (xi + 1)), -1, 1, 16)
            ref_b = _gauss_panel_quad(lambda xi: w(xi) * np.sin(kn * (xi + 1)), -1, 1, 16)
            assert abs(coef[n] - ref_a) < 1e-12
            assert abs(coef[ax.spec.n_modes + n] - ref_b) < 1e-12


def test_constant_panel_dc_matches_ntoada():
    """PC panels on a legendre axis: C[0, i] = 1/N_p (NTO-ADA a0 = mean(W))."""
    ax = Axis(AxisSpec("legendre", n_panels=32, n_modes=8, panel="constant"))
    assert np.allclose(ax.C[0], 1.0 / 32, atol=1e-14)


# ---------------------------------------------------------------------------
# anti-derivative closed forms
# ---------------------------------------------------------------------------

def test_derivative_of_antiderivative_is_identity():
    """d/dx [ int_lo^x f ] = f, exactly, on a non-trivial physical domain."""
    x = np.linspace(0.3, 2.7, 41)
    for kind in ("legendre", "fourier"):
        e1 = Axis(AxisSpec(kind, 12, 8, k=1, domain=(0.3, 2.7))).eval(x, deriv=1)
        e0 = Axis(AxisSpec(kind, 12, 8, k=0, domain=(0.3, 2.7))).eval(x, deriv=0)
        assert np.abs(e1 - e0).max() < 1e-11, kind


def test_antiderivative_vanishes_at_base():
    """Hard IC/BC-at-lo anchor: I^(k) and all lower orders vanish at x = lo."""
    for kind in ("legendre", "fourier"):
        for k in (1, 2):
            ax = Axis(AxisSpec(kind, 12, 8, k=k, domain=(0.0, 5.0)))
            for d in range(k):
                assert np.abs(ax.eval(np.array([0.0]), deriv=d)).max() < 1e-13, (kind, k, d)


def test_antiderivative_matches_quadrature():
    """g(x) = int f (quadrature-limited reference, fine trapz)."""
    dom = (0.0, 4.0)
    for kind in ("legendre", "fourier"):
        ax0 = Axis(AxisSpec(kind, 16, 10, k=0, domain=dom))
        ax1 = Axis(AxisSpec(kind, 16, 10, k=1, domain=dom))
        W = RNG.standard_normal(ax0.n_dof)
        A = ax0.C @ W
        xf = np.linspace(*dom, 16001)
        f = ax0.eval(xf) @ A
        g = ax1.eval(xf) @ A
        g_ref = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(xf))])
        assert np.abs(g - g_ref).max() < 1e-6, kind


# ---------------------------------------------------------------------------
# hard boundary conditions (ADA-L null-space projection, per axis)
# ---------------------------------------------------------------------------

def test_hard_dirichlet_2d_edges_and_commutation():
    axes = [Axis(AxisSpec("legendre", 24, 24)), Axis(AxisSpec("legendre", 24, 24))]
    W = RNG.standard_normal((axes[0].n_dof, axes[1].n_dof))
    conds = {0: [("lo", 0), ("hi", 0)], 1: [("lo", 0), ("hi", 0)]}
    Wc = apply_bc(W, axes, conds)
    Wc_rev = apply_bc(W, axes, {1: conds[1], 0: conds[0]})
    assert np.abs(Wc - Wc_rev).max() < 1e-12
    A = project(Wc, axes)
    xf = np.linspace(-1, 1, 101)
    edges = np.array([-1.0, 1.0])
    u_x_edges = eval_grid(A, axes, [edges, xf])
    u_y_edges = eval_grid(A, axes, [xf, edges])
    assert max(np.abs(u_x_edges).max(), np.abs(u_y_edges).max()) < 1e-12
    # the projection must not be a trivial zeroing of the field
    assert np.abs(eval_grid(A, axes, [xf, xf])).max() > 0.1


def test_hard_neumann_and_inhomogeneous():
    ax = Axis(AxisSpec("legendre", 24, 24, domain=(0.0, 2.0)))
    B, Bp, Pi = ax.bc([("lo", 0), ("hi", 0), ("hi", 1)])   # u(0)=c0, u(2)=c1, u'(2)=c2
    c = np.array([1.5, -0.7, 0.3])
    W = Bp @ c + Pi @ RNG.standard_normal(ax.n_dof)
    assert np.abs(B @ W - c).max() < 1e-12
    A = ax.C @ W
    ends = np.array([0.0, 2.0])
    assert abs(ax.eval(ends)[0] @ A - 1.5) < 1e-12
    assert abs(ax.eval(ends)[1] @ A - (-0.7)) < 1e-12
    assert abs(ax.eval(ends, deriv=1)[1] @ A - 0.3) < 1e-12


def test_fourier_axis_is_periodic():
    ax = Axis(AxisSpec("fourier", 16, 6, domain=(0.0, 3.0)))
    A = ax.C @ RNG.standard_normal(ax.n_dof)
    ends = np.array([0.0, 3.0])
    for d in (0, 1):
        v = ax.eval(ends, deriv=d) @ A
        assert abs(v[0] - v[1]) < 1e-11, d


# ---------------------------------------------------------------------------
# multi-axis synthesis
# ---------------------------------------------------------------------------

def _axes3():
    return [Axis(AxisSpec("legendre", 12, 12)),                       # x: walls
            Axis(AxisSpec("fourier", 12, 5)),                          # y: periodic
            Axis(AxisSpec("legendre", 8, 8, k=1, domain=(0.0, 2.0)))]  # t: ADA


def test_rank1_separability():
    axes = _axes3()
    ws = [RNG.standard_normal(ax.n_dof) for ax in axes]
    W = np.einsum("i,j,k->ijk", *ws)
    pts = [np.linspace(-1, 1, 7), np.linspace(-1, 1, 6), np.linspace(0, 2, 5)]
    u = eval_grid(project(W, axes), axes, pts)
    u1 = [ax.eval(p) @ (ax.C @ w) for ax, p, w in zip(axes, pts, ws)]
    assert np.abs(u - np.einsum("i,j,k->ijk", *u1)).max() < 1e-12


def test_hard_ic_through_synthesis():
    """u(x, y, t=0) = 0 for ANY panel tensor (t axis k=1): the ADA anchor in 3-D."""
    axes = _axes3()
    W = RNG.standard_normal((3, axes[0].n_dof, axes[1].n_dof, axes[2].n_dof))  # batch dim
    u0 = eval_grid(project(W, axes), axes,
                   [np.linspace(-1, 1, 9), np.linspace(-1, 1, 9), np.array([0.0])])
    assert np.abs(u0).max() < 1e-12


def test_eval_points_matches_grid():
    axes = _axes3()
    W = RNG.standard_normal((axes[0].n_dof, axes[1].n_dof, axes[2].n_dof))
    A = project(W, axes)
    pts = np.column_stack([RNG.uniform(-1, 1, 50), RNG.uniform(-1, 1, 50),
                           RNG.uniform(0, 2, 50)])
    up = eval_points(A, axes, pts)
    ug = np.array([eval_grid(A, axes, [p[:1] for p in np.split(pt, 3)])[0, 0, 0]
                   for pt in pts])
    assert np.abs(up - ug).max() < 1e-12


def test_mixed_derivative_sanity():
    """d^2 u / dx dt via closed form vs central FD (FD-limited tolerance)."""
    axes = _axes3()
    W = RNG.standard_normal((axes[0].n_dof, axes[1].n_dof, axes[2].n_dof))
    A = project(W, axes)
    x0, y0, t0, h = 0.21, 0.4, 1.1, 1e-4
    exact = eval_points(A, axes, [[x0, y0, t0]], derivs=(1, 0, 1))[0]
    corners = [eval_points(A, axes, [[x0 + sx * h, y0, t0 + st * h]])[0]
               for sx in (1, -1) for st in (1, -1)]
    fd = (corners[0] - corners[1] - corners[2] + corners[3]) / (4 * h * h)
    assert abs(exact - fd) < 1e-4 * max(1.0, abs(exact))


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"PASS {name}")
    print(f"\n{len(fns)} tests passed")
