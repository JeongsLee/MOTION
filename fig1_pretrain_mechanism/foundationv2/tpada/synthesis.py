"""Multi-axis TP-ADA synthesis: mode products and analytic evaluation.

The panel tensor W has its LAST len(axes) dimensions matching the axes'
DOF counts; any leading dimensions (batch, channel) are carried through.

    A = project(W, axes)                 # W x_1 C^(1) ... x_K C^(K)
    u = eval_grid(A, axes, [xs, ys, ts]) # tensor-product grid, any resolution
    u = eval_points(A, axes, P)          # scattered points (P, K) — mesh-free

Derivatives: pass derivs=(dx, dy, dt); each axis contributes its closed-form
derivative/anti-derivative table, so PDE residuals are exact termwise.
"""
from __future__ import annotations

import numpy as np

from .basis import Axis

__all__ = ["project", "eval_grid", "eval_points", "apply_bc"]


def _axis_dims(W_ndim: int, n_axes: int) -> list[int]:
    return list(range(W_ndim - n_axes, W_ndim))


def project(W: np.ndarray, axes: list[Axis]) -> np.ndarray:
    """Panel tensor -> spectral coefficient tensor, one fixed matrix per axis."""
    A = np.asarray(W, dtype=float)
    for d in _axis_dims(A.ndim, len(axes)):
        ax = axes[d - (A.ndim - len(axes))]
        if A.shape[d] != ax.n_dof:
            raise ValueError(f"axis dim {d}: W has {A.shape[d]}, axis expects {ax.n_dof}")
        A = np.moveaxis(np.tensordot(A, ax.C, axes=([d], [1])), -1, d)
    return A


def eval_grid(A: np.ndarray, axes: list[Axis], points: list[np.ndarray],
              derivs: tuple[int, ...] | None = None) -> np.ndarray:
    """Evaluate u (or a mixed derivative) on the tensor-product grid of `points`."""
    derivs = derivs or (0,) * len(axes)
    U = np.asarray(A, dtype=float)
    for d in _axis_dims(U.ndim, len(axes)):
        j = d - (U.ndim - len(axes))
        E = axes[j].eval(points[j], deriv=derivs[j])       # (n_pts, n_coef)
        U = np.moveaxis(np.tensordot(U, E, axes=([d], [1])), -1, d)
    return U


def eval_points(A: np.ndarray, axes: list[Axis], pts: np.ndarray,
                derivs: tuple[int, ...] | None = None) -> np.ndarray:
    """Evaluate at scattered points pts (P, K) -> (..., P). Mesh-free query."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    K = len(axes)
    if pts.shape[1] != K:
        raise ValueError(f"pts must be (P, {K})")
    derivs = derivs or (0,) * K
    letters = "nmlqr"[:K]
    ops = f"...{letters}," + ",".join(f"p{c}" for c in letters) + "->...p"
    Es = [axes[j].eval(pts[:, j], deriv=derivs[j]) for j in range(K)]
    return np.einsum(ops, A, *Es)


def apply_bc(W: np.ndarray, axes: list[Axis],
             conditions: dict[int, list[tuple[str, int]]]) -> np.ndarray:
    """Enforce homogeneous hard BCs: per-axis null-space projection of W.

    conditions: {axis_index: [(side, deriv), ...]} on legendre axes. The
    projectors act on distinct tensor modes, hence commute; order is free.
    Inhomogeneous data goes through the lift term (transfinite interpolation),
    added separately in synthesis — see DESIGN.md §2.4.
    """
    out = np.asarray(W, dtype=float)
    for j, conds in conditions.items():
        _, _, Pi = axes[j].bc(conds)
        d = out.ndim - len(axes) + j
        out = np.moveaxis(np.tensordot(out, Pi, axes=([d], [1])), -1, d)
    return out
