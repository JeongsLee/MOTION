"""2-D compressible Euler generator (shock-capturing) — the hardest physics gate.

Conserved U = [ρ, ρu, ρv, E], E = p/(γ−1) + ½ρ(u²+v²).
Finite-volume with Rusanov (local Lax–Friedrichs) flux + adaptive CFL substepping, periodic
BC. Random smoothed 2-D Riemann initial conditions (4 quadrants) → interacting shocks /
contacts / rarefactions. 1st-order (diffusive but robust) — enough to TEST whether the shock
expert captures discontinuities; high-order is a later refinement.

Channels (4): [ρ, ρu, ρv, E]. This is the multi-channel / systemic test for the operator basis.
"""
from __future__ import annotations
import numpy as np

GAMMA = 1.4


def _prim_to_cons(rho, u, v, p):
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
    return np.stack([rho, rho * u, rho * v, E], axis=-1)


def _cons_to_prim(U):
    rho = np.maximum(U[..., 0], 1e-6)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    p = np.maximum((GAMMA - 1.0) * (U[..., 3] - 0.5 * rho * (u ** 2 + v ** 2)), 1e-6)
    return rho, u, v, p


def _fluxes(U):
    rho, u, v, p = _cons_to_prim(U)
    E = U[..., 3]
    F = np.stack([rho * u, rho * u * u + p, rho * u * v, (E + p) * u], -1)   # x-flux
    G = np.stack([rho * v, rho * u * v, rho * v * v + p, (E + p) * v], -1)   # y-flux
    c = np.sqrt(GAMMA * p / rho)
    return F, G, np.abs(u) + c, np.abs(v) + c


def _step(U, dt, dx):
    F, G, sx, syy = _fluxes(U)
    # x-faces (between i and i+1 along axis=0): Rusanov
    UR = np.roll(U, -1, axis=0); FR = np.roll(F, -1, axis=0)
    s = np.maximum(sx, np.roll(sx, -1, axis=0))[..., None]
    Fhat = 0.5 * (F + FR) - 0.5 * s * (UR - U)                  # flux at i+1/2
    divx = (Fhat - np.roll(Fhat, 1, axis=0)) / dx
    # y-faces (axis=1)
    UU = np.roll(U, -1, axis=1); GU = np.roll(G, -1, axis=1)
    s2 = np.maximum(syy, np.roll(syy, -1, axis=1))[..., None]
    Ghat = 0.5 * (G + GU) - 0.5 * s2 * (UU - U)
    divy = (Ghat - np.roll(Ghat, 1, axis=1)) / dx
    return U - dt * (divx + divy)


def _smooth(field, k=2):
    """Light periodic box-blur to aid the 1st-order solver at quadrant interfaces."""
    f = field
    for _ in range(k):
        f = 0.25 * (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1))
        f = 0.5 * field + 0.5 * f
    return f


def _riemann_ic(rng, Nx):
    """Random 4-quadrant Riemann IC (smoothed). Returns rho,u,v,p (Nx,Nx)."""
    def q(lo, hi): return rng.uniform(lo, hi)
    rho = np.ones((Nx, Nx)); u = np.zeros((Nx, Nx)); v = np.zeros((Nx, Nx)); p = np.ones((Nx, Nx))
    h = Nx // 2
    for (a, b) in [(slice(0, h), slice(0, h)), (slice(0, h), slice(h, Nx)),
                   (slice(h, Nx), slice(0, h)), (slice(h, Nx), slice(h, Nx))]:
        rho[a, b] = q(0.5, 2.0); p[a, b] = q(0.5, 2.0)
        u[a, b] = q(-0.6, 0.6); v[a, b] = q(-0.6, 0.6)
    return _smooth(rho), _smooth(u), _smooth(v), _smooth(p)


def generate(n_samples=128, Nx=64, Nt=11, T=0.18, cfl=0.3, seed=0):
    """Returns u:(n,Nt,Nx,Nx,4) [ρ,ρu,ρv,E] float32."""
    rng = np.random.default_rng(seed)
    L = 1.0; dx = L / Nx
    t_snaps = np.linspace(0.0, T, Nt)
    out = np.zeros((n_samples, Nt, Nx, Nx, 4), np.float32)
    for s in range(n_samples):
        rho, u, v, p = _riemann_ic(rng, Nx)
        U = _prim_to_cons(rho, u, v, p)
        out[s, 0] = U.astype(np.float32)
        t = 0.0
        for ti in range(1, Nt):
            target = t_snaps[ti]
            it = 0
            while t < target - 1e-9 and it < 5000:
                _, _, sx, sy = _fluxes(U)
                smax = max(float(sx.max()), float(sy.max()), 1e-6)
                dt = min(cfl * dx / smax, target - t)
                U = _step(U, dt, dx)
                t += dt; it += 1
            out[s, ti] = U.astype(np.float32)
    return {"u": out}
