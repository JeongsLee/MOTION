"""Unified 2-D periodic spectral generator for a family-spanning set of PDEs.

All families share the form   ∂_t u = −a·∇u + ν Δu + r·u(1−u)   on [0,2π)², differing only
in WHICH operators are active — exactly the operator-composition span the family gate must
recover (DESIGN.md §7). Linear advection + diffusion are advanced exactly in Fourier
(split-step, unconditionally stable); the logistic reaction is added explicitly.

    family      | advection | diffusion | reaction  | tests experts
    ------------|-----------|-----------|-----------|------------------------
    heat        |     –     |    ν      |    –      | diffusive
    advection   |    a      |    –      |    –      | convective
    reacdiff    |     –     |    ν      |  r·u(1−u) | diffusive + reaction
    adr         |    a      |    ν      |  r·u(1−u) | convective+diffusive+reaction

Each sample stores coeffs = [a_x, a_y, ν, r]; the gate descriptor (make_descriptor) is the
operator-presence vector [|a|, ν, r] aligned to the (convective, diffusive, reaction) experts.
Burgers (nonlinear self-advection) is a planned 5th family (needs the convective expert to use
the local state as advecting velocity) — added in the next iteration.
"""
from __future__ import annotations
import numpy as np

FAMILIES = ("heat", "advection", "reacdiff", "adr")


def _smooth_ic(rng, Nx, n_modes=4):
    xs = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    f = np.zeros((Nx, Nx))
    for _ in range(n_modes):
        kx, ky = rng.integers(1, 4), rng.integers(1, 4)
        f += rng.uniform(0.5, 1.0) * np.sin(kx * gx + ky * gy + rng.uniform(0, 2 * np.pi))
    f = (f - f.mean()) / (f.std() + 1e-8)
    return 0.5 + 0.25 * f


def _sample_coeffs(rng, family):
    """Return [a_x, a_y, ν, r] for a sample of `family`."""
    ax = ay = nu = r = 0.0
    if family in ("advection", "adr"):
        ang = rng.uniform(0, 2 * np.pi)
        mag = rng.uniform(0.5, 1.5)
        ax, ay = mag * np.cos(ang), mag * np.sin(ang)
    if family in ("heat", "reacdiff"):
        nu = rng.uniform(0.02, 0.10)
    if family == "adr":
        nu = rng.uniform(0.01, 0.05)
    if family in ("reacdiff", "adr"):
        r = rng.uniform(0.5, 2.0)
    return np.array([ax, ay, nu, r], dtype=np.float64)


def generate(family, n_samples, Nx=32, Nt=16, T=1.0, n_sub=20, seed=0):
    rng = np.random.default_rng(seed)
    L = 2 * np.pi
    k1 = np.fft.fftfreq(Nx, d=L / Nx) * 2 * np.pi
    kx, ky = k1[:, None], k1[None, :]
    k2 = kx ** 2 + ky ** 2
    dt = (T / (Nt - 1)) / n_sub

    U = np.zeros((n_samples, Nt, Nx, Nx), np.float32)
    C = np.zeros((n_samples, 4), np.float32)
    for s in range(n_samples):
        ax, ay, nu, r = _sample_coeffs(rng, family)
        C[s] = [ax, ay, nu, r]
        prop = np.exp((-1j * (ax * kx + ay * ky) - nu * k2) * dt)
        u = _smooth_ic(rng, Nx)
        U[s, 0] = u.astype(np.float32)
        for t in range(1, Nt):
            for _ in range(n_sub):
                u = np.real(np.fft.ifft2(np.fft.fft2(u) * prop))
                if r != 0.0:
                    u = u + dt * r * u * (1.0 - u)
            U[s, t] = u.astype(np.float32)
    return {"u": U, "coeffs": C, "family": family}


def generate_multi(families, n_per, Nx=32, Nt=16, T=1.0, n_sub=20, seed=0):
    """Concatenate several families. Returns u, coeffs, fam_idx (int per sample)."""
    us, cs, fids = [], [], []
    for fi, fam in enumerate(families):
        d = generate(fam, n_per, Nx=Nx, Nt=Nt, T=T, n_sub=n_sub, seed=seed + 100 * fi)
        us.append(d["u"]); cs.append(d["coeffs"])
        fids.append(np.full(n_per, fi, dtype=np.int32))
    return {"u": np.concatenate(us), "coeffs": np.concatenate(cs),
            "fam_idx": np.concatenate(fids), "families": list(families)}


def make_descriptor(coeffs):
    """Operator-presence descriptor [|a|, ν, r], aligned to (convective, diffusive, reaction)."""
    c = np.asarray(coeffs, np.float32)
    conv = np.maximum(np.abs(c[:, 0]), np.abs(c[:, 1]))
    return np.stack([conv, c[:, 2] * 50.0, c[:, 3] * 0.5], axis=-1).astype(np.float32)
