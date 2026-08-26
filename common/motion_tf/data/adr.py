"""2-D Advection–Diffusion–Reaction generator (Track A, controlled operators).

    ∂_t u = −(a_x ∂_x u + a_y ∂_y u)  +  ν (∂_xx + ∂_yy) u  +  r · u(1−u)
            └──── advection ────┘      └──── diffusion ────┘   └ reaction ┘

Periodic domain [0, 2π)². Split-step spectral integrator: the linear (advection+diffusion)
part is advanced *exactly* in Fourier each sub-step (unconditionally stable); the reaction
term is added explicitly. Each sample carries its own (a_x, a_y, ν, r) — these are exactly
the operator-composition coefficients the family gate must recover (DESIGN.md §7), so ADR
is the ideal first interpretability testbed: turning a coefficient off should turn the
corresponding gate weight off.
"""
from __future__ import annotations
import numpy as np


def _random_smooth_field(rng, Nx, n_modes=4):
    """Sum of a few low Fourier modes, normalized to unit std, shifted to [0,1]-ish."""
    field = np.zeros((Nx, Nx), dtype=np.float64)
    xs = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    for _ in range(n_modes):
        kx, ky = rng.integers(1, 4), rng.integers(1, 4)
        ph = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.0)
        field += amp * np.sin(kx * gx + ky * gy + ph)
    field = (field - field.mean()) / (field.std() + 1e-8)
    return 0.5 + 0.25 * field                                  # keep in a mild range


def generate(n_samples=64, Nx=32, Nt=16, T=1.0, n_sub=20, seed=0,
             a_range=(-1.0, 1.0), nu_range=(2e-3, 2e-2), r_range=(0.0, 2.0),
             active_prob=0.7):
    """Returns dict with u:(n,Nt,Nx,Nx) float32 and coeffs:(n,4)=[a_x,a_y,ν,r] float32.

    `active_prob`: each operator is independently switched fully on/off per sample so the
    dataset spans advection-only, diffusion-only, reaction-only and mixed regimes — this
    is what lets the gate-vs-coefficient interpretability check have signal.
    """
    rng = np.random.default_rng(seed)
    L = 2 * np.pi
    k1 = np.fft.fftfreq(Nx, d=L / Nx) * 2 * np.pi
    kx = k1[:, None]; ky = k1[None, :]
    k2 = kx ** 2 + ky ** 2
    dt = (T / (Nt - 1)) / n_sub

    U = np.zeros((n_samples, Nt, Nx, Nx), dtype=np.float32)
    C = np.zeros((n_samples, 4), dtype=np.float32)

    for s in range(n_samples):
        a_x = rng.uniform(*a_range) * (rng.random() < active_prob)
        a_y = rng.uniform(*a_range) * (rng.random() < active_prob)
        nu = rng.uniform(*nu_range) * (rng.random() < active_prob)
        r = rng.uniform(*r_range) * (rng.random() < active_prob)
        C[s] = [a_x, a_y, nu, r]

        symbol = -1j * (a_x * kx + a_y * ky) - nu * k2          # linear Fourier symbol
        prop = np.exp(symbol * dt)                              # exact linear propagator

        u = _random_smooth_field(rng, Nx)
        U[s, 0] = u.astype(np.float32)
        for t in range(1, Nt):
            for _ in range(n_sub):
                u_hat = np.fft.fft2(u) * prop                   # advection + diffusion
                u = np.real(np.fft.ifft2(u_hat))
                u = u + dt * r * u * (1.0 - u)                  # reaction (explicit)
            U[s, t] = u.astype(np.float32)

    return {"u": U, "coeffs": C}


def make_descriptor(coeffs):
    """PDE descriptor for the family gate (DESIGN.md §4.3, option a).

    v1: normalised coefficient magnitudes [|a_x|, |a_y|, ν, r] — directly encodes which
    operators are active. (Symbolic-token descriptor is the §6.1(b) upgrade.)"""
    c = np.asarray(coeffs, dtype=np.float32)
    desc = np.stack([np.abs(c[:, 0]), np.abs(c[:, 1]),
                     c[:, 2] * 50.0, c[:, 3] * 0.5], axis=-1)
    return desc.astype(np.float32)
