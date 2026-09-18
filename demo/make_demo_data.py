"""Generate the small simulated dataset used by `demo/run_demo.py`.

A passive scalar is transported by a frozen divergence-free velocity field on a
periodic 32x32 box:

    dc/dt + (u . grad) c = nu * lap(c),    div(u) = 0

which is the advection + diffusion pair of the incompressible Navier-Stokes
family the demo registers the data as. Integration is pseudo-spectral with an
exact integrating factor for the diffusive part and RK2 for the advective part.

The generated file is small (~2 MB) and is committed to the repository, so the
demo runs without it; regenerate with

    python demo/make_demo_data.py

Output: demo/data/demo_advdiff.npz
    fields (n_traj, T, 32, 32, 3)  channels = (u, v, c)
    nu     (n_traj,)               diffusivity per trajectory
"""
from __future__ import annotations

import os

import numpy as np

N = 32                      # grid points per axis
T = 12                      # stored frames per trajectory
NT = 16                     # trajectories
SUB = 20                    # solver steps between stored frames
DT = 2.0e-3
SEED = 20260918


def _wavenumbers(n):
    k = np.fft.fftfreq(n, d=1.0 / n) * 2.0 * np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx, ky


def _stream_velocity(rng, kx, ky, n_modes=4):
    """Frozen solenoidal velocity from a random band-limited stream function."""
    psi_h = np.zeros(kx.shape, complex)
    for _ in range(n_modes):
        a, b = rng.integers(1, 4), rng.integers(1, 4)
        for sx, sy in ((a, b), (-a, b)):
            amp = rng.normal() + 1j * rng.normal()
            psi_h[sx % N, sy % N] += amp
    psi_h[0, 0] = 0.0
    u = np.real(np.fft.ifft2(1j * ky * psi_h))
    v = np.real(np.fft.ifft2(-1j * kx * psi_h))
    s = float(np.max(np.hypot(u, v))) + 1e-12
    return (0.8 * u / s), (0.8 * v / s)


def _initial_scalar(rng):
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = np.zeros((N, N))
    for _ in range(3):
        x0, y0 = rng.uniform(size=2)
        w = rng.uniform(0.06, 0.13)
        dx = np.minimum(np.abs(X - x0), 1.0 - np.abs(X - x0))
        dy = np.minimum(np.abs(Y - y0), 1.0 - np.abs(Y - y0))
        c += rng.choice([-1.0, 1.0]) * np.exp(-(dx ** 2 + dy ** 2) / (2 * w ** 2))
    return c / (np.abs(c).max() + 1e-12)


def _advect(c_h, u, v, kx, ky):
    """-(u.grad)c evaluated in physical space, returned in spectral space."""
    cx = np.real(np.fft.ifft2(1j * kx * c_h))
    cy = np.real(np.fft.ifft2(1j * ky * c_h))
    return np.fft.fft2(-(u * cx + v * cy))


def simulate(rng):
    kx, ky = _wavenumbers(N)
    k2 = kx ** 2 + ky ** 2
    u, v = _stream_velocity(rng, kx, ky)
    nu = float(rng.uniform(2e-4, 1e-3))
    c_h = np.fft.fft2(_initial_scalar(rng))
    frames = []
    for t in range(T * SUB):
        if t % SUB == 0:
            frames.append(np.real(np.fft.ifft2(c_h)))
        # integrating factor for diffusion, RK2 (midpoint) for advection
        e_half, e_full = np.exp(-nu * k2 * DT / 2), np.exp(-nu * k2 * DT)
        a1 = _advect(c_h, u, v, kx, ky)
        c_mid = e_half * (c_h + 0.5 * DT * a1)
        a2 = _advect(c_mid, u, v, kx, ky)
        c_h = e_full * c_h + DT * e_half * a2
    c = np.stack(frames)                                    # (T,N,N)
    uu = np.broadcast_to(u, c.shape)
    vv = np.broadcast_to(v, c.shape)
    return np.stack([uu, vv, c], -1).astype(np.float32), nu


def main():
    rng = np.random.default_rng(SEED)
    fields, nus = [], []
    for i in range(NT):
        f, nu = simulate(rng)
        fields.append(f)
        nus.append(nu)
        print(f"  [{i + 1:2d}/{NT}] nu={nu:.2e}  |c| range "
              f"[{f[..., 2].min():+.3f}, {f[..., 2].max():+.3f}]", flush=True)
    fields = np.stack(fields)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "demo_advdiff.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, fields=fields, nu=np.asarray(nus, np.float32),
                        dt=np.float32(DT * SUB), n=np.int32(N))
    print(f"[demo-data] wrote {out}  fields{fields.shape}  "
          f"{os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
