"""Loader for the NTO-ADA NS vorticity cache (decaying 2-D Navier–Stokes).

data_cache_ns_v20.npz: ic (1024,32,32), traj (1024,33,32,32), nu (1024,) in [5e-4,5e-3].
Scalar vorticity ω(t). NS RHS = −u·∇ω + νΔω with u=∇⊥ψ, Δψ=−ω → the elliptic streamfunction
coupling is exactly what the Elliptic expert must supply.
"""
from __future__ import annotations
import numpy as np

NS_V20 = "/mnt/d/working/2026/neuraladaf/v6/data_cache_ns_v20.npz"


def load(n_train=256, n_test=128, Nt=16, nu_range=None, path=NS_V20):
    """nu_range=(lo,hi) restricts viscosity to a NARROW band (fewer orders of magnitude →
    weaker turbulence, quicker/easier). None = full [5e-4,5e-3] decade."""
    d = np.load(path)
    traj = d["traj"][:, :Nt].astype(np.float32)        # (N, Nt, 32, 32)
    nu = d["nu"].astype(np.float32)
    if nu_range is not None:
        lo, hi = nu_range
        m = (nu >= lo) & (nu <= hi)
        traj, nu = traj[m], nu[m]
        print(f"  [ns] ν∈{nu_range}: {traj.shape[0]} samples (ν actual [{nu.min():.1e},{nu.max():.1e}])",
              flush=True)
    u_tr, nu_tr = traj[:n_train], nu[:n_train]
    u_te, nu_te = traj[n_train:n_train + n_test], nu[n_train:n_train + n_test]
    return {"u_tr": u_tr, "nu_tr": nu_tr, "u_te": u_te, "nu_te": nu_te}


def make_descriptor(nu):
    """[convective, diffusive, elliptic] operator-presence. NS: advection+elliptic always on,
    diffusive ∝ ν."""
    nu = np.asarray(nu, np.float32)
    n = nu.shape[0]
    return np.stack([np.ones(n, np.float32), nu * 200.0, np.ones(n, np.float32)], -1)


def coeffs(n):
    """NS advection is by a spatially-varying velocity (from elliptic), not a global direction
    → zero advection-direction coeffs (convective expert direction unused)."""
    return np.zeros((n, 4), np.float32)
