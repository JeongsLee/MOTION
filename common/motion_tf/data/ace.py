"""Loader for the Poseidon ACE (Allen-Cahn) benchmark — reaction-diffusion, our experts'
wheelhouse (diffusive + reaction; convective should stay off).

Dataset (HF camlab-ethz/ACE): variable "solution", dims (sample, time=20, x=128, y=128),
unit square, single scalar channel. Official split 14700/60/240 (train/val/test) over the
full 15000; chunk files solution_0..3.nc hold them in order, so the 240 test trajectories
live at the end of solution_3.nc.

netCDF4 files are HDF5 under the hood, so h5py (present in the tf env) reads them directly.
"""
from __future__ import annotations
import h5py
import numpy as np

ACE_DIR = "/mnt/e/pdefoundation_data/ACE"
N_TOTAL = 15000
N_TEST = 240
CHUNK = 3750  # 15000 / 4 files (verified at load time)


def _read_solution(path, sl=None):
    with h5py.File(path, "r") as f:
        ds = f["solution"]
        return ds[sl] if sl is not None else ds[:]


def load_train_pool(n, chunk_file=f"{ACE_DIR}/solution_0.nc"):
    """First `n` training trajectories from chunk 0. Returns (n,20,128,128) float32."""
    u = _read_solution(chunk_file, np.s_[:n]).astype(np.float32)
    return u


def load_test(n=N_TEST, chunk_file=f"{ACE_DIR}/solution_3.nc"):
    """Official last `n` test trajectories (tail of the final chunk)."""
    with h5py.File(chunk_file, "r") as f:
        total = f["solution"].shape[0]
        u = f["solution"][total - n:].astype(np.float32)
    return u


def descriptor(n, conv=0.0, diff=1.0, react=1.0):
    """Fixed reaction-diffusion descriptor for ACE (single operator, no per-traj coeffs).
    [convective, diffusive, reaction] presence — convective off."""
    return np.tile(np.array([conv, diff, react], np.float32), (n, 1))


def coeffs(n):
    """ACE has no advection → zero advection coeffs (convective expert direction unused)."""
    return np.zeros((n, 4), np.float32)
