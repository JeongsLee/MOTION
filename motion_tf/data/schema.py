"""Unified sample schema for the universal corpus (foundationv2).

Every raw source is mapped by an adapter to a UnifiedSample. Two storage modes,
one contract:
  * GRID  : fields (T, *dims, NUM_SLOTS), coords implicit (cell centers).
  * POINT : fields (T, N, NUM_SLOTS), coords (N, K) explicit.
Both carry: geom (SDF/mask/normal channels, per node/cell), a channel-validity
mask over the NUM_SLOTS slots, a temporal-validity mask (T,), roles (K,), and the
conditioning arrays from symbolic.encode. Coordinates are normalized to [0,1]^K.

The training collocation loss samples points from `fields`; grid mode is the
degenerate dense case. Steady families carry T=1 (relaxation-trajectory target is
formed at train time).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import NUM_SLOTS


@dataclass
class UnifiedSample:
    family: str
    K: int
    mode: str                       # "grid" | "point"
    fields: np.ndarray              # grid: (T,*dims,S) ; point: (T,N,S)
    cmask: np.ndarray               # (S,) which slots are real
    tmask: np.ndarray               # (T,) valid frames
    roles: tuple
    cond: dict                      # symbolic.encode output
    coords: np.ndarray = None       # point mode only: (N,K) in [0,1]^K
    geom: np.ndarray = None         # (…,G) SDF/mask/normal per node/cell (optional)
    dims: tuple = None              # grid mode only
    times: np.ndarray = None        # (T,) physical times (None -> uniform [0,1])
    sdf_vol: np.ndarray = None      # (R,)*K signed-distance VOLUME on a uniform [0,1]^K lattice.
    # GINO lever for co-dimension-1 families (surface meshes): a 2-manifold in 3D occupies ~3% of a
    # volumetric latent box, so a point-scatter encoder sees an almost-empty box. Sampling the SDF
    # on a lattice turns the surface into a DENSE volumetric field (this is how GINO conditions on
    # geometry). Stored at a fixed resolution and trilinearly resampled to the model's box dims.

    def validate(self):
        S = NUM_SLOTS
        assert self.fields.shape[-1] == S, (self.fields.shape, S)
        assert self.cmask.shape == (S,)
        T = self.fields.shape[0]
        assert self.tmask.shape == (T,)
        assert len(self.roles) == self.K
        if self.mode == "grid":
            assert self.dims is not None and len(self.dims) == self.K
            assert self.fields.shape == (T, *self.dims, S)
        else:
            assert self.coords is not None and self.coords.shape[1] == self.K
            assert self.fields.shape == (T, self.coords.shape[0], S)
            assert self.coords.min() >= -1e-6 and self.coords.max() <= 1 + 1e-6
        return self


def to_slots(raw, channels):
    """(…, C_raw) at raw channel order -> (…, NUM_SLOTS) at universal slots.
    Returns (padded, cmask)."""
    lead = raw.shape[:-1]
    out = np.zeros(lead + (NUM_SLOTS,), np.float32)
    cmask = np.zeros(NUM_SLOTS, np.float32)
    for raw_i, slot in enumerate(channels):
        out[..., slot] = raw[..., raw_i]
        cmask[slot] = 1.0
    return out, cmask


def grid_sample(family, K, fields_raw, spec, cond, dims, geom=None,
                t_valid=None, times=None):
    """Build a validated GRID UnifiedSample. fields_raw: (T,*dims,C_raw)."""
    fields, cmask = to_slots(fields_raw, spec.channels)
    T = fields.shape[0]
    tmask = np.ones(T, np.float32)
    if t_valid is not None:
        tmask[t_valid:] = 0.0
    return UnifiedSample(family, K, "grid", fields, cmask, tmask, tuple(spec.roles[:K]),
                         cond, dims=tuple(dims), geom=geom, times=times).validate()


def point_sample(family, K, coords, fields_raw, spec, cond, geom=None, times=None,
                 sdf_vol=None):
    """Build a validated POINT UnifiedSample. coords:(N,K) in [0,1]; fields_raw:(T,N,C_raw)."""
    fields, cmask = to_slots(fields_raw, spec.channels)
    T = fields.shape[0]
    return UnifiedSample(family, K, "point", fields, cmask, np.ones(T, np.float32),
                         tuple(spec.roles[:K]), cond,
                         coords=coords.astype(np.float32), geom=geom, times=times,
                         sdf_vol=sdf_vol).validate()


def normalize_coords(coords, lo=None, hi=None):
    """Map physical coordinates (N,K) into [0,1]^K by per-axis min/max (or given bounds)."""
    coords = np.asarray(coords, np.float64)
    lo = coords.min(0) if lo is None else np.asarray(lo)
    hi = coords.max(0) if hi is None else np.asarray(hi)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return ((coords - lo) / span).astype(np.float32), (lo, hi)
