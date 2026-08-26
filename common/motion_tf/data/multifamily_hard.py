"""Unified hard-physics multi-family dataset (common channel layout, Poseidon-style).

Families span the full operator basis at a COMMON grid (64², Nt=11), padded to C_max=4 channels
with a per-family channel mask:

    family   | channels (slots)         | mask        | dominant operator → expert
    ---------|--------------------------|-------------|----------------------------
    heat     | [u,0,0,0]                | [1,0,0,0]   | diffusive
    adr      | [u,0,0,0]                | [1,0,0,0]   | convective+diffusive+reaction
    ns       | [ω,0,0,0]                | [1,0,0,0]   | elliptic (streamfunction)
    euler    | [ρ,ρu,ρv,E]              | [1,1,1,1]   | shock

Descriptor = family one-hot (gate LEARNS family→experts; the fingerprint is then a learned
result, not handed in). Loss/eval use the channel mask so inactive slots don't count.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf

from . import families as fam
from . import euler as eul
from . import ns as ns_mod

FAMILIES = ("heat", "adr", "ns", "euler")
C_MAX = 4


def _pad_scalar(u):                       # (n,Nt,Nx,Nx) -> (n,Nt,Nx,Nx,4)
    n, Nt, Nx, _ = u.shape
    out = np.zeros((n, Nt, Nx, Nx, C_MAX), np.float32)
    out[..., 0] = u
    return out


def _resize(u, Nx):                       # (n,Nt,h,w) -> (n,Nt,Nx,Nx)
    n, Nt, h, w = u.shape
    if h == Nx:
        return u
    x = tf.image.resize(u.reshape(n * Nt, h, w, 1), [Nx, Nx]).numpy()
    return x.reshape(n, Nt, Nx, Nx)


# Per-family physical domain. NOTE: using TRUE per-family L (2π for heat/adr) REGRESSED the gate
# fingerprint — it broke operator-magnitude comparability (diffusive Laplacian ∝1/dx² shrank 40×
# for heat → shock's numerical diffusion absorbed heat's smoothing → heat mis-routed to shock).
# COMMON dx keeps expert output magnitudes comparable → clean routing (heat→diffusive etc.).
# So we use a UNIFORM L=1 here (the known-good interpretable setting). The per-sample-metric
# plumbing is kept as a capability; proper per-family physics + clean routing would instead need
# per-EXPERT output normalization, not per-family dx.
_DOMAIN_L = {"heat": 1.0, "adr": 1.0, "ns": 1.0, "euler": 1.0}


def _build_family(name, n, Nx, Nt, seed):
    """Returns u(n,Nt,Nx,Nx,4), mask(4,), coeffs(n,4), dx(scalar)."""
    if name in ("heat", "adr"):
        d = fam.generate(name, n, Nx=Nx, Nt=Nt, seed=seed)
        u = _pad_scalar(d["u"]); mask = np.array([1, 0, 0, 0], np.float32); coef = d["coeffs"]
    elif name == "ns":
        d = ns_mod.load(n_train=n, n_test=0, Nt=Nt, nu_range=(1e-3, 5e-3))
        u = _pad_scalar(_resize(d["u_tr"], Nx)); mask = np.array([1, 0, 0, 0], np.float32)
        coef = np.zeros((u.shape[0], 4), np.float32)
    elif name == "euler":
        d = eul.generate(n, Nx=Nx, Nt=Nt, seed=seed)
        u = d["u"]; mask = np.array([1, 1, 1, 1], np.float32)
        coef = np.zeros((n, 4), np.float32)
    dx = float(_DOMAIN_L[name]) / Nx
    return u, mask, coef, dx


def build(n_per, Nx=64, Nt=11, seed=0, families=FAMILIES):
    us, masks, coefs, fids, mets = [], [], [], [], []
    for fi, name in enumerate(families):
        u, mask, coef, dx = _build_family(name, n_per, Nx, Nt, seed + 13 * fi)
        m = u.shape[0]
        us.append(u); coefs.append(coef.astype(np.float32))
        masks.append(np.tile(mask, (m, 1))); fids.append(np.full(m, fi, np.int32))
        mets.append(np.tile(np.array([dx, dx], np.float32), (m, 1)))   # per-sample (dx,dy)
    u = np.concatenate(us); coef = np.concatenate(coefs)
    mask = np.concatenate(masks); fid = np.concatenate(fids); metric = np.concatenate(mets)
    desc = np.eye(len(families), dtype=np.float32)[fid]          # family one-hot
    return {"u": u, "mask": mask, "coeffs": coef, "desc": desc, "metric": metric,
            "fam_idx": fid, "families": list(families)}
