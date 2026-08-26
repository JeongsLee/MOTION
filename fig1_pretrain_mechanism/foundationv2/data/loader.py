"""Collocation dataloader: unified corpus -> training batches for the P2 core.

Bridges preprocessing to UniversalTPADA. Per family it reads the split manifest and
serves units either lazily via the adapter (grid/3D — cheap arrays, no materialized
copy) or from the materialized unified npz (mesh — parse-once). Every unit becomes a
training example with a shared contract:

  input   : observed frames  (T_in) as the encoder input
            grid  -> (T_in, *dims, S) ;  point -> coords (N,K) + feats (N, T_in*S)
  target  : collocation set — Q sampled (coords, t) with ground-truth field values,
            for the mesh-free loss. grid: sample cells+frames; point: sample nodes+frames.
  cond    : symbolic/param arrays (from the sample) ; roles ; cmask ; tmask

Steady families (T==1) form a pseudo-time target: input = the single frame, target =
the same frame at t=1 (relaxation endpoint); the model's IC anchor handles t=0.

This module is framework-light: it yields numpy dicts; a thin TF/torch wrapper batches
them. Determinism: unit order from the manifest; collocation RNG seeded per (unit, epoch).
"""
from __future__ import annotations

import glob
import os

import numpy as np

from . import adapters
from .registry import FAMILIES, NUM_SLOTS
from .splits import load_manifest

UNIFIED = os.path.join(os.environ.get("CORPUS_ROOT", "/corpus"), "cache", "unified")
# Families re-materialized with the GINO dense shape context (sdf_vol) + PCA-fallback surface
# normals live in a SEPARATE cache dir, so the original one stays intact as a rollback. Checked
# first, per family: a family absent from it simply falls through to the base cache.
UNIFIED_SDF = os.path.join(os.environ.get("CORPUS_ROOT", "/corpus"), "cache", "unified_sdf")


def _materialized(family, split):
    pref = sorted(glob.glob(os.path.join(UNIFIED_SDF, family, f"{split}_*.npz")))
    return pref or sorted(glob.glob(os.path.join(UNIFIED, family, f"{split}_*.npz")))


def _iter_materialized(family, split):
    for shard in _materialized(family, split):
        for s in np.load(shard, allow_pickle=True)["samples"]:
            yield dict(s.item()) if hasattr(s, "item") else dict(s)


def _iter_lazy(spec, split):
    man = load_manifest(spec.name)
    wanted = set(man["splits"][split])
    for uid, ref in adapters.units(spec):
        if uid in wanted:
            yield adapters.load(spec, ref)


# OUTPUT-SLOT SEPARATION FOR THE 3D SURFACE FAMILIES.
# All five K=3 families wrote to slot 4, and the quantity means two incompatible things there:
# for pdebench3d CNS it is the thermodynamic pressure of a periodic compressible turbulence
# volume, for shapenet_car / drivaernet_pressure it is the static pressure on a car surface.
# Those five are the ONLY families that exercise the 3D axial weights, so the 3D path was being
# asked to emit both through one channel — and splitting the output is what broke the same wall
# on the 2D side (per-family head + mask->geom took drivaernet 38.5 -> 24.3).
# EMPTY ON PURPOSE. The obvious target was slot 8, whose only other user is shallow_water — but
# shallow_water is the ONE family on a private slot (n=1) and also the easiest in the corpus
# (single smooth channel), so "private slot" and "easy problem" are perfectly confounded there,
# and across all 26 families contention barely predicts anything: Pearson +0.07 against headroom,
# Spearman +0.13, with Wave-Layer the second-LEAST contended slot and the second-worst headroom.
# Moving a 3D surface family onto slot 8 would spend our one clean channel on a hunch. The hook
# stays because the remap is the right shape of fix if the gradient-conflict measurement
# (diag_gradconflict.py) shows the K=3 slot-4 collision is real; the destination should then be a
# NEW slot (S=9 -> 10), not one taken from a family that is already solved.
SLOT_REMAP = {
    "airfrans": {4: 9}, "shapenet_car": {4: 9}, "drivaernet_pressure": {4: 9},
    "Wave-Layer": {6: 10},
    "geofno_airfoil": {4: 11}, "geofno_elasticity": {4: 11},
}


def _remap_slots(d, family):
    """Widen to NUM_SLOTS and move this family's channels to their semantic slots.

    Widening is needed because the materialized caches were written at the old width; the lazy
    adapters already emit NUM_SLOTS. Doing both here keeps one code path and rebuilds no cache.
    """
    mv = SLOT_REMAP.get(family)
    for key in ("fields", "cmask"):
        v = d.get(key)
        if v is None:
            continue
        v = np.asarray(v)
        if v.shape[-1] < NUM_SLOTS:                       # pad old 9-wide cache entries
            v = np.concatenate(
                [v, np.zeros(v.shape[:-1] + (NUM_SLOTS - v.shape[-1],), v.dtype)], -1)
        elif mv:
            v = np.array(v, copy=True)
        if mv:
            for a, b in mv.items():
                v[..., b], v[..., a] = v[..., a].copy(), np.zeros_like(v[..., a])
        d[key] = v
    return d


def units_for(family, split):
    """Yield UnifiedSample-like dicts for (family, split): materialized if present, else lazy."""
    spec = FAMILIES[family]
    if _materialized(family, split):
        for d in _iter_materialized(family, split):
            yield _remap_slots(d, family)
    else:
        for s in _iter_lazy(spec, split):
            yield _remap_slots(_as_dict(s), family)


def _as_dict(s):
    d = {"family": s.family, "K": s.K, "mode": s.mode, "fields": s.fields,
         "cmask": s.cmask, "tmask": s.tmask, "roles": np.array(s.roles),
         "op_ids": s.cond["op_ids"], "op_multihot": s.cond["op_multihot"],
         "param_ids": s.cond["param_ids"], "param_feats": s.cond["param_feats"],
         "param_mask": s.cond["param_mask"]}
    if s.mode == "grid":
        d["dims"] = np.array(s.dims)
    else:
        d["coords"] = s.coords
    if s.geom is not None:
        d["geom"] = s.geom
    if getattr(s, "sdf_vol", None) is not None:
        d["sdf_vol"] = s.sdf_vol
    return d


def _cell_coords(dims):
    axes = [(np.arange(n) + 0.5) / n for n in dims]
    return np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], -1).astype(np.float32)


def make_example(sample, t_in, n_colloc, rng):
    """One training example dict from a unified sample.
    Returns: input frames, collocation (coords_q (Q,K), t_q (Q,), y_q (Q,S)), cond, meta.
    Grid input kept dense (encoder patchifies); point input as (coords, feats)."""
    K, mode = int(sample["K"]), sample["mode"]
    fields = sample["fields"]                                     # grid (T,*dims,S) | point (T,N,S)
    S = fields.shape[-1]
    if mode == "grid":
        dims = tuple(int(x) for x in sample["dims"])
        T = fields.shape[0]
        node = _cell_coords(dims)                                 # (Ncell,K)
        flat = fields.reshape(T, -1, S)                           # (T,Ncell,S)
    else:
        node = sample["coords"]
        flat = fields                                             # (T,N,S)
        T = flat.shape[0]

    N = node.shape[0]
    S = flat.shape[-1]
    steady = (T == 1)

    # per-channel STD-only normalization: balance the wildly different family/channel scales
    # (pressure O(60) vs velocity O(1)) WITHOUT removing the mean — the predictable DC component
    # is kept (unsteady: carried by the hard-IC anchor; steady: the field head must reproduce it).
    # Subtracting the mean (full instance-norm) removed the DC freebie and hurt smooth families.
    cm = sample["cmask"].astype(bool)
    ref = flat[:1] if steady else flat[:min(t_in, T - 1)]        # steady: field; else obs window
    sd = ref.std(axis=(0, 1), keepdims=True)                    # (1,1,S)
    scale = np.where((sd > 1e-6) & cm[None, None], sd, 1.0)
    flat = flat / scale                                          # std-only; DC preserved

    # GEOMETRY LEVERS (MARIO/AirfRANS): for SDF families replace the raw signed-distance geom with
    # [scale-normalized SDF, boundary-layer mask] and bias collocation toward the near-wall region.
    # Attacks our airfrans pathology directly (pressure std ~198 concentrated at the wall + ~2%
    # boundary-layer undersampling); MARIO reports 6x worse drag with the mask removed. Architecture
    # is UNCHANGED (geom slots 0/1 already exist), so run13 weights warm-start exactly (slot1 mask
    # column was trained on zeros -> benign perturbation; near-wall sampling changes no weights).
    # GEOM canonical 8-ch layout (adapters fill what they have): [0=SDF,1=BL mask,2=nx,3=ny,4=nz,
    # 5=mean curv,6=gauss curv,7=interior/material]. For "sdf" families we scale-normalize the raw
    # SDF in ch0, build the near-wall bump mask in ch1, and bias collocation to the near-wall region
    # (MARIO/AirfRANS lever). "surface" families (drivaernet/shapenet: normals+curvature, no volume
    # SDF) pass through unchanged. Encoder geom_gate is zero-init so warm-start is benign.
    GEOM_W = 8
    wall_w = None
    geom_kind = FAMILIES[sample["family"]].geom_kind if sample["family"] in FAMILIES else "none"
    if geom_kind in ("sdf", "surface") and sample.get("geom") is not None:
        g = np.asarray(sample["geom"], np.float32).reshape(N, -1)
        if g.shape[1] < GEOM_W:                                   # pad to the canonical width
            g = np.concatenate([g, np.zeros((N, GEOM_W - g.shape[1]), np.float32)], -1)
        g = g[:, :GEOM_W]
        if geom_kind == "sdf":
            sdf = g[:, 0].copy()
            L = float(np.percentile(np.abs(sdf), 90)) + 1e-6      # robust characteristic length
            dist = np.abs(sdf) / L                                # 0 at wall, ~1 in the far field
            tau = 0.1                                             # boundary-layer thickness (norm. dist)
            g[:, 0] = np.clip(sdf / L, -4.0, 4.0)                 # scale-normalized signed distance
            g[:, 1] = np.clip(1.0 - dist / tau, 0.0, 1.0) ** 2    # MARIO bump: 1 at wall -> 0 at tau
            wall_w = np.exp(-dist / tau).astype(np.float64)       # near-wall collocation weight
        sample = dict(sample)
        sample["geom"] = g.astype(np.float32)

    if wall_w is not None:
        # half near-wall (resolve the boundary layer where pressure gradients live), half uniform
        # (far-field coverage) — a robust mixture, no delicate tuning.
        p = wall_w / wall_w.sum()
        n_wall = n_colloc // 2
        qi_node = np.concatenate([rng.choice(N, size=n_wall, p=p),
                                  rng.integers(0, N, size=n_colloc - n_wall)])
    else:
        qi_node = rng.integers(0, N, size=n_colloc)
    coords_q = node[qi_node]                                      # (Q,K)
    # PER-QUERY geometry (native resolution, NOT box-averaged) for the decoder geometry-FiLM: the
    # geometry sampled at the SAME nodes as the collocation targets. Zeros where a family has no geom.
    geom_q = np.zeros((n_colloc, GEOM_W), np.float32)
    if sample.get("geom") is not None:
        gfull = np.asarray(sample["geom"], np.float32).reshape(N, -1)
        gfull = gfull[:, :GEOM_W] if gfull.shape[1] >= GEOM_W else np.concatenate(
            [gfull, np.zeros((N, GEOM_W - gfull.shape[1]), np.float32)], -1)
        geom_q = gfull[qi_node]

    if steady:
        # input = geometry/params only (the stored field IS the answer -> no leakage);
        # target = the field at the relaxation endpoint t=1, hard-IC anchor = 0.
        x_frames = np.zeros((1, N, S), np.float32)
        t_q = np.ones(n_colloc, np.float32)
        ic_q = np.zeros((n_colloc, S), np.float32)
        y_q = flat[0, qi_node]
    else:
        ti = min(t_in, T - 1)                                     # observed frames 0..ti-1
        anchor = ti - 1                                           # last observed = synthesis t=0
        x_frames = flat[:ti]                                      # (ti,Nnode,S)
        fut = np.arange(anchor + 1, T)                            # pure future frames
        win = float(T - 1 - anchor)
        qi_t = rng.integers(0, len(fut), size=n_colloc)
        t_q = ((fut[qi_t] - anchor) / win).astype(np.float32)     # forecast-window time in (0,1]
        ic_q = flat[anchor, qi_node]                              # hard-IC anchor value
        y_q = flat[fut[qi_t], qi_node]

    ex = {
        "K": K, "mode": mode, "family": sample["family"], "steady": steady,
        "coords_node": node.astype(np.float32),                   # encoder point positions
        "x_frames": x_frames.astype(np.float32),                  # (ti,Nnode,S)
        "dims": np.array(sample.get("dims", [])) if mode == "grid" else None,
        "coords_q": coords_q.astype(np.float32), "t_q": t_q,
        "geom_q": geom_q.astype(np.float32),                      # (Q,GEOM_W) per-query geometry
        "ic_q": ic_q.astype(np.float32), "y_q": y_q.astype(np.float32),
        "cmask": sample["cmask"], "roles": sample["roles"],
        "scale": scale.reshape(-1).astype(np.float32),            # per-channel std (denormalize -> physical)
        "op_ids": sample["op_ids"], "op_multihot": sample["op_multihot"],
        "param_ids": sample["param_ids"], "param_feats": sample["param_feats"],
        "param_mask": sample["param_mask"],
    }
    if "geom" in sample and sample["geom"] is not None:
        ex["geom"] = sample["geom"]
    return ex


def stream(families, split, t_in=10, n_colloc=2048, seed=0, max_per_fam=None):
    """Deterministic multi-family example stream (numpy dicts). A TF/torch collate wraps this."""
    rng = np.random.default_rng(seed)
    for fam in families:
        n = 0
        for sample in units_for(fam, split):
            yield make_example(sample, t_in, n_colloc, rng)
            n += 1
            if max_per_fam and n >= max_per_fam:
                break
