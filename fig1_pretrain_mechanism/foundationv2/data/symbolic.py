"""Symbolic-equation + parameter conditioning tokens (foundationv2).

Turns a FamilySpec's governing equation into a fixed-vocabulary token id sequence
and a parameter feature vector, for the optional equation encoder (DESIGN.md §4c).
Absent conditioning -> the NULL token only (learned null); this module just builds
the ids/features, the model owns the embedding + dropout.

Two channels:
  * operator tokens: a small closed vocabulary of mechanism tokens (advection,
    diffusion, ...) present in the equation — order-independent bag, emitted as a
    multi-hot plus an id list for a set-transformer encoder.
  * parameters: named scalars (nu, mach, reynolds, ...) embedded as signed
    log-Fourier features so magnitudes spanning many decades stay well-scaled;
    a per-parameter learned name embedding is added model-side via the name id.
"""
from __future__ import annotations

import numpy as np

# closed operator vocabulary (id 0 = NULL / absent-conditioning token)
OPERATOR_VOCAB = [
    "<null>", "advection", "diffusion", "reaction", "divergence", "pressure_grad",
    "pressure_projection", "buoyancy", "shear", "wave", "laplacian", "elliptic",
    "turbulence",
    "phase_change",     # Fig2c: NEW mechanism token (PoolBoil arm B) — appended LAST, ids stable
]
OP2ID = {o: i for i, o in enumerate(OPERATOR_VOCAB)}

# closed parameter-name vocabulary (id 0 = padding)
PARAM_VOCAB = [
    "<pad>", "nu", "mach", "eta", "zeta", "reynolds", "aoa", "buoyancy",
    "Du", "Dv", "k", "eps", "inlet_velocity",
]
PARAM2ID = {p: i for i, p in enumerate(PARAM_VOCAB)}

N_FREQ = 8              # log-Fourier bands per parameter
MAX_OPS = 8
MAX_PARAMS = 6


def op_ids(operators):
    """Operator token id list (padded/truncated to MAX_OPS) + multi-hot vector."""
    ids = [OP2ID.get(o, 0) for o in operators][:MAX_OPS]
    multihot = np.zeros(len(OPERATOR_VOCAB), np.float32)
    for i in ids:
        multihot[i] = 1.0
    ids = ids + [0] * (MAX_OPS - len(ids))
    return np.array(ids, np.int32), multihot


def _logfourier(value):
    """Signed log-magnitude Fourier features for a scalar spanning many decades."""
    v = float(value)
    s = np.sign(v)
    lm = np.log10(abs(v) + 1e-12)                      # decades
    bands = lm / (2.0 ** np.arange(N_FREQ))            # multi-scale
    return np.concatenate([[s, lm / 10.0],
                           np.sin(bands), np.cos(bands)]).astype(np.float32)


PARAM_FEAT_DIM = 2 + 2 * N_FREQ


def param_features(names, values):
    """(name_ids (MAX_PARAMS,), feats (MAX_PARAMS, PARAM_FEAT_DIM), mask (MAX_PARAMS,))."""
    name_ids = np.zeros(MAX_PARAMS, np.int32)
    feats = np.zeros((MAX_PARAMS, PARAM_FEAT_DIM), np.float32)
    mask = np.zeros(MAX_PARAMS, np.float32)
    for j, (nm, val) in enumerate(zip(names, values)):
        if j >= MAX_PARAMS or val is None:
            continue
        name_ids[j] = PARAM2ID.get(nm, 0)
        feats[j] = _logfourier(val)
        mask[j] = 1.0
    return name_ids, feats, mask


def encode(spec, param_values=None):
    """FamilySpec (+ optional resolved param values dict) -> conditioning arrays.
    param_values overrides const/registry values (e.g. per-sample nu)."""
    oids, omh = op_ids(spec.operators)
    names = list(spec.param_names)
    if param_values is not None:
        vals = [param_values.get(n) for n in names]
    elif spec.param_source == "const":
        vals = list(spec.param_const) + [None] * (len(names) - len(spec.param_const))
    else:
        vals = [None] * len(names)          # per-sample/filename resolved by adapter
    pids, pfeat, pmask = param_features(names, vals)
    return {"op_ids": oids, "op_multihot": omh,
            "param_ids": pids, "param_feats": pfeat, "param_mask": pmask,
            "symbolic": spec.symbolic}
