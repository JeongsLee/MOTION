"""Symbolic-equation tokens and parameter features (numpy). Port of motion_tf/data/symbolic.py.

Difference to the reference: the operator vocabulary is a FUNCTION of the legacy flag (MotionConfig.legacy_vocab)
instead of being fixed at import time by MOTION_LEGACY_VOCAB. Units/samples always carry the full 14-token multi-hot
(the corpus is built with it); `fit_multihot` drops the appended 'phase_change' token for legacy checkpoints exactly
as latent_control_k.py's LOCAL_UNITS restore does.
"""
from __future__ import annotations

import numpy as np

OPERATOR_VOCAB_FULL = [
    "<null>", "advection", "diffusion", "reaction", "divergence", "pressure_grad",
    "pressure_projection", "buoyancy", "shear", "wave", "laplacian", "elliptic",
    "turbulence",
    "phase_change",
]

PARAM_VOCAB = [
    "<pad>", "nu", "mach", "eta", "zeta", "reynolds", "aoa", "buoyancy",
    "Du", "Dv", "k", "eps", "inlet_velocity",
]
PARAM2ID = {p: i for i, p in enumerate(PARAM_VOCAB)}

N_FREQ = 8
MAX_OPS = 8
MAX_PARAMS = 6
PARAM_FEAT_DIM = 2 + 2 * N_FREQ


def operator_vocab(legacy=False):
    return OPERATOR_VOCAB_FULL[:-1] if legacy else list(OPERATOR_VOCAB_FULL)


def op_ids(operators, legacy=False):
    vocab = operator_vocab(legacy)
    op2id = {o: i for i, o in enumerate(vocab)}
    ids = [op2id.get(o, 0) for o in operators][:MAX_OPS]
    multihot = np.zeros(len(vocab), np.float32)
    for i in ids:
        multihot[i] = 1.0
    ids = ids + [0] * (MAX_OPS - len(ids))
    return np.array(ids, np.int32), multihot


def fit_multihot(mh, legacy=False):
    """Bring a stored multi-hot to the model's vocabulary length (legacy: drop the trailing phase_change token)."""
    mh = np.asarray(mh, np.float32)
    n = len(operator_vocab(legacy))
    if len(mh) == n:
        return mh
    if len(mh) == n + 1:
        assert float(mh[-1]) == 0.0, "sample uses 'phase_change', which a legacy-vocabulary model cannot represent"
        return mh[:-1]
    raise ValueError(f"op_multihot of length {len(mh)} for a {n}-token vocabulary")


def _logfourier(value):
    v = float(value)
    s = np.sign(v)
    lm = np.log10(abs(v) + 1e-12)
    bands = lm / (2.0 ** np.arange(N_FREQ))
    return np.concatenate([[s, lm / 10.0], np.sin(bands), np.cos(bands)]).astype(np.float32)


def param_features(names, values):
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


def encode(spec, param_values=None, legacy=False):
    """FamilySpec -> conditioning arrays (same as the reference `encode`)."""
    oids, omh = op_ids(spec.operators, legacy)
    names = list(spec.param_names)
    if param_values is not None:
        vals = [param_values.get(n) for n in names]
    elif spec.param_source == "const":
        vals = list(spec.param_const) + [None] * (len(names) - len(spec.param_const))
    else:
        vals = [None] * len(names)
    pids, pfeat, pmask = param_features(names, vals)
    return {"op_ids": oids, "op_multihot": omh, "param_ids": pids, "param_feats": pfeat, "param_mask": pmask,
            "symbolic": spec.symbolic}
