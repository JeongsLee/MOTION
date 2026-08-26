"""Symbolic equation / IC / BC conditioning (DESIGN_V3 §2.1, requirement 3).

The IVP contract is (equation, IC, BC) -> solution. IC enters structurally (the AR window +
per-step hard anchor). This module carries the other two as an explicit conditioning vector:

  cond = [operator multi-hot | masked param log-Fourier features | param-name multi-hot |
          per-axis BC one-hot | steady flag | K one-hot]

built ONCE per example (numpy) from the registry FamilySpec + the sample's resolved params,
and consumed model-side by CondEncoder -> e_cond, which FiLM-modulates every core block and
biases the decoder. All FiLM projections are zero-init so conditioning starts inert.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from data.registry import FAMILIES
from data.symbolic import MAX_PARAMS, OPERATOR_VOCAB, PARAM_FEAT_DIM, PARAM_VOCAB

BC_TYPES = ("periodic", "wall", "open")
COND_DIM = (len(OPERATOR_VOCAB) + MAX_PARAMS * PARAM_FEAT_DIM + len(PARAM_VOCAB)
            + len(BC_TYPES) * 3 + 1 + 2)


def build_cond(ex):
    """Training-example dict (loader/make_example fields) -> (COND_DIM,) float32."""
    spec = FAMILIES[ex["family"]]
    op_mh = np.asarray(ex["op_multihot"], np.float32)
    pf = (np.asarray(ex["param_feats"], np.float32)
          * np.asarray(ex["param_mask"], np.float32)[:, None]).reshape(-1)
    pname = np.zeros(len(PARAM_VOCAB), np.float32)
    for pid in np.asarray(ex["param_ids"]).reshape(-1):
        pname[int(pid)] = 1.0
    pname[0] = 0.0                                            # <pad> is not a name
    bc = np.zeros((3, len(BC_TYPES)), np.float32)
    for a in range(3):
        t = spec.bc[a] if a < len(spec.bc) else "periodic"
        bc[a, BC_TYPES.index(t) if t in BC_TYPES else 0] = 1.0
    steady = np.array([1.0 if ex["steady"] else 0.0], np.float32)
    kk = np.zeros(2, np.float32)
    kk[int(ex["K"]) - 2] = 1.0
    v = np.concatenate([op_mh, pf, pname, bc.reshape(-1), steady, kk])
    assert v.shape[0] == COND_DIM, (v.shape, COND_DIM)
    return v.astype(np.float32)


class CondEncoder(tf.keras.layers.Layer):
    """cond (B, COND_DIM) -> e (B, d_cond) + per-block FiLM (gamma, beta), zero-init."""

    def __init__(self, d_cond, d_core, n_blocks, name="cond", **kw):
        super().__init__(name=name, **kw)
        self.h1 = tf.keras.layers.Dense(d_cond, activation="gelu", name=f"{name}_h1")
        self.h2 = tf.keras.layers.Dense(d_cond, activation="gelu", name=f"{name}_h2")
        self.film = [tf.keras.layers.Dense(2 * d_core, kernel_initializer="zeros",
                                           name=f"{name}_film{i}") for i in range(n_blocks)]

    def call(self, cond):
        return self.h2(self.h1(cond))                         # e (B, d_cond)

    def modulate(self, i, e, h):
        """FiLM block i: h (B,*dims,d) -> h*(1+gamma)+beta with (gamma,beta)=film_i(e)."""
        gb = self.film[i](e)                                  # (B, 2d)
        d = h.shape[-1]
        shape = [-1] + [1] * (h.shape.rank - 2) + [2 * d]
        gb = tf.reshape(gb, shape)
        return h * (1.0 + gb[..., :d]) + gb[..., d:]
