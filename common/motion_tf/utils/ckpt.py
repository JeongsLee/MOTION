"""Minimal weight save/load for tf.Module models (trainable_variables, by construction order).
Valid because the architecture is deterministic given a fixed config."""
from __future__ import annotations
import numpy as np


def save(model, path):
    vs = model.trainable_variables
    np.savez(path, **{f"v{i}": v.numpy() for i, v in enumerate(vs)})


def load(model, path):
    d = np.load(path)
    if "__names__" in d:                       # NAMED ckpt -> delegate (name-keyed, tolerates new vars)
        from . import ckpt_named               # legacy positional ckpts have no __names__ -> unaffected
        return ckpt_named.load_named(model, path)
    for i, v in enumerate(model.trainable_variables):
        v.assign(d[f"v{i}"])
