"""Name-keyed weight save/load + positional->named migration (SEPARATE from the legacy ckpt.py).

The legacy ckpt.py loads by POSITION (v{i} = trainable_variables construction order). Any architecture
change that INSERTS or reorders variables (e.g. adding a reaction branch inside an expert) shifts every
subsequent index -> positional load silently corrupts. This module keys by variable NAME instead:
  - existing variables match by name regardless of order,
  - NEW variables absent from the checkpoint are simply skipped (left at their fresh init).

Format: a named ckpt is a normal positional npz (v0..vN) PLUS a "__names__" string array (the manifest).
The legacy ckpt.load auto-detects "__names__" and delegates here, so legacy positional ckpts are unaffected.
"""
from __future__ import annotations
import numpy as np


def save_named(model, path):
    vs = list(model.trainable_variables)
    arrs = {f"v{i}": v.numpy() for i, v in enumerate(vs)}
    names = [v.name for v in vs]
    if len(set(names)) != len(names):
        # name collisions would make name-keyed load ambiguous; refuse rather than corrupt silently.
        dupes = {n for n in names if names.count(n) > 1}
        raise ValueError(f"save_named: duplicate variable names {sorted(dupes)[:5]}... -> use positional ckpt")
    arrs["__names__"] = np.array(names)
    np.savez(path, **arrs)
    print(f"[ckpt_named] saved {len(vs)} named vars -> {path}", flush=True)


def load_named(model, path, verbose=True):
    """Assign by variable name; ckpt-missing vars stay at init, model-missing ckpt entries are ignored."""
    d = np.load(path, allow_pickle=False)
    if "__names__" not in d:
        raise ValueError(f"{path} is not a named ckpt (no __names__); load it positionally via ckpt.load")
    ck_names = [str(n) for n in d["__names__"]]
    name2val = {nm: d[f"v{i}"] for i, nm in enumerate(ck_names)}
    loaded = missing = 0
    for v in model.trainable_variables:
        if v.name in name2val:
            v.assign(name2val[v.name]); loaded += 1
        else:
            missing += 1
            if verbose:
                print(f"  [ckpt_named] FRESH-INIT (not in ckpt): {v.name}", flush=True)
    extra = len(ck_names) - loaded
    print(f"[ckpt_named] loaded {loaded} by name | {missing} fresh-init (new) | {extra} ckpt entries unused",
          flush=True)
    return loaded, missing


def migrate(model, pos_path, out_path):
    """One-time: load a legacy POSITIONAL ckpt into `model` (which must be the SAME arch that saved it),
    then re-save as a NAMED ckpt so a later (architecturally-extended) model can load it by name."""
    from . import ckpt as _legacy
    _legacy.load(model, pos_path)
    save_named(model, out_path)
    print(f"[ckpt_named] migrated {pos_path} -> {out_path}", flush=True)
