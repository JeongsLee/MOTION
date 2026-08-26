"""Split-manifest generation (foundationv2, DESIGN.md §4c).

Per family: OFFICIAL split if the source provides one, else a DETERMINISTIC
80/10/10 train/val/test partition over the split UNIT (trajectory/design/case)
using a fixed seed. Manifests are written once to /corpus/meta/splits/<family>.json
and are the ONLY thing preprocessing/training read to decide membership — no
leakage from window/frame-level shuffling (units are whole trajectories/designs).

A manifest entry lists unit IDs per split; how a unit maps to raw storage (file,
index, shard) is the adapter's job, but MUST be deterministic given the unit id.
"""
from __future__ import annotations

import hashlib
import json
import os

FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _default_dir():
    return os.path.join(os.environ.get("CORPUS_ROOT", "/corpus"), "meta", "splits")


def deterministic_split(unit_ids, seed=0):
    """Stable hash-based assignment (order-independent, reproducible, incremental-safe)."""
    ids = list(unit_ids)
    buckets = {"train": [], "val": [], "test": []}
    for uid in ids:
        h = int(hashlib.sha1(f"{seed}:{uid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < FRACS["train"]:
            buckets["train"].append(uid)
        elif h < FRACS["train"] + FRACS["val"]:
            buckets["val"].append(uid)
        else:
            buckets["test"].append(uid)
    return buckets


def make_manifest(spec, unit_ids, official=None, seed=0):
    """official: {"train":[...],"val":[...],"test":[...]} when split_source starts 'official'."""
    if official is not None:
        buckets = official
        source = spec.split_source
    else:
        buckets = deterministic_split(unit_ids, seed)
        source = f"derived:sha1:seed{seed}:80-10-10"
    return {
        "family": spec.name, "K": spec.K, "unit": spec.unit, "stage": spec.stage,
        "source": source, "seed": seed,
        "counts": {k: len(v) for k, v in buckets.items()},
        "splits": buckets,
    }


def write_manifest(manifest, out_dir=None):
    out_dir = out_dir or _default_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{manifest['family']}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    return path


def load_manifest(family, out_dir=None):
    out_dir = out_dir or _default_dir()
    with open(os.path.join(out_dir, f"{family}.json")) as f:
        return json.load(f)
