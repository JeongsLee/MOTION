"""Preprocess CLI: enumerate units -> write split manifest -> materialize unified cache.

    python -m data.preprocess --families incom_ns,pdebench3d_cns_Rand_M0.1... [--manifest-only]
                              [--max-per-fam N] [--out /corpus/cache/unified]

For each family: enumerate unit ids (headers only), build/write the split manifest,
then (unless --manifest-only) materialize each unit through its adapter and append to
per-(family,split) shards of UnifiedSample dicts (npz). Grid families store dense
fields; mesh families store coords+fields. Steady families keep T=1.

RAM-safe: one unit in memory at a time; shards flushed every --shard-units.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from . import adapters, splits
from .registry import FAMILIES


def _official_split(spec, unit_ids):
    """Return an official split dict if the source publishes one, else None.
    (Hook: DrivAerNet/AirfRANS/Geo-FNO id files parsed here when present; falls back
    to derived when the id files are absent so preprocessing never blocks.)"""
    if not spec.split_source.startswith("official"):
        return None
    # id files land under /corpus/meta/official/<family>/{train,val,test}.txt
    base = os.path.join(os.environ.get("CORPUS_ROOT", "/corpus"),
                        "meta", "official", spec.name)
    got = {}
    for sp in ("train", "val", "test"):
        fp = os.path.join(base, f"{sp}.txt")
        if os.path.exists(fp):
            got[sp] = [l.strip() for l in open(fp) if l.strip()]
    if len(got) == 3:
        return got
    return None                                    # not staged yet -> derived fallback


def _sample_to_npz(s):
    d = {"family": s.family, "K": s.K, "mode": s.mode, "fields": s.fields,
         "cmask": s.cmask, "tmask": s.tmask, "roles": np.array(s.roles),
         "op_ids": s.cond["op_ids"], "op_multihot": s.cond["op_multihot"],
         "param_ids": s.cond["param_ids"], "param_feats": s.cond["param_feats"],
         "param_mask": s.cond["param_mask"], "symbolic": s.cond["symbolic"]}
    if s.mode == "grid":
        d["dims"] = np.array(s.dims)
    else:
        d["coords"] = s.coords
    if s.geom is not None:
        d["geom"] = s.geom
    if getattr(s, "sdf_vol", None) is not None:
        d["sdf_vol"] = s.sdf_vol            # (R,)*K dense shape context (GINO lever)
    return d


def process_family(spec, out_dir, manifest_only=False, max_per_fam=None, shard_units=64):
    unit_refs = list(adapters.units(spec)) if spec.fmt in adapters.IMPLEMENTED else []
    if not unit_refs:
        print(f"[{spec.name}] fmt={spec.fmt} not implemented — SKIP (manifest deferred)", flush=True)
        return
    unit_ids = [u for u, _ in unit_refs]
    manifest = splits.make_manifest(spec, unit_ids, official=_official_split(spec, unit_ids))
    splits.write_manifest(manifest)
    print(f"[{spec.name}] {len(unit_ids)} units -> split {manifest['counts']} ({manifest['source']})",
          flush=True)
    if manifest_only:
        return

    which = {u: sp for sp in ("train", "val", "test") for u in manifest["splits"][sp]}
    ref_by_id = dict(unit_refs)
    buf = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    seen = 0
    fam_dir = os.path.join(out_dir, spec.name)
    os.makedirs(fam_dir, exist_ok=True)

    def flush(sp):
        if not buf[sp]:
            return
        k = counts[sp]
        np.savez_compressed(os.path.join(fam_dir, f"{sp}_{k:04d}.npz"),
                            samples=np.array(buf[sp], dtype=object))
        counts[sp] += 1
        buf[sp] = []

    for uid in unit_ids:
        if max_per_fam and seen >= max_per_fam:
            break
        sp = which[uid]
        s = adapters.load(spec, ref_by_id[uid])
        buf[sp].append(_sample_to_npz(s))
        seen += 1
        if len(buf[sp]) >= shard_units:
            flush(sp)
    for sp in buf:
        flush(sp)
    print(f"[{spec.name}] materialized {seen} units -> shards {counts}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="all")
    ap.add_argument("--out", default=os.path.join(os.environ.get("CORPUS_ROOT", "/corpus"),
                                                   "cache", "unified"))
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--max-per-fam", type=int, default=0)
    ap.add_argument("--shard-units", type=int, default=64)
    a = ap.parse_args()
    names = list(FAMILIES) if a.families == "all" else a.families.split(",")
    print(f"preprocess: {len(names)} families -> {a.out}  (manifest_only={a.manifest_only})", flush=True)
    for nm in names:
        if nm not in FAMILIES:
            print(f"UNKNOWN family {nm}", flush=True)
            continue
        try:
            process_family(FAMILIES[nm], a.out, a.manifest_only,
                           a.max_per_fam or None, a.shard_units)
        except Exception as e:
            print(f"[{nm}] ERROR {type(e).__name__}: {e}", flush=True)
    print("PREPROCESS_DONE", flush=True)


if __name__ == "__main__":
    main()
