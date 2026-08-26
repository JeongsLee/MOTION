"""Convert a PDEgym dataset's SOURCE chunks (velocity_*.nc / data_*.nc, ~4.86GB each, readable in small
slices) directly into native .npy SHARDS — bypassing the monolithic 82GB .nc whose FUSE write corrupts
(metadata OK, all data reads -> 'NetCDF: HDF error'). .npy is a sequential write → FUSE-safe. The Phase-2
loader (poseidon.py) reads <ds>_shards/ when present. env: DSETS (space-sep), SHARD (samples per shard).
Reads chunks in BLK-sample slices (small slices read fine over FUSE; it's only the big single-.nc WRITE
that corrupts). Output: /code-vol/data/poseidon/_assembled/<ds>_shards/shard_XXXX.npy native (n,21,C,128,128).
"""
from __future__ import annotations
import os, re
import numpy as np
from netCDF4 import Dataset

ROOT = os.environ.get("POSE_ROOT", "/code-vol/data/poseidon")
OUT = os.path.join(ROOT, "_assembled")
SHARD = int(os.environ.get("SHARD", "256"))
BLK = int(os.environ.get("BLK", "8"))
DSETS = os.environ.get("DSETS", "NS-Gauss").split()


def chunk_files(d):
    ncs = [f for f in os.listdir(d) if f.endswith(".nc")]
    groups = {}
    for f in ncs:
        m = re.match(r"(.+?)_(\d+)\.nc$", f)
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), f))
    if not groups:
        return None, []
    pref = next(iter(groups))
    return pref, [os.path.join(d, f) for _, f in sorted(groups[pref])]


def convert(ds):
    d = os.path.join(ROOT, ds)
    pref, files = chunk_files(d)
    if not files:
        print(f"[{ds}] no source chunks in {d}", flush=True); return
    od = os.path.join(OUT, f"{ds}_shards"); os.makedirs(od, exist_ok=True)
    if os.path.exists(os.path.join(od, "_DONE")):
        print(f"[{ds}] already done -> {od}", flush=True); return
    print(f"[{ds}] {len(files)} chunks var='{pref}' -> {od} (SHARD={SHARD})", flush=True)
    buf = []; sh = 0; total = 0
    def flush():
        nonlocal buf, sh
        if not buf:
            return
        arr = np.concatenate(buf, axis=0).astype(np.float32)
        np.save(os.path.join(od, f"shard_{sh:04d}.npy"), arr)
        print(f"  [{ds}] shard_{sh:04d} {arr.shape}", flush=True)
        sh += 1; buf = []
    for fi, f in enumerate(files):
        with Dataset(f, "r") as nc:
            v = nc.variables[pref]; n = nc.dimensions["sample"].size
            for s in range(0, n, BLK):
                e = min(s + BLK, n)
                buf.append(np.asarray(v[s:e]).astype(np.float32))   # (be,21,C,128,128)
                total += e - s
                if sum(b.shape[0] for b in buf) >= SHARD:
                    flush()
        print(f"  [{ds}] chunk {fi+1}/{len(files)} read ({total} samples)", flush=True)
    flush()
    open(os.path.join(od, "_DONE"), "w").write(str(total))
    print(f"[{ds}] DONE {total} samples -> {sh} shards", flush=True)


if __name__ == "__main__":
    for ds in DSETS:
        try:
            convert(ds)
        except Exception as e:
            print(f"[{ds}] FAILED: {type(e).__name__}: {e}", flush=True)
    print("NC_TO_SHARDS_DONE", flush=True)
