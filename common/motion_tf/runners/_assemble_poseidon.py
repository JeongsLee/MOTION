#!/usr/bin/env python
"""Reassemble Poseidon/PDEgym chunked NetCDF (velocity_0.nc, velocity_1.nc, ...) into one .nc per
dataset, concatenating along the `sample` dim. Equivalent to the repo's assemble_data.py but:
  - writes to a SEPARATE _assembled/<ds>.nc (so re-globbing never picks up its own output),
  - copies in small sample-BLOCKS (bounds RAM well under the 13GB CPU box; stock script does
    variable[:] = a full ~5GB chunk),
  - resume-safe (skips a dataset whose assembled file already has the full sample count),
  - groups chunks by filename prefix and refuses mixed-prefix dirs (the stock script would KeyError).
CPU/IO only -> runs in parallel with GPU phase-1. env: DSETS (space-sep; default = 6 pretraining sets).
"""
import os, sys, glob, re
import numpy as np
from netCDF4 import Dataset

ROOT = os.environ.get("POSE_ROOT", "/code-vol/data/poseidon")
# POSE_OUT lets the big assembled .nc be written to LOCAL disk (read chunks from FUSE, write local) —
# avoids the 82GB single-file FUSE write that corrupts (B-tree error, the NS-Gauss saga).
OUT  = os.environ.get("POSE_OUT", os.path.join(ROOT, "_assembled"))
BLK  = int(os.environ.get("BLK", "8"))            # samples copied per read/write step
DSETS = os.environ.get("DSETS",
        "CE-RP CE-KH CE-CRP CE-Gauss NS-Sines NS-Gauss").split()
os.makedirs(OUT, exist_ok=True)


def chunk_files(d):
    """Return (prefix, [sorted chunk paths]) for dataset dir d, or (None, [])."""
    ncs = [f for f in os.listdir(d) if f.endswith(".nc")]
    groups = {}
    for f in ncs:
        m = re.match(r"(.+?)_(\d+)\.nc$", f)
        if not m:
            continue
        groups.setdefault(m.group(1), []).append((int(m.group(2)), f))
    if not groups:
        return None, []
    if len(groups) > 1:
        print(f"  !! multiple prefixes {list(groups)} in {d} -- skipping", flush=True)
        return None, []
    pref = next(iter(groups))
    files = [os.path.join(d, f) for _, f in sorted(groups[pref])]
    return pref, files


def assemble(ds):
    d = os.path.join(ROOT, ds)
    if not os.path.isdir(d):
        print(f"[{ds}] dir missing -- skip", flush=True)
        return
    pref, files = chunk_files(d)
    if not files:
        print(f"[{ds}] no chunk files -- skip", flush=True)
        return
    # DOWNSTREAM N-shot only needs a small subset (max-N train + test). CHUNK_LIMITS="NS-PwC:2 ACE:1"
    # caps chunks per dataset to avoid a full cross-region copy of all 16+ chunks.
    limits = dict(kv.split(":") for kv in os.environ.get("CHUNK_LIMITS", "").split() if ":" in kv)
    maxc = int(limits.get(ds, "0"))
    if maxc > 0 and len(files) > maxc:
        print(f"[{ds}] limiting {len(files)} chunks -> first {maxc} (N-shot subset)", flush=True)
        files = files[:maxc]

    # dims + total sample count from the chunks
    per = []
    with Dataset(files[0], "r") as nc0:
        dims = nc0.dimensions
        num_times = dims["time"].size
        num_ch = dims["channel"].size if "channel" in dims else None
        xs, ys = dims["x"].size, dims["y"].size
        dtype = nc0.variables[pref].dtype
        per.append(dims["sample"].size)
    for f in files[1:]:
        with Dataset(f, "r") as nc:
            per.append(nc.dimensions["sample"].size)
    total = sum(per)

    outf = os.path.join(OUT, f"{ds}.nc")
    if os.path.exists(outf):
        try:
            with Dataset(outf, "r") as o:
                if o.dimensions["sample"].size == total and pref in o.variables:
                    print(f"[{ds}] already assembled ({total} samples) -- skip", flush=True)
                    return
        except Exception:
            pass  # corrupt/partial -> rewrite

    print(f"[{ds}] {len(files)} chunks -> {total} samples, t={num_times} c={num_ch} {xs}x{ys} "
          f"var='{pref}' -> {outf}", flush=True)
    vdims = ("sample", "time") + (("channel",) if num_ch else ()) + ("x", "y")
    chunksz = (1, 1) + ((num_ch,) if num_ch else ()) + (xs, ys)
    with Dataset(outf, "w") as out:
        out.createDimension("sample", total)
        out.createDimension("time", num_times)
        if num_ch:
            out.createDimension("channel", num_ch)
        out.createDimension("x", xs)
        out.createDimension("y", ys)
        ov = out.createVariable(pref, dtype, vdims, chunksizes=chunksz)
        off = 0
        for fi, f in enumerate(files):
            with Dataset(f, "r") as nc:
                v = nc.variables[pref]
                n = nc.dimensions["sample"].size
                for s in range(0, n, BLK):
                    e = min(s + BLK, n)
                    ov[off + s: off + e] = v[s:e]
                off += n
            print(f"  [{ds}] chunk {fi+1}/{len(files)} done ({off}/{total})", flush=True)
    print(f"[{ds}] SAVED {outf}", flush=True)


def assemble_wave_layer(ds="Wave-Layer"):
    """Wave-Layer has TWO variables: solution_*.nc (sample,time,x,y) + a SINGLE static c_0.nc (sample,x,y)
    = the layered wave-speed field. We merge them into a 2-CHANNEL trajectory (sample,time,channel=2,x,y):
    channel0 = solution(t), channel1 = c broadcast over time (static field, sample-aligned). Downstream the
    c channel is fed (its frame-0 value informs the wave dynamics) and supervised to stay constant-in-time
    (Poseidon dummy-channel style) so it survives the rollout. No loader change needed."""
    d = os.path.join(ROOT, ds)
    sol_files = sorted(glob.glob(os.path.join(d, "solution_*.nc")),
                       key=lambda f: int(re.search(r"_(\d+)\.nc$", f).group(1)))
    c_file = os.path.join(d, "c_0.nc")
    if not sol_files or not os.path.exists(c_file):
        print(f"[{ds}] missing solution_*.nc or c_0.nc -- skip", flush=True); return
    limits = dict(kv.split(":") for kv in os.environ.get("CHUNK_LIMITS", "").split() if ":" in kv)
    maxc = int(limits.get(ds, "0"))
    if maxc > 0 and len(sol_files) > maxc:
        print(f"[{ds}] limiting {len(sol_files)} solution chunks -> first {maxc}", flush=True)
        sol_files = sol_files[:maxc]
    with Dataset(sol_files[0], "r") as nc0:
        num_times = nc0.dimensions["time"].size
        xs, ys = nc0.dimensions["x"].size, nc0.dimensions["y"].size
        dtype = nc0.variables["solution"].dtype
    per = []
    for f in sol_files:
        with Dataset(f, "r") as nc:
            per.append(nc.dimensions["sample"].size)
    total = sum(per)
    outf = os.path.join(OUT, f"{ds}.nc")
    if os.path.exists(outf):
        try:
            with Dataset(outf, "r") as o:
                if o.dimensions["sample"].size == total and o.dimensions.get("channel") and \
                   o.dimensions["channel"].size == 2:
                    print(f"[{ds}] already assembled ({total} samples, 2ch) -- skip", flush=True); return
        except Exception:
            pass
    print(f"[{ds}] {len(sol_files)} sol chunks -> {total} samples, t={num_times} 2ch {xs}x{ys} "
          f"(solution + c) -> {outf}", flush=True)
    with Dataset(outf, "w") as out, Dataset(c_file, "r") as cnc:
        out.createDimension("sample", total); out.createDimension("time", num_times)
        out.createDimension("channel", 2); out.createDimension("x", xs); out.createDimension("y", ys)
        ov = out.createVariable("solution", dtype, ("sample", "time", "channel", "x", "y"),
                                chunksizes=(1, 1, 2, xs, ys))
        cv = cnc.variables["c"]
        off = 0
        for fi, f in enumerate(sol_files):
            with Dataset(f, "r") as nc:
                sv = nc.variables["solution"]; n = nc.dimensions["sample"].size
                for s in range(0, n, BLK):
                    e = min(s + BLK, n)
                    sol = np.asarray(sv[s:e]).astype(dtype)                  # (b,T,H,W)
                    cc = np.asarray(cv[off + s:off + e]).astype(dtype)       # (b,H,W) sample-aligned
                    cc_b = np.broadcast_to(cc[:, None], sol.shape)           # (b,T,H,W)
                    ov[off + s:off + e] = np.stack([sol, cc_b], axis=2)      # (b,T,2,H,W)
                off += n
            print(f"  [{ds}] chunk {fi+1}/{len(sol_files)} done ({off}/{total})", flush=True)
    print(f"[{ds}] SAVED {outf}", flush=True)


def assemble_wave_layer_scot(ds="Wave-Layer"):
    """scOT-FORMAT Wave-Layer.nc = SEPARATE variables `solution`(sample,time,x,y) + `c`(sample,x,y),
    exactly what scOT/problems/wave/acoustic.py:Layer reads (reader['solution'][i,t], reader['c'][i]).
    DISTINCT from assemble_wave_layer (our merged 2-channel `solution`). Written to a SEPARATE dir
    (SCOT_OUT, default OUT/_scot) so it doesn't clobber our merged Wave-Layer.nc (same filename)."""
    d = os.path.join(ROOT, ds)
    sol_files = sorted(glob.glob(os.path.join(d, "solution_*.nc")),
                       key=lambda f: int(re.search(r"_(\d+)\.nc$", f).group(1)))
    c_file = os.path.join(d, "c_0.nc")
    if not sol_files or not os.path.exists(c_file):
        print(f"[{ds}-scot] missing files -- skip", flush=True); return
    limits = dict(kv.split(":") for kv in os.environ.get("CHUNK_LIMITS", "").split() if ":" in kv)
    maxc = int(limits.get(ds, "0"))
    if maxc > 0 and len(sol_files) > maxc:
        print(f"[{ds}-scot] limiting -> first {maxc} solution chunks", flush=True)
        sol_files = sol_files[:maxc]
    with Dataset(sol_files[0], "r") as nc0:
        num_times = nc0.dimensions["time"].size
        xs, ys = nc0.dimensions["x"].size, nc0.dimensions["y"].size
        sdt = nc0.variables["solution"].dtype
    per = []
    for f in sol_files:
        with Dataset(f, "r") as nc:
            per.append(nc.dimensions["sample"].size)
    total = sum(per)
    out_dir = os.environ.get("SCOT_OUT", os.path.join(OUT, "_scot")); os.makedirs(out_dir, exist_ok=True)
    outf = os.path.join(out_dir, f"{ds}.nc")
    if os.path.exists(outf):
        try:
            with Dataset(outf, "r") as o:
                if o.dimensions["sample"].size == total and "c" in o.variables:
                    print(f"[{ds}-scot] already assembled ({total}) -- skip", flush=True); return
        except Exception:
            pass
    print(f"[{ds}-scot] {len(sol_files)} chunks -> {total} samples (separate solution + c) -> {outf}", flush=True)
    with Dataset(outf, "w") as out, Dataset(c_file, "r") as cnc:
        out.createDimension("sample", total); out.createDimension("time", num_times)
        out.createDimension("x", xs); out.createDimension("y", ys)
        cv = cnc.variables["c"]
        svo = out.createVariable("solution", sdt, ("sample", "time", "x", "y"), chunksizes=(1, 1, xs, ys))
        cvo = out.createVariable("c", cv.dtype, ("sample", "x", "y"), chunksizes=(1, xs, ys))
        off = 0
        for fi, f in enumerate(sol_files):
            with Dataset(f, "r") as nc:
                sv = nc.variables["solution"]; n = nc.dimensions["sample"].size
                for s in range(0, n, BLK):
                    e = min(s + BLK, n)
                    svo[off + s:off + e] = np.asarray(sv[s:e])
                    cvo[off + s:off + e] = np.asarray(cv[off + s:off + e])
                off += n
            print(f"  [{ds}-scot] chunk {fi+1}/{len(sol_files)} done ({off}/{total})", flush=True)
    print(f"[{ds}-scot] SAVED {outf}", flush=True)


if __name__ == "__main__":
    print(f"reassemble: {DSETS}  (BLK={BLK}, OUT={OUT})", flush=True)
    for ds in DSETS:
        try:
            if ds == "Wave-Layer":
                (assemble_wave_layer_scot if os.environ.get("WAVE_SCOT") else assemble_wave_layer)(ds); continue
            assemble(ds)
        except Exception as e:
            print(f"[{ds}] FAILED: {e}", flush=True)
    print("POSEIDON_ASSEMBLE_DONE", flush=True)
