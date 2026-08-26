"""Find TRUNCATED .npy shards (FUSE write corruption): the .npy header declares a shape whose byte size
exceeds the actual file size. Reports every bad file per family so they can be rewritten."""
import os, glob
import numpy as np
from numpy.lib import format as npf

PRE = os.environ.get("PREBUILT_DIR", "/code-vol/data/prose/prebuilt")
N, T = 100000, 20
FAMS = ["shallow_water", "diff_react", "com_ns", "pdearena_ns", "incom_ns"]


def files(fam):
    p = os.path.join(PRE, f"{fam}_n{N}_t{T}.npy")
    if os.path.exists(p):
        return [p]
    return sorted(glob.glob(os.path.join(PRE, f"{fam}_n{N}_t{T}_shards", "shard_*.npy")))


def expected_bytes(f):
    with open(f, "rb") as fh:
        ver = npf.read_magic(fh)
        shape, fortran, dtype = npf._read_array_header(fh, ver)
        header_end = fh.tell()
    return header_end + int(np.prod(shape)) * dtype.itemsize, shape


for fam in FAMS:
    fs = files(fam)
    if not fs:
        print(f"{fam}: no cache", flush=True); continue
    bad = []; ok = 0; total_rows = 0
    for f in fs:
        try:
            exp, shape = expected_bytes(f)
            actual = os.path.getsize(f)
            if actual < exp:
                bad.append((os.path.basename(f), shape, actual, exp))
            else:
                ok += 1; total_rows += int(shape[0])
        except Exception as e:
            bad.append((os.path.basename(f), "HEADER_ERR", str(e), 0))
    print(f"{fam:14s} files={len(fs)} ok={ok} rows_ok={total_rows} BAD={len(bad)}", flush=True)
    for b in bad[:20]:
        print(f"    TRUNC {b[0]} shape={b[1]} actual={b[2]} expected={b[3]}", flush=True)
print("INTEGRITY DONE", flush=True)
