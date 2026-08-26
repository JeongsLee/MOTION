"""Check each family's _shards dir directly (what stream actually uses when _DONE present): _DONE marker,
shard count, and any TRUNCATED shard. Also flag the standalone truncated .npy."""
import os, glob
import numpy as np
from numpy.lib import format as npf

PRE = os.environ.get("PREBUILT_DIR", "/code-vol/data/prose/prebuilt")
N, T = 100000, 20
FAMS = ["shallow_water", "diff_react", "com_ns", "pdearena_ns", "incom_ns"]


def exp_bytes(f):
    with open(f, "rb") as fh:
        ver = npf.read_magic(fh); shape, fo, dt = npf._read_array_header(fh, ver); h = fh.tell()
    return h + int(np.prod(shape)) * dt.itemsize, shape


for fam in FAMS:
    npy = os.path.join(PRE, f"{fam}_n{N}_t{T}.npy")
    shdir = os.path.join(PRE, f"{fam}_n{N}_t{T}_shards")
    # standalone .npy
    if os.path.exists(npy):
        try:
            e, sh = exp_bytes(npy); a = os.path.getsize(npy)
            print(f"{fam} .npy: shape={sh} {'TRUNC' if a < e else 'ok'} actual={a} exp={e}", flush=True)
        except Exception as ex:
            print(f"{fam} .npy: HEADER_ERR {ex}", flush=True)
    # shards dir
    if os.path.isdir(shdir):
        done = os.path.exists(os.path.join(shdir, "_DONE"))
        shs = sorted(glob.glob(os.path.join(shdir, "shard_*.npy")))
        bad = []
        for f in shs:
            try:
                e, sh = exp_bytes(f); a = os.path.getsize(f)
                if a < e:
                    bad.append((os.path.basename(f), sh, a, e))
            except Exception as ex:
                bad.append((os.path.basename(f), "HDR_ERR", str(ex), 0))
        print(f"{fam} _shards: _DONE={done} nshards={len(shs)} BAD={len(bad)}", flush=True)
        for b in bad[:30]:
            print(f"    TRUNC {b[0]} shape={b[1]} actual={b[2]} exp={b[3]}", flush=True)
print("SHARDCHECK DONE", flush=True)
