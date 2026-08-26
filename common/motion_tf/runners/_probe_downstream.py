"""Probe downstream PDEgym datasets: locate raw chunks (/eu vs /code-vol) and dump .nc structure."""
import os, glob, re
from netCDF4 import Dataset
DSETS = ["NS-PwC", "CE-RM", "Wave-Layer", "ACE"]
ROOTS = ["/eu/data/poseidon", "/code-vol/data/poseidon"]
for ds in DSETS:
    print(f"\n=== {ds} ===", flush=True)
    found = None
    for root in ROOTS:
        d = os.path.join(root, ds)
        if os.path.isdir(d):
            ncs = sorted([f for f in os.listdir(d) if f.endswith(".nc")])
            print(f"  {root}/{ds}: {len(ncs)} .nc files: {ncs[:6]}{'...' if len(ncs)>6 else ''}", flush=True)
            if ncs and found is None:
                found = os.path.join(d, ncs[0])
        else:
            print(f"  {root}/{ds}: MISSING", flush=True)
    # also check assembled
    for root in ROOTS:
        af = os.path.join(root, "_assembled", f"{ds}.nc")
        if os.path.exists(af):
            print(f"  ASSEMBLED already at {af}", flush=True); found = found or af
    if found:
        try:
            with Dataset(found, "r") as nc:
                print(f"  HEADER {found}", flush=True)
                print(f"    dims: { {k:v.size for k,v in nc.dimensions.items()} }", flush=True)
                print(f"    vars: {list(nc.variables.keys())}", flush=True)
                for vn, v in nc.variables.items():
                    if v.ndim >= 3:
                        print(f"    var '{vn}': shape={v.shape} dtype={v.dtype} dims={v.dimensions}", flush=True)
        except Exception as e:
            print(f"  HEADER read FAIL: {type(e).__name__}: {e}", flush=True)
print("\nPROBE_DONE", flush=True)
