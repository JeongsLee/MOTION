"""Rebuild ALL family caches with the dt=1 + sliding-window preprocessing (consecutive frames, matching
PROSE; sliding windows boost sample counts — esp incom 340→~3.5k). Runs 5 families in parallel
processes. Old caches are removed first. Run on a many-CPU node (PROSE_DATA_DIR set to the volume)."""
import os, shutil, glob, sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, "/tmp/work")            # where the code/ tree is copied
from motion_tf.data import prose

PRE = os.environ.get("PREBUILT_DIR", "/code-vol/data/prose/prebuilt")
N = int(os.environ.get("NPER", "100000"))
T = int(os.environ.get("TNUM", "20"))


def build(fam):
    try:
        if fam in ("shallow_water", "diff_react"):
            p = os.path.join(PRE, f"{fam}_n{N}_t{T}.npy")
            if os.path.exists(p):
                os.remove(p)
            arr = prose.load_swe(N, T) if fam == "shallow_water" else prose.load_diffreact(N, T)
            np.save(p, np.ascontiguousarray(arr))
            return f"{fam}: {arr.shape} -> {p}"
        d = os.path.join(PRE, f"{fam}_n{N}_t{T}_shards")
        if os.path.exists(d):
            shutil.rmtree(d)
        if fam == "com_ns":
            w = prose._write_comns_shards(N, T, d)
        elif fam == "pdearena_ns":
            w = prose._write_pdearena_shards(N, T, d)
        elif fam == "pdearena_uncond":         # 14-frame uncond set — build at TNUM<=14
            w = prose._write_pdearena_uncond_shards(N, T, d)
        else:                                  # incom_ns
            w = prose._write_incomns_shards(N, T, d)
        return f"{fam}: {w} samples -> {d}"
    except Exception as e:
        import traceback
        return f"{fam}: ERROR {e}\n{traceback.format_exc()}"


if __name__ == "__main__":
    _all = ["shallow_water", "diff_react", "com_ns", "pdearena_ns", "incom_ns"]
    fams = os.environ.get("FAMILIES", ",".join(_all)).split(",")   # subset rebuild via FAMILIES env
    print(f"rebuild dt=1 caches | N={N} T={T} | families={fams}", flush=True)
    with Pool(len(fams)) as pool:
        for r in pool.imap_unordered(build, fams):
            print("DONE", r, flush=True)
    print("ALL REBUILT", flush=True)
