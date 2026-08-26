"""Convert our per-family .npy caches (N,T=20,128,128,C) into the HDF5 layouts the FM baselines expect,
so MPP and DPOT can train on the IDENTICAL dt=1 data (Stage-1). Streams shard-by-shard (pdearena ~44GB)
so nothing huge is held in RAM. Writes a train/test split per family.

  TARGET=dpot : one monolithic <fam>_{train,test}.hdf5, key 'data', shape (n,128,128,T,C)  [T moved to pos 3]
  TARGET=mpp  : one <fam>_{train,test}.hdf5, sample-keyed groups '000000'/'data' = (T,128,128,C) channels-last

Run on deneb-kr (data is on the volume). Env: TARGET, OUT_DIR, TEST_FRAC, FAMILIES."""
import os, glob, sys
import numpy as np
import h5py

PRE = os.environ.get("PREBUILT_DIR", "/code-vol/data/prose/prebuilt")
OUT = os.environ.get("OUT_DIR", "/code-vol/data/prose/baseline_h5")
TARGET = os.environ.get("TARGET", "dpot").lower()
TEST_FRAC = float(os.environ.get("TEST_FRAC", "0.1"))
N, T = 100000, 20

# (cache stem, true channel count) — matches the verified dt=1 caches
FAM_C = {"shallow_water": 1, "diff_react": 2, "com_ns": 4, "pdearena_ns": 3, "incom_ns": 3, "cfdbench": 3}


def _shards(fam):
    """List of arrays-on-disk (mmap), in order. PREFER the _shards/ dir over a single .npy (matches
    stream._resolve_cache) — a stale/truncated pdearena .npy exists and must NOT be read."""
    d = os.path.join(PRE, f"{fam}_n{N}_t{T}_shards")
    sh = sorted(glob.glob(os.path.join(d, "shard_*.npy")))
    if sh:
        return sh
    p = os.path.join(PRE, f"{fam}_n{N}_t{T}.npy")
    return [p] if os.path.exists(p) else []


def _iter_samples(files):
    """Yield (sample (T,128,128,C)) across all shards, lazily."""
    for f in files:
        arr = np.load(f, mmap_mode="r")
        for i in range(arr.shape[0]):
            yield np.asarray(arr[i], dtype=np.float32)


def _counts(files):
    return [int(np.load(f, mmap_mode="r").shape[0]) for f in files]


def convert_family(fam):
    files = _shards(fam)
    if not files:
        print(f"{fam}: NO CACHE — skip", flush=True); return
    total = sum(_counts(files))
    n_test = max(1, int(round(total * TEST_FRAC)))
    n_train = total - n_test
    C = FAM_C[fam]
    os.makedirs(OUT, exist_ok=True)

    if TARGET == "dpot":
        # monolithic hdf5, key 'data', (n,128,128,T,C): transpose each sample (T,H,W,C)->(H,W,T,C).
        # single pass: route first n_train samples to train, rest to test.
        ftr = h5py.File(os.path.join(OUT, f"{fam}_train.hdf5"), "w")
        fte = h5py.File(os.path.join(OUT, f"{fam}_test.hdf5"), "w")
        dtr = ftr.create_dataset("data", shape=(n_train, 128, 128, T, C), dtype="float32")
        dte = fte.create_dataset("data", shape=(n_test, 128, 128, T, C), dtype="float32")
        itr = ite = 0
        for k, s in enumerate(_iter_samples(files)):
            s = np.transpose(s[..., :C], (1, 2, 0, 3))      # (T,H,W,C) -> (H,W,T,C)
            if k < n_train:
                dtr[itr] = s; itr += 1
            else:
                dte[ite] = s; ite += 1
        ftr.close(); fte.close()
        print(f"{fam}: DPOT train={itr} test={ite} C={C} -> {OUT}/{fam}_(train,test).hdf5", flush=True)

    elif TARGET == "mpp":
        # sample-keyed CHUNK files in a per-family subdir OUT/<fam>/chunk_NNNN.h5 (each group 'data'=(T,H,W,C)).
        # Per-family subdir avoids the include_string collision (com_ns vs incom_ns); chunking (<=CHUNK groups
        # ~<10GB/file) avoids the ~40GB single-file FUSE write corruption ("bad object header version number").
        # MPP globs the dir + aggregates samples (split_level='sample' → it makes its own train/val/test split),
        # so we write only the train portion (n_train) here; MPP splits it .8/.1/.1.
        CHUNK = int(os.environ.get("MPP_CHUNK", "2000"))
        fam_dir = os.path.join(OUT, fam); os.makedirs(fam_dir, exist_ok=True)
        for old in glob.glob(os.path.join(fam_dir, "*.h5")):
            os.remove(old)
        itr = 0; ci = 0; fh = None
        for k, s in enumerate(_iter_samples(files)):
            if k >= n_train:
                break
            if itr % CHUNK == 0:
                if fh is not None:
                    fh.close()
                fh = h5py.File(os.path.join(fam_dir, f"chunk_{ci:04d}.h5"), "w"); ci += 1
            fh.create_group(f"{itr:06d}").create_dataset("data", data=s[..., :C])
            itr += 1
        if fh is not None:
            fh.close()
        print(f"{fam}: MPP train={itr} in {ci} chunks C={C} -> {fam_dir}/chunk_*.h5", flush=True)
    else:
        raise ValueError(f"unknown TARGET={TARGET}")


if __name__ == "__main__":
    from multiprocessing import Pool
    fams = os.environ.get("FAMILIES", ",".join(FAM_C)).split(",")
    print(f"convert TARGET={TARGET} OUT={OUT} families={fams}", flush=True)
    os.makedirs(OUT, exist_ok=True)

    def _safe(fam):
        try:
            convert_family(fam); return f"{fam} ok"
        except Exception as e:
            import traceback; return f"{fam} ERROR {e}\n{traceback.format_exc()}"

    with Pool(len(fams)) as pool:
        for r in pool.imap_unordered(_safe, fams):
            print("DONE", r, flush=True)
    print("CONVERT DONE", flush=True)
