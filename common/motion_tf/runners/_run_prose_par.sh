#!/bin/bash
# PARALLEL-COPY small-run launcher (1 GPU). bash _run_prose_par.sh <config>
# Same as _run_prose_small.sh but copies the n900 cache from the volume with a THREAD POOL instead of a
# sequential loop. The S3-FUSE mount on capella is latency-bound: a single `cp` stream gets ~0.7 MB/s,
# while the S3 API (parallel/multipart) gets ~11 MB/s on the same volume. Copying many shards
# concurrently should approach the backend bandwidth with ZERO credentials/registry/docker.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
ulimit -u 1000000 2>/dev/null || ulimit -u unlimited 2>/dev/null || true
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4
export TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=2
CONFIG="${1:?config}"
CODE_VOL="${CODE_VOL:-/code-vol}"
SRC="$CODE_VOL/data/prose"; DST=/tmp/work/data/prose
COPY_WORKERS="${COPY_WORKERS:-16}"

mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy 2>/dev/null || true
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

echo "=== PARALLEL copy of the dedicated small (n${N_CAP:-900}) cache (workers=${COPY_WORKERS}) ==="
mkdir -p "$DST/prebuilt"
N_CAP="${N_CAP:-900}" SRC="$SRC" DST="$DST" COPY_WORKERS="$COPY_WORKERS" python - <<'PY'
import os, glob, re, shutil, time
import numpy.lib.format as fmt
from concurrent.futures import ThreadPoolExecutor, as_completed
SRC=os.environ["SRC"]; DST=os.environ["DST"]; NCAP=int(os.environ["N_CAP"]); W=int(os.environ["COPY_WORKERS"])
def nrows(fp):
    with open(fp,"rb") as f:
        fmt.read_magic(f); sh,_,_=fmt.read_array_header_1_0(f); return int(sh[0])
def pick(pattern):                                                    # prefer the exact _n{NCAP}_ cache
    cs=glob.glob(pattern)
    if not cs: return None
    exact=[c for c in cs if re.search(rf"_n{NCAP}_t", os.path.basename(c))]
    if exact: return exact[0]
    return max(cs, key=lambda c:int(re.search(r"_n(\d+)_t",os.path.basename(c)).group(1)))

jobs=[]                                                               # (src_file, dst_dir) to copy
done_markers=[]                                                       # (shard_dir_dst, family) to stamp _DONE after
for fam in ["pdearena_ns","com_ns","incom_ns"]:                       # sharded → head shards up to NCAP
    src=pick(f"{SRC}/prebuilt/{fam}_n*_t*_shards")
    if not src: continue
    dst=os.path.join(DST,"prebuilt",os.path.basename(src)); os.makedirs(dst,exist_ok=True)
    # we don't know per-shard counts without reading headers over FUSE; take enough shards that the
    # cumulative cap is guaranteed — for n900 dirs every shard is wanted, so just take all shards.
    for sh in sorted(glob.glob(os.path.join(src,"shard_*.npy"))):
        jobs.append((sh, dst))
    done_markers.append((dst, fam))
for fam in ["shallow_water","diff_react"]:                            # single .npy → whole file
    src=pick(f"{SRC}/prebuilt/{fam}_n*_t*.npy")
    if src: jobs.append((src, os.path.join(DST,"prebuilt")))

print(f"  {len(jobs)} files to copy with {W} workers", flush=True)
t0=time.time(); n=0; nbytes=0
def cp(job):
    s,d=job; shutil.copy(s,d); return os.path.getsize(s)
with ThreadPoolExecutor(max_workers=W) as ex:
    futs={ex.submit(cp,j):j for j in jobs}
    for f in as_completed(futs):
        sz=f.result(); n+=1; nbytes+=sz
        if n%10==0 or n==len(jobs):
            mb=nbytes/1e6; el=time.time()-t0
            print(f"    copied {n}/{len(jobs)} files, {mb:.0f} MB in {el:.0f}s = {mb/max(el,1):.1f} MB/s", flush=True)
for dst,fam in done_markers:                                          # stamp _DONE (existence is what matters)
    run=sum(nrows(os.path.join(dst,os.path.basename(s))) for s in glob.glob(os.path.join(dst,"shard_*.npy")))
    open(os.path.join(dst,"_DONE"),"w").write(str(run))
    print(f"  {fam}: {run} traj ready", flush=True)
print(f"  PARALLEL COPY DONE: {nbytes/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
PY
export PREBUILT_DIR="$DST/prebuilt"; export PROSE_DATA_DIR="$DST"
echo "cache: $(ls -1 "$DST/prebuilt" 2>/dev/null | tr '\n' ' ')  ($(du -sh "$DST/prebuilt" 2>/dev/null | cut -f1))"

SAVE_DIR=$(python -u -c "import importlib; print(importlib.import_module('$CONFIG').Cfg.save_dir)")
LOCAL="/tmp/work/$SAVE_DIR"; REMOTE="$CODE_VOL/$SAVE_DIR"
mkdir -p "$LOCAL" "$REMOTE"
cp -un "$REMOTE"/ckpt_*.npz "$LOCAL"/ 2>/dev/null || true
( while sleep 120; do cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true; done ) &
SYNC=$!

echo "=== launching $CONFIG (parallel-copy, 1-GPU) ==="
set +e
python -u -m motion_tf.train.train_prose_mgpu "$CONFIG" 2>&1
RC=$?
set -e
kill $SYNC 2>/dev/null||true
cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true
echo "=== done (rc=$RC) ==="; exit $RC
