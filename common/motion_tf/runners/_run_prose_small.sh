#!/bin/bash
# Small-data ARCHITECTURE-ITERATION launcher (1 GPU). bash _run_prose_small.sh <config>
# Copies only the tiny n900 caches for all 5 families (~1-2GB → ~1min, vs ~30min for the full
# n100000 cache) and trains on whatever GPUs are visible (use resourcespec-ch100x1 = 1 GPU, 8x cheaper
# than ch100x8). For fast gate/expert/architecture comparisons — NOT for SOTA numbers.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
ulimit -u 1000000 2>/dev/null || ulimit -u unlimited 2>/dev/null || true
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4
export TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=2
CONFIG="${1:?config}"
CODE_VOL="${CODE_VOL:-/code-vol}"
SRC="$CODE_VOL/data/prose"; DST=/tmp/work/data/prose

mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy 2>/dev/null || true
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

echo "=== copying the dedicated small (n${N_CAP:-900}) cache from the volume ==="
# Prefer the dedicated _n${N_CAP}_ cache over the full _n100000_ cache (old largest() picked n100000 →
# read the 36GB full pdearena shard dir over S3-FUSE). pick() selects the exact n${N_CAP} dir/file when
# it exists. Plain cp/shutil.copy: the FUSE copy is SLOW (can take a while) but it does complete — be
# patient, do NOT cancel on log silence (the per-family line prints only after the whole family copies).
mkdir -p "$DST/prebuilt"
N_CAP="${N_CAP:-900}" SRC="$SRC" DST="$DST" python - <<'PY'
import os, glob, re, shutil
import numpy.lib.format as fmt
SRC=os.environ["SRC"]; DST=os.environ["DST"]; NCAP=int(os.environ["N_CAP"])
def nrows(fp):
    with open(fp,"rb") as f:
        fmt.read_magic(f); sh,_,_=fmt.read_array_header_1_0(f); return int(sh[0])
def pick(pattern):                                                    # prefer the exact _n{NCAP}_ cache
    cs=glob.glob(pattern)
    if not cs: return None
    exact=[c for c in cs if re.search(rf"_n{NCAP}_t", os.path.basename(c))]
    if exact: return exact[0]
    return max(cs, key=lambda c:int(re.search(r"_n(\d+)_t",os.path.basename(c)).group(1)))
for fam in ["pdearena_ns","com_ns","incom_ns"]:                       # sharded → copy head shards
    src=pick(f"{SRC}/prebuilt/{fam}_n*_t*_shards")
    if not src: continue
    dst=os.path.join(DST,"prebuilt",os.path.basename(src)); os.makedirs(dst,exist_ok=True)
    run=0
    for sh in sorted(glob.glob(os.path.join(src,"shard_*.npy"))):
        if run>=NCAP: break
        shutil.copy(sh,dst); run+=nrows(os.path.join(dst,os.path.basename(sh)))
        print(f"    {fam} shard {os.path.basename(sh)} -> {run} traj",flush=True)   # per-shard heartbeat
    open(os.path.join(dst,"_DONE"),"w").write(str(run))
    print(f"  {fam}: copied head shards ({run} traj) from {os.path.basename(src)}",flush=True)
for fam in ["shallow_water","diff_react"]:                           # single .npy (small) → copy whole
    src=pick(f"{SRC}/prebuilt/{fam}_n*_t*.npy")
    if src:
        shutil.copy(src, os.path.join(DST,"prebuilt",os.path.basename(src)))
        print(f"  {fam}: copied {os.path.basename(src)}",flush=True)
PY
export PREBUILT_DIR="$DST/prebuilt"; export PROSE_DATA_DIR="$DST"
echo "cache: $(ls -1 "$DST/prebuilt" 2>/dev/null | tr '\n' ' ')  ($(du -sh "$DST/prebuilt" 2>/dev/null | cut -f1))"

SAVE_DIR=$(python -u -c "import importlib; print(importlib.import_module('$CONFIG').Cfg.save_dir)")
LOCAL="/tmp/work/$SAVE_DIR"; REMOTE="$CODE_VOL/$SAVE_DIR"
mkdir -p "$LOCAL" "$REMOTE"
cp -un "$REMOTE"/ckpt_*.npz "$LOCAL"/ 2>/dev/null || true
( while sleep 120; do cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true; done ) &
SYNC=$!

echo "=== launching $CONFIG (small, 1-GPU) ==="
set +e
python -u -m motion_tf.train.train_prose_mgpu "$CONFIG" 2>&1
RC=$?
set -e
kill $SYNC 2>/dev/null||true
cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true
echo "=== done (rc=$RC) ==="; exit $RC
