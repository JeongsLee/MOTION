#!/bin/bash
# Cache-based PROSE launcher — NO raw staging.  bash _run_prose_cached.sh <config>
# Requires the per-family native .npy cache built once by `motion_tf.data.prep_multi` (cheap CPU job)
# living at $CODE_VOL/data/prose/prebuilt/. Copies that small cache (~7GB) to local SSD and points
# build_multi at it via PREBUILT_DIR → GPU job starts training in seconds, every launch.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
ulimit -u 1000000 2>/dev/null || ulimit -u unlimited 2>/dev/null || true
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4
export TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=2
echo "ulimit -u = $(ulimit -u)"
CONFIG="${1:?config}"
CODE_VOL="${CODE_VOL:-/code-vol}"
SRC="$CODE_VOL/data/prose"; DST=/tmp/work/data/prose

mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy 2>/dev/null || true
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

echo "=== copying prebuilt cache to local SSD (no raw staging) ==="
mkdir -p "$DST/prebuilt"
# copy the n100000 caches: SWE/DR single .npy + the pdearena_ns shard DIRECTORY (recursive).
cp "$SRC/prebuilt/"shallow_water_n100000_*.npy "$SRC/prebuilt/"diff_react_n100000_*.npy "$DST/prebuilt/" 2>/dev/null || true
cp -r "$SRC/prebuilt/"pdearena_ns_n100000_*_shards "$DST/prebuilt/" 2>/dev/null || true
cp -r "$SRC/prebuilt/"com_ns_n100000_*_shards "$DST/prebuilt/" 2>/dev/null || true
cp -r "$SRC/prebuilt/"incom_ns_n100000_*_shards "$DST/prebuilt/" 2>/dev/null || true
export PREBUILT_DIR="$DST/prebuilt"
# SWE/diff-react are read only via the cache; PDEArena too. PROSE_DATA_DIR set for completeness.
export PROSE_DATA_DIR="$DST"
echo "cache: $(ls -1 "$DST/prebuilt" 2>/dev/null | tr '\n' ' ')  ($(du -sh "$DST/prebuilt" 2>/dev/null | cut -f1))"

SAVE_DIR=$(python -u -c "import importlib; print(importlib.import_module('$CONFIG').Cfg.save_dir)")
LOCAL="/tmp/work/$SAVE_DIR"; REMOTE="$CODE_VOL/$SAVE_DIR"
mkdir -p "$LOCAL" "$REMOTE"
cp -un "$REMOTE"/ckpt_*.npz "$LOCAL"/ 2>/dev/null || true
( while sleep 240; do cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true; done ) &
SYNC=$!

echo "=== launching $CONFIG (cached, multi-family mGPU) ==="
set +e
python -u -m motion_tf.train.train_prose_mgpu "$CONFIG" 2>&1
RC=$?
set -e
kill $SYNC 2>/dev/null||true
cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true
echo "=== done (rc=$RC) ==="; exit $RC
