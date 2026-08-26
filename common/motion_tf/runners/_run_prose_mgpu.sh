#!/bin/bash
# Multi-GPU PROSE launcher.  bash _run_prose_mgpu.sh motion_tf.train.configs.prose_swe_mgpu [subpath]
# Copies ONLY the needed dataset subpath to local SSD (NOT all of data/prose — the download
# jobs fill it with hundreds of GB). Default subpath = SWE.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
# H100 nodes have very many vCPUs; TF/XLA/Eigen size thread pools to CPU count and MirroredStrategy
# multiplies that per replica → pthread_create hits the container's nproc limit (errno 11 EAGAIN,
# SIGABRT). Cap thread pools and raise the process limit.
ulimit -u 1000000 2>/dev/null || ulimit -u unlimited 2>/dev/null || true
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4
export TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=2
echo "ulimit -u = $(ulimit -u)"
CONFIG="${1:?config}"
SUB="${2:-pdebench/2D/shallow-water}"
CODE_VOL="${CODE_VOL:-/code-vol}"

mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy 2>/dev/null || true
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

echo "=== copying $SUB to local ==="
mkdir -p "/tmp/work/data/prose/$SUB"
cp "$CODE_VOL/data/prose/$SUB"/* "/tmp/work/data/prose/$SUB/" 2>/dev/null || true
export PROSE_DATA_DIR=/tmp/work/data/prose
du -sh "/tmp/work/data/prose/$SUB" 2>/dev/null

SAVE_DIR=$(python -u -c "import importlib; print(importlib.import_module('$CONFIG').Cfg.save_dir)")
LOCAL="/tmp/work/$SAVE_DIR"; REMOTE="$CODE_VOL/$SAVE_DIR"
mkdir -p "$LOCAL" "$REMOTE"
cp -un "$REMOTE"/ckpt_*.npz "$LOCAL"/ 2>/dev/null || true
( while sleep 240; do cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true; done ) &
SYNC=$!

echo "=== launching $CONFIG (mGPU) ==="
set +e
python -u -m motion_tf.train.train_prose_mgpu "$CONFIG" 2>&1
RC=$?
set -e
kill $SYNC 2>/dev/null||true
cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true
echo "=== done (rc=$RC) ==="; exit $RC
