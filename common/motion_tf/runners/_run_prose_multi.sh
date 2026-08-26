#!/bin/bash
# Multi-family PROSE launcher.  bash _run_prose_multi.sh motion_tf.train.configs.prose_multi_mgpu
# Stages ONLY the needed family data to local SSD: SWE + diffusion-reaction copied whole (single
# files w/ many groups → object-storage random reads are slow → local copy), PDEArena capped to
# ~35 files/subdir (273 files ≈ 190GB total but loader needs only ~29 for 900 samples).
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
ulimit -u 1000000 2>/dev/null || ulimit -u unlimited 2>/dev/null || true
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4
export TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=2
echo "ulimit -u = $(ulimit -u)"
CONFIG="${1:?config}"
CAP="${2:-35}"
CODE_VOL="${CODE_VOL:-/code-vol}"
SRC="$CODE_VOL/data/prose"; DST=/tmp/work/data/prose

mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy 2>/dev/null || true
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

# Stage only the many-small-group single files (SWE, diff-react) to local SSD — these need fast
# random reads. PDEArena is many large contiguous-array files → read from the volume directly
# (copying its ~49GB over object storage is the staging bottleneck). One sequential cp each.
echo "=== staging SWE + diffusion-reaction to local SSD (pdearena read from volume) ==="
mkdir -p "$DST/pdebench/2D/shallow-water" "$DST/pdebench/2D/diffusion-reaction"
cp "$SRC/pdebench/2D/shallow-water/"*.h5 "$DST/pdebench/2D/shallow-water/" 2>/dev/null || true
echo "  SWE staged: $(du -sh "$DST/pdebench/2D/shallow-water" 2>/dev/null | cut -f1)"
cp "$SRC/pdebench/2D/diffusion-reaction/"*.h5 "$DST/pdebench/2D/diffusion-reaction/" 2>/dev/null || true
echo "  diff-react staged: $(du -sh "$DST/pdebench/2D/diffusion-reaction" 2>/dev/null | cut -f1)"
export PROSE_DATA_DIR="$DST"
export PDEARENA_ROOT="$SRC/pdearena"            # read PDEArena directly from the volume mount
echo "staged local: $(du -sh "$DST" 2>/dev/null | cut -f1)  | pdearena<-volume"

SAVE_DIR=$(python -u -c "import importlib; print(importlib.import_module('$CONFIG').Cfg.save_dir)")
LOCAL="/tmp/work/$SAVE_DIR"; REMOTE="$CODE_VOL/$SAVE_DIR"
mkdir -p "$LOCAL" "$REMOTE"
cp -un "$REMOTE"/ckpt_*.npz "$LOCAL"/ 2>/dev/null || true
( while sleep 240; do cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true; done ) &
SYNC=$!

echo "=== launching $CONFIG (multi-family mGPU) ==="
set +e
python -u -m motion_tf.train.train_prose_mgpu "$CONFIG" 2>&1
RC=$?
set -e
kill $SYNC 2>/dev/null||true
cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true
echo "=== done (rc=$RC) ==="; exit $RC
