#!/bin/bash
# Variant of _run_prose_stream_eu.sh for BOUNDED-DISK convergence runs (cfg.ckpt_keep):
#  - the 240s sync loop MIRRORS deletions: remote ckpts pruned by the trainer are removed from /eu too,
#    so the run dir never holds more than ckpt_keep checkpoints on either side.
#  - the legacy "stage init ckpt from /code-vol/results/prose_150M_fluid5_6x_unified_bcat/" fallback is
#    REMOVED: cfg.init_ckpt must already exist on the mounted volume, else FATAL (prevents silently
#    warm-starting from a same-named checkpoint of a different run).
#   bash _run_prose_stream_eu_keep.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_task2_advb2_ext
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
ulimit -u 1000000 2>/dev/null || ulimit -u unlimited 2>/dev/null || true
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4
export TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=2
echo "ulimit -u = $(ulimit -u)"
CONFIG="${1:?config}"
CODE_VOL="${CODE_VOL:-/code-vol}"      # object volume (code)
EU_VOL="${EU_VOL:-/eu}"                # cluster volume (data + results)

mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy 2>/dev/null || true
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

export PREBUILT_DIR="$EU_VOL/data/prose/prebuilt"
export PROSE_DATA_DIR="$EU_VOL/data/prose"
echo "streaming from EU mirror: $PREBUILT_DIR"; ls "$PREBUILT_DIR" 2>/dev/null | head

INIT=$(python -u -c "import importlib;print(getattr(importlib.import_module('$CONFIG').Cfg,'init_ckpt',''))")
if [ -n "$INIT" ]; then
  [ -f "$INIT" ] || { echo "FATAL: init ckpt missing: $INIT"; exit 3; }
  echo "=== warm-start from $INIT ==="
fi

SAVE_DIR=$(python -u -c "import importlib; print(importlib.import_module('$CONFIG').Cfg.save_dir)")
LOCAL="/tmp/work/$SAVE_DIR"; REMOTE="$EU_VOL/$SAVE_DIR"; mkdir -p "$LOCAL" "$REMOTE"
cp -un "$REMOTE"/ckpt_*.npz "$LOCAL"/ 2>/dev/null || true
sync_once() {
  cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null || true
  for f in "$REMOTE"/ckpt_*.npz; do                       # mirror trainer's pruning to /eu
    [ -e "$f" ] || continue
    [ -f "$LOCAL/$(basename "$f")" ] || rm -f "$f"
  done
  cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null || true
  cp -u "$LOCAL"/train.log "$REMOTE"/ 2>/dev/null || true
}
( while sleep 240; do sync_once; done ) &
SYNC=$!

echo "=== launching $CONFIG (STREAMING, EU convergence extension, ckpt_keep) ==="
set +e
python -u -m motion_tf.train.train_prose_mgpu "$CONFIG" 2>&1 | tee "$LOCAL/train.log"
RC=${PIPESTATUS[0]}
set -e
kill $SYNC 2>/dev/null || true
sync_once
echo "=== done (rc=$RC) ==="; exit $RC
