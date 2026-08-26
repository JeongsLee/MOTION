#!/bin/bash
# PHASE-1 Task-2 CONTINUATION launcher for capella EU (2xH100), warm-started from the deneb ckpt.
#   bash _run_prose_stream_eu.sh motion_tf.train.configs.prose_150M_fluid5_6x_unified_bcat_eu
# Streams prose data + writes ckpts on the capella-local cluster volume /eu (cephfs -> fast). Code comes
# from the (tiny) object-volume mount /code-vol, which also holds the init ckpt to stage once. Job mounts:
#   --object-volume objvol-vhsq8cmcd9z5:/code-vol     (Seoul; code + init ckpt)
#   --volume        clustervol-buqg4ecoj305:/eu       (capella-local; 245GB prose data + ckpts)
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
ulimit -u 1000000 2>/dev/null || ulimit -u unlimited 2>/dev/null || true
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4
export TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=2
echo "ulimit -u = $(ulimit -u)"
CONFIG="${1:?config}"
CODE_VOL="${CODE_VOL:-/code-vol}"      # object volume (code + init ckpt)
EU_VOL="${EU_VOL:-/eu}"                # cluster volume (data + results)

mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy 2>/dev/null || true
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

# stream from the capella-local mirror (fast cephfs), NOT the cross-region object volume
export PREBUILT_DIR="$EU_VOL/data/prose/prebuilt"
export PROSE_DATA_DIR="$EU_VOL/data/prose"
echo "streaming from EU mirror: $PREBUILT_DIR"; ls "$PREBUILT_DIR" 2>/dev/null | head

# stage the warm-start ckpt onto EU once (one-time cross-region cp over mounted volumes; no creds/tools)
INIT=$(python -u -c "import importlib;print(getattr(importlib.import_module('$CONFIG').Cfg,'init_ckpt',''))")
if [ -n "$INIT" ] && [ ! -f "$INIT" ]; then
  SRC="$CODE_VOL/results/prose_150M_fluid5_6x_unified_bcat/$(basename "$INIT")"
  echo "=== staging init ckpt from $SRC ==="; date
  if [ ! -f "$SRC" ]; then echo "FATAL: source ckpt not found: $SRC"; exit 3; fi
  mkdir -p "$(dirname "$INIT")"; cp "$SRC" "$INIT"; date
fi
[ -n "$INIT" ] && { [ -f "$INIT" ] || { echo "FATAL: init ckpt missing: $INIT"; exit 3; }; echo "=== warm-start from $INIT ==="; }

SAVE_DIR=$(python -u -c "import importlib; print(importlib.import_module('$CONFIG').Cfg.save_dir)")
LOCAL="/tmp/work/$SAVE_DIR"; REMOTE="$EU_VOL/$SAVE_DIR"; mkdir -p "$LOCAL" "$REMOTE"
cp -un "$REMOTE"/ckpt_*.npz "$LOCAL"/ 2>/dev/null || true
( while sleep 240; do cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/train.log "$REMOTE"/ 2>/dev/null||true; done ) &
SYNC=$!

echo "=== launching $CONFIG (STREAMING, EU continuation) ==="
set +e
# tee the full stdout (incl every [EVAL] line) to a LOCAL log, synced to /eu every 240s + at exit, so the
# data-efficacy curve survives even if vessl truncates the live log. RC from PIPESTATUS (python, not tee).
python -u -m motion_tf.train.train_prose_mgpu "$CONFIG" 2>&1 | tee "$LOCAL/train.log"
RC=${PIPESTATUS[0]}
set -e
kill $SYNC 2>/dev/null||true
cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true
cp -u "$LOCAL"/train.log "$REMOTE"/ 2>/dev/null||true
echo "=== done (rc=$RC) ==="; exit $RC
