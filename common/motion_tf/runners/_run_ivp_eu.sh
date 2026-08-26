#!/bin/bash
# PHASE-2 IVP CONTINUATION launcher for capella EU (2xH100), warm-started from the 120k ckpt.
#   bash _run_ivp_eu.sh motion_tf.train.configs.poseidon_pretrain_ivp_unified_eu
# Reads DATA + writes CKPTS on the capella-local cluster volume /eu (cephfs -> fast, no cross-region FUSE).
# Code is copied from the (tiny) object-volume mount /code-vol. Job mounts:
#   --object-volume objvol-vhsq8cmcd9z5:/code-vol   (Seoul; code only, tiny reads)
#   --volume        clustervol-buqg4ecoj305:/eu     (capella-local; 488GB data + ckpts)
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
# H100 nodes have many vCPUs; TF/XLA size thread pools to CPU count and MirroredStrategy multiplies that
# per replica -> pthread_create EAGAIN/SIGABRT without these caps (same fix as _run_prose_mgpu.sh).
ulimit -u 1000000 2>/dev/null || ulimit -u unlimited 2>/dev/null || true
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4
export TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=2
echo "ulimit -u = $(ulimit -u)"

CONFIG="${1:-motion_tf.train.configs.poseidon_pretrain_ivp_unified_eu}"
CODE_VOL="${CODE_VOL:-/code-vol}"      # object volume (code)
EU_VOL="${EU_VOL:-/eu}"                # cluster volume (data + results)
export POSE_ASSEMBLED="$EU_VOL/data/poseidon/_assembled"

mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy netCDF4 2>/dev/null || pip install h5py sympy netCDF4 2>&1 | tail -3
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"
echo "=== POSE_ASSEMBLED=$POSE_ASSEMBLED ==="; ls -la "$POSE_ASSEMBLED" 2>/dev/null | head

# init_from (the 120k warm-start ckpt) must be present on the EU volume. The mirror predates ckpt_120000,
# so stage it once from the object volume (/code-vol, cross-region but a single ~433MB copy) if missing.
# Plain cp across already-mounted volumes — no creds, no external tooling.
INIT=$(python -u -c "import importlib;print(getattr(importlib.import_module('$CONFIG').Cfg,'init_from',''))")
if [ -n "$INIT" ] && [ ! -f "$INIT" ]; then
  SRC="$CODE_VOL/results/poseidon_pretrain_ivp_unified/$(basename "$INIT")"
  echo "=== init ckpt missing on EU; staging from $SRC (one-time cross-region copy) ==="; date
  if [ ! -f "$SRC" ]; then echo "FATAL: source ckpt not found: $SRC"; exit 3; fi
  mkdir -p "$(dirname "$INIT")"; cp "$SRC" "$INIT"; date
fi
if [ -n "$INIT" ] && [ ! -f "$INIT" ]; then echo "FATAL: init_from still missing: $INIT"; exit 3; fi
[ -n "$INIT" ] && echo "=== warm-start from $INIT ($(du -h "$INIT" | cut -f1)) ==="

SAVE_DIR=$(python -u -c "import importlib; print(importlib.import_module('$CONFIG').Cfg.save_dir)")
LOCAL="/tmp/work/$SAVE_DIR"; REMOTE="$EU_VOL/$SAVE_DIR"; mkdir -p "$LOCAL" "$REMOTE"
# resume-on-crash: pull any ckpts already saved on the EU volume into the local working dir
cp -un "$REMOTE"/ckpt_*.npz "$LOCAL"/ 2>/dev/null || true
( while sleep 240; do cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true; done ) &
SYNC=$!

echo "=== launching $CONFIG (2-GPU EU continuation) ==="
set +e
python -u -m motion_tf.train.train_ivp "$CONFIG" 2>&1
RC=$?
set -e
kill $SYNC 2>/dev/null || true
cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true
echo "=== done (rc=$RC) ==="; exit $RC
