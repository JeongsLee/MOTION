#!/bin/bash
# VESSL launcher for a PROSE-FD training config.
#   bash _run_prose.sh motion_tf.train.configs.prose_swe_10M
# Mounts (set in the VESSL job):
#   /code-vol   → pdefoundation-data volume root (has code/ and data/prose/)
# Mirrors code into /tmp/work, installs deps, trains, and SYNCS intermediate ckpts back to
# the volume every 240s so weights persist across preemption.
set -e
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

CONFIG="${1:?usage: bash _run_prose.sh <config-dotted-path>}"
CODE_VOL="${CODE_VOL:-/code-vol}"

mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/ 2>/dev/null || cp -r "$CODE_VOL"/code code/
pip install --quiet numpy h5py sympy 2>/dev/null || true
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

# Copy data to LOCAL disk first — object-storage random h5 reads (900 groups) are slow and
# leave the GPU idle; one sequential copy is fast, then h5py reads from local SSD.
echo "=== copying PROSE data to local disk ==="
mkdir -p /tmp/work/data
cp -r "$CODE_VOL"/data/prose /tmp/work/data/
export PROSE_DATA_DIR=/tmp/work/data/prose
echo "data copied: $(du -sh /tmp/work/data/prose 2>/dev/null)"

SAVE_DIR=$(python -u -c "import importlib,sys; m=importlib.import_module('$CONFIG'); print(m.Cfg.save_dir)")
LOCAL_DIR="/tmp/work/$SAVE_DIR"
REMOTE_DIR="$CODE_VOL/$SAVE_DIR"
mkdir -p "$LOCAL_DIR" "$REMOTE_DIR"
# resume: pull any existing ckpts from the volume
cp -un "$REMOTE_DIR"/ckpt_*.npz "$LOCAL_DIR"/ 2>/dev/null || true
cp -un "$REMOTE_DIR"/state.json "$LOCAL_DIR"/ 2>/dev/null || true

# periodic ckpt sync local → volume
( while sleep 240; do
    cp -u "$LOCAL_DIR"/ckpt_*.npz "$REMOTE_DIR"/ 2>/dev/null || true
    cp -u "$LOCAL_DIR"/*.json "$REMOTE_DIR"/ 2>/dev/null || true
  done ) &
SYNC_PID=$!

echo "=== launching $CONFIG ==="
set +e
python -u -m motion_tf.train.train_prose "$CONFIG" 2>&1
RC=$?
set -e
kill $SYNC_PID 2>/dev/null || true
cp -u "$LOCAL_DIR"/ckpt_*.npz "$REMOTE_DIR"/ 2>/dev/null || true
cp -u "$LOCAL_DIR"/*.json "$REMOTE_DIR"/ 2>/dev/null || true
echo "=== done (rc=$RC) ==="
exit $RC
