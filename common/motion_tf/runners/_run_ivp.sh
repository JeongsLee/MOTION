#!/bin/bash
# PHASE-2 IVP training launcher (separate from _run_prose_stream.sh; Phase-1 untouched).
# bash _run_ivp.sh <config>   e.g. motion_tf.train.configs.poseidon_pretrain_ivp
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4
CONFIG="${1:-motion_tf.train.configs.poseidon_pretrain_ivp}"
CODE_VOL="${CODE_VOL:-/code-vol}"
export POSE_ASSEMBLED="$CODE_VOL/data/poseidon/_assembled"
mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy netCDF4 2>/dev/null || pip install h5py sympy netCDF4 2>&1 | tail -3
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"
SAVE_DIR=$(python -u -c "import importlib; print(importlib.import_module('$CONFIG').Cfg.save_dir)")
LOCAL="/tmp/work/$SAVE_DIR"; REMOTE="$CODE_VOL/$SAVE_DIR"; mkdir -p "$LOCAL" "$REMOTE"
cp -un "$REMOTE"/ckpt_*.npz "$LOCAL"/ 2>/dev/null || true
( while sleep 240; do cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true; done ) &
python -u -m motion_tf.train.train_ivp "$CONFIG" 2>&1
cp -u "$LOCAL"/ckpt_*.npz "$REMOTE"/ 2>/dev/null||true; cp -u "$LOCAL"/*.json "$REMOTE"/ 2>/dev/null||true
