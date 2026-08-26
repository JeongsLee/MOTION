#!/bin/bash
# Dump physical pred-vs-GT tensors for contour figures (Phase-1 Task-1) from a trained ckpt.
# Mirrors _run_prose_stream.sh env. bash _run_dump_contours.sh <config>
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
CONFIG="${1:-motion_tf.train.configs.prose_150M_fluid5_6x}"
CODE_VOL="${CODE_VOL:-/code-vol}"
export PREBUILT_DIR="$CODE_VOL/data/prose/prebuilt"
export PROSE_DATA_DIR="$CODE_VOL/data/prose"
mkdir -p /tmp/work && cd /tmp/work
cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy 2>/dev/null || true
python -u -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"
python -u -m motion_tf.train._dump_contours "$CONFIG" 2>&1
