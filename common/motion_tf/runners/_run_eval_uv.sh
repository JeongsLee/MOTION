#!/usr/bin/env bash
# Per-channel (uv) scOT eval for one-or-more (config, ckpt-step) pairs on capella.
# Mirrors _run_ivp_eu env. Args: STEP then CONFIG... ; CKPT pulled from /eu/results/<save_dir>/ckpt_STEP.npz
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=4
EU_VOL="${EU_VOL:-/eu}"; CODE_VOL="${CODE_VOL:-/code-vol}"
export POSE_ASSEMBLED="$EU_VOL/data/poseidon/_assembled"
mkdir -p /tmp/work && cd /tmp/work && cp -r "$CODE_VOL"/code/. code/
pip install --quiet numpy h5py sympy netCDF4 2>/dev/null || pip install h5py sympy netCDF4 2>&1 | tail -2
STEP="$1"; shift
for CFG in "$@"; do
  SD=$(python -u -c "import importlib;print(importlib.import_module('$CFG').Cfg.save_dir)")
  CK="$EU_VOL/$SD/ckpt_${STEP}.npz"
  echo "##### UVEVAL $CFG step=$STEP ck=$CK exists=$([ -f "$CK" ] && echo Y || echo N) #####"
  [ -f "$CK" ] && CKPT="$CK" python -u -m motion_tf.train._eval_scot_metric "$CFG" || echo "  SKIP missing"
done
echo "=== uv eval done ==="
