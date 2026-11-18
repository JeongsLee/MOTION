#!/bin/bash
# Local GPU chain 11 (m1f): Fig. 2c fields - NS-Sines vortex (curl) at 1 and 3 std, fields stored (base, model, flow map, solver).
L=${MOTION_LOCAL:-motion_local}; O=$L/out_ctl3v20; mkdir -p $O
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp
PY=python
CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
LOCAL_UNITS=$L/units_ce.pkl $CAP $PY -u latent_control3.py --ckpt $L/m1f_ckpt_best.npz --fams NS-Sines --kinds curl --amps 1,3 --rad 20 \
  --out $O/control3v_m1f_NS-Sines.npz > $O/control3v_m1f_NS-Sines.log 2>&1
grep -E "^\[NS-Sines\]|Traceback|CONTROL3_OK" $O/control3v_m1f_NS-Sines.log | cut -c1-300
