#!/bin/bash
# Local GPU chain 29 (m1f): centred control-volume mass balance of the expansion / compression commands, 4 test traj, amps 1/2/4 (user 10-20).
L=${MOTION_LOCAL:-motion_local}; O=$L/out_ctl3dm; mkdir -p $O
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp
PY=python
CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
for f in CE-Gauss CE-RP com_ns; do
  LOCAL_UNITS=$L/units_ce.pkl $CAP $PY -u latent_control3.py --ckpt $L/m1f_ckpt_best.npz --fams $f --kinds div,div- --amps 1,2,4 --rad 20 \
    --out $O/control3dv_m1f_$f.npz > $O/control3dv_m1f_$f.log 2>&1
  grep -E "^\[$f\]|Traceback" $O/control3dv_m1f_$f.log | sed -E 's/\| k5.*//' | cut -c1-260
done
echo CHAIN29_DONE
