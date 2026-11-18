#!/bin/bash
# Local GPU chain 8 (m1f): ONE-STEP control, eta fitted on A over the SAME command grid used on B (LS slope through 0).
L=${MOTION_LOCAL:-motion_local}; O=$L/out_ctl1g; mkdir -p $O
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp
PY=python
CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
for f in NS-Sines incom_ns com_ns CE-RP CE-Gauss; do
  grep -q CONTROLK_OK $O/control1g_m1f_$f.log 2>/dev/null && { echo "skip $f"; continue; }
  echo "$(date +%T) start $f"
  LOCAL_UNITS=$L/units_ce.pkl $CAP $PY -u latent_control_k.py --ckpt $L/m1f_ckpt_best.npz --fams $f --K 1 \
    --calib_alphas=-4,-2,-1,-0.5,0.5,1,2,4 --alphas=-4,-2,-1,-0.5,0.5,1,2,4 --out $O/control1g_m1f_$f.npz > $O/control1g_m1f_$f.log 2>&1
  echo "$(date +%T) end $f rc=$?"; grep -E "calib" $O/control1g_m1f_$f.log
done
echo CHAIN8_DONE
