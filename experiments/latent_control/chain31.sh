#!/bin/bash
# Local GPU chain 31 (zsc = MOTION-NS-S z-score arm, ckpt_160000, MTF test units): uniform advection (K=1), expansion /
# compression (com_ns, continuity reference), shear / vortex (incom_ns, kinematic flow-map reference).
L=${MOTION_LOCAL:-motion_local}; O=$L/out_zsc_ctl1; mkdir -p $O
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 \
  MOTION_BANK_GENERIC_MERGE=1 MOTION_BOX_UNIFORM=64 MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1
PY=python; CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
C=$L/zsc/zsc_ckpt_160000.npz; U=$L/zsc/units_nss_mtf.pkl
for f in com_ns shallow_water; do
  LOCAL_UNITS=$U $CAP $PY -u latent_control_k.py --ckpt $C --fams $f --K 1 --calib_alphas=-4,-2,-1,-0.5,0.5,1,2,4 \
    --alphas=-4,-2,-1,-0.5,0.5,1,2,4 --out $O/control1g_zsc_$f.npz > $O/control1g_zsc_$f.log 2>&1
  grep -E "calib|Traceback" $O/control1g_zsc_$f.log
done
LOCAL_UNITS=$U $CAP $PY -u latent_control3.py --ckpt $C --fams com_ns --kinds div,div- --amps 1,2,4 --rad 20 \
  --out $O/control3dv_zsc_com_ns.npz > $O/control3dv_zsc_com_ns.log 2>&1
grep -E "^\[com_ns\]|Traceback" $O/control3dv_zsc_com_ns.log | sed -E 's/\| k5.*//' | cut -c1-260
for k in shear curl; do
  LOCAL_UNITS=$U $CAP $PY -u latent_control3.py --ckpt $C --fams incom_ns --kinds $k --amps 1,3 --rad 20 \
    --out $O/control3_${k}_zsc_incom_ns.npz > $O/control3_${k}_zsc_incom_ns.log 2>&1
  grep -E "^\[incom_ns\]|Traceback|CONTROL3_OK" $O/control3_${k}_zsc_incom_ns.log | cut -c1-260
done
echo CHAIN31_DONE
