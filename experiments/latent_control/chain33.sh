#!/bin/bash
# Local GPU chain 33: uniform latent-velocity command on PDEArena (pdearena_ns, MTF test units), zsc and lr1e4 ckpt_160000.
# PDEArena is not periodic (walls, buoyancy axis): the periodic Fourier shift search is approximate near the walls.
L=${MOTION_LOCAL:-motion_local}; O=$L/out_pda_ctl1; mkdir -p $O
for i in $(seq 1 80); do [ -s $L/lr1e4/lr1e4_ckpt_160000.npz ] && break; sleep 30; done
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 \
  MOTION_BANK_GENERIC_MERGE=1 MOTION_BOX_UNIFORM=64
PY=python; CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"; U=$L/zsc/units_nss_mtf.pkl
MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1 LOCAL_UNITS=$U $CAP $PY -u latent_control_k.py --ckpt $L/zsc/zsc_ckpt_160000.npz --fams pdearena_ns --K 1 \
  --calib_alphas=-4,-2,-1,-0.5,0.5,1,2,4 --alphas=-4,-2,-1,-0.5,0.5,1,2,4 --out $O/control1g_zsc_pdearena_ns.npz > $O/control1g_zsc_pdearena_ns.log 2>&1
grep -E "calib|Traceback" $O/control1g_zsc_pdearena_ns.log
LOCAL_UNITS=$U $CAP $PY -u latent_control_k.py --ckpt $L/lr1e4/lr1e4_ckpt_160000.npz --fams pdearena_ns --K 1 \
  --calib_alphas=-4,-2,-1,-0.5,0.5,1,2,4 --alphas=-4,-2,-1,-0.5,0.5,1,2,4 --out $O/control1g_lr1e4_pdearena_ns.npz > $O/control1g_lr1e4_pdearena_ns.log 2>&1
grep -E "calib|Traceback" $O/control1g_lr1e4_pdearena_ns.log
echo CHAIN33_DONE
