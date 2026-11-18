#!/bin/bash
# Local GPU chain 34: incom_ns shear / vortex at MATCHED amplitudes (0.5/1/2 cells, rad 20, flow-map reference):
# zsc, lr1e4 (MTF units) and MOTION-S m1f (its own test units) for a like-for-like comparison.
L=${MOTION_LOCAL:-motion_local}; O=$L/out_inc_ctl3; mkdir -p $O
for i in $(seq 1 80); do [ -s $L/lr1e4/lr1e4_ckpt_160000.npz ] && break; sleep 30; done
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
PY=python; CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
B="TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp"
NSS="MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_BOX_UNIFORM=64"
run() { tag=$1; envs=$2; ck=$3; units=$4
  for k in shear curl; do
    env $B $envs LOCAL_UNITS=$units $CAP $PY -u latent_control3.py --ckpt $ck --fams incom_ns --kinds $k --amps 0.5,1,2 --rad 20 \
      --out $O/c3_${k}_${tag}.npz > $O/c3_${k}_${tag}.log 2>&1
    echo "== $tag $k"; grep -E "^\[incom_ns\] (eta|$k)|Traceback" $O/c3_${k}_${tag}.log | cut -c1-250
  done; }
run zsc "$NSS MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1" $L/zsc/zsc_ckpt_160000.npz $L/zsc/units_nss_mtf.pkl
run lr1e4 "$NSS" $L/lr1e4/lr1e4_ckpt_160000.npz $L/zsc/units_nss_mtf.pkl
run m1f "MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1" $L/m1f_ckpt_best.npz $L/units_ce.pkl
echo CHAIN34_DONE
