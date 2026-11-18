#!/bin/bash
# Local GPU chain 39 (zsc): vortex (curl) and source / sink (div / div-) commands across the NS-S families
# (kinematic flow-map reference; continuity reference for compressible density), matched 0.5/1/2 cells, rad 20.
L=${MOTION_LOCAL:-motion_local}; O=$L/out_zsc_vs; mkdir -p $O
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 \
  MOTION_BANK_GENERIC_MERGE=1 MOTION_BOX_UNIFORM=64 MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1
PY=python; CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
C=$L/zsc/zsc_ckpt_160000.npz; U=$L/zsc/units_nss_mtf.pkl
for spec in "com_ns curl" "pdearena_ns curl" "incom_ns div,div-" "pdearena_ns div,div-"; do
  set -- $spec; fam=$1; k=$2; tag=${k//,/_}
  LOCAL_UNITS=$U $CAP $PY -u latent_control3.py --ckpt $C --fams $fam --kinds $k --amps 0.5,1,2 --rad 20 \
    --out $O/c3_${tag}_${fam}.npz > $O/c3_${tag}_${fam}.log 2>&1
  echo "== $fam $k"; grep -E "^\[$fam\] (eta|curl|div)|Traceback" $O/c3_${tag}_${fam}.log | sed -E 's/\| k5.*//' | cut -c1-230
done
echo CHAIN39_DONE
