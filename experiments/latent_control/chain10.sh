#!/bin/bash
# Local GPU chain 9 (m1f): Fig. 2a illustration - command tuned on the shown trajectory so u_x moves exactly 10 cells in 10 steps.
L=${MOTION_LOCAL:-motion_local}; O=$L/out_ctltune10; mkdir -p $O
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp
PY=python
CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
for f in NS-Sines incom_ns com_ns CE-RP CE-Gauss; do
  LOCAL_UNITS=$L/units_ce.pkl $CAP $PY -u latent_control_k.py --ckpt $L/m1f_ckpt_best.npz --fams $f --K 10 --fig_tune 1 --figD 10 \
    --out $O/tune_m1f_$f.npz > $O/tune_m1f_$f.log 2>&1; grep -E "fig_tune|Traceback" $O/tune_m1f_$f.log
done
echo CHAIN10_DONE
