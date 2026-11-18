#!/bin/bash
L=${MOTION_LOCAL:-motion_local}; exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp
LOCAL_UNITS=$L/units.pkl systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 python -u latent_compose.py --ckpt $L/m1f_ckpt_best.npz --out $L/out_comp/compose_m1f.npz --n 4 --amps 1,3 --K 10 > $L/out_comp/compose_m1f.log 2>&1
echo "rc=$?" >> $L/out_comp/compose_m1f.log
