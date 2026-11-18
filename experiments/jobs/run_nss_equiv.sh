#!/bin/bash
# REVISION-R1: equivariance (D4 x periodic shifts, K4/K8/T3/K8xT3 ensembles) of a MOTION-NS-S checkpoint on the
# six joint-benchmark families (Fig. 4c; CFDBench / PDEArena: admitted row mirror only, K2 ensemble), plus last-frame
# prediction dumps for the Fig. 3d contour panel, test-600 MTF split (first 100/family),
# for the replacement of the published MOTION-NS (bylfa) values. Env: RUN, CK, plus the arm's input flags
# (zsc: MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1). eval_equiv.py honours MOTION_BOX_UNIFORM (patched 2026-10-21).
set -e -o pipefail
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_SPLIT_MTF=1 MOTION_BOX_UNIFORM=64
RUN=${RUN:-motion_nss_zsc}; CK=${CK:-160000}
OUT=/work/runs/nss_equiv/${RUN}_${CK}; mkdir -p $OUT
mkdir -p /tmp/corpus/cache /tmp/corpus/raw
ln -s /corpus/meta /tmp/corpus/meta; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled; ln -s /work/data/poseidon /tmp/corpus/raw/poseidon   # vector-axis check refs
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/cache/unified /tmp/corpus/cache/unified
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
C=/corpus/results/$RUN/ckpt_$CK.npz; ls -la $C
env | grep -E "^MOTION_" | tee $OUT/env.txt
python -u eval_equiv.py --ckpt $C --out $OUT/equiv.npz --split test --n ${N:-100} --batch 8 --xla 1 \
  --fams ${FAMS:-shallow_water,com_ns,incom_ns,cfdbench,pdearena_ns,pdearena_uncond} --members d4xshift --dump ${DUMP:-100} 2>&1 | grep --line-buffered -v graph_launch | tee $OUT/equiv.log
echo NSS_EQUIV_DONE | tee $OUT/done.txt
