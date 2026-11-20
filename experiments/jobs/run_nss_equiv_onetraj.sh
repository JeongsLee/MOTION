#!/bin/bash
# REVISION-R1 11-03 (R1 comment 4): MOTION-NS inverse-transformed shallow-water predictions (fr, d1, id_s8; last frame) of the first 16 test trajectories, for the one-trajectory response figure. CPU. Derived from run_nss_equiv.sh.
# six joint-benchmark families (Fig. 4d; CFDBench / PDEArena: admitted row mirror only, K2 ensemble), plus last-frame
# prediction dumps for the contour panels of Supplementary Fig. 13, test-600 MTF split (first 100/family),
# for the replacement of the published MOTION-NS (bylfa) values. Env: RUN, CK, plus the arm's input flags
# (zsc: MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1). eval_equiv.py honours MOTION_BOX_UNIFORM (patched 2026-10-21).
set -e -o pipefail
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export EQUIV_DUMP_MEMBERS=fr,d1,id_s8
export MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_SPLIT_MTF=1 MOTION_BOX_UNIFORM=64
RUN=motion_nss_zsc; CK=160000; export MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1 MOTION_EQUIV_VA1=grid_npy:0
OUT=/work/runs/nss_equiv_onetraj; mkdir -p $OUT
mkdir -p /tmp/corpus/cache /tmp/corpus/raw
ln -s /corpus/meta /tmp/corpus/meta; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled; ln -s /work/data/poseidon /tmp/corpus/raw/poseidon   # vector-axis check refs
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/cache/unified /tmp/corpus/cache/unified
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
C=/corpus/results/$RUN/ckpt_$CK.npz; ls -la $C
env | grep -E "^MOTION_" | tee $OUT/env.txt
python -u eval_equiv.py --ckpt $C --out $OUT/equiv.npz --split test --n 16 --batch 8 --xla 0 \
  --fams shallow_water --members d4xshift --dump 16 2>&1 | grep --line-buffered -v graph_launch | tee $OUT/equiv.log
echo NSS_EQUIV_ONETRAJ_DONE | tee $OUT/done.txt
