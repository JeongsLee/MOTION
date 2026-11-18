#!/bin/bash
# REVISION-R1 (10-22): MOTION-S (paper m1f) equivariance on the full test sets (n per family, default 128) of its equivalence-eligible
# 2D families (FAMS) for the three-weights statistical test (stats_equiv). Env: TAG, FAMS, N, SLOTS (e.g. 3,0,1,4 = Poseidon
# channels for the Poseidon-T comparison; empty = all scored slots), CK (checkpoint, default the paper MOTION-S m1f), XARGS
# (model size flags, e.g. "--d 896 --depth 12 --d_w 48" for MOTION m2s2).
set -e -o pipefail
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 NO_DEAD_REPAIR=1 MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 EQUIV_SCORE_SLOTS=${SLOTS:-}
OUT=/work/runs/equiv_full/${TAG:-m1f_all}; mkdir -p $OUT
mkdir -p /tmp/corpus/cache /tmp/corpus/raw/poseidon
ln -s /corpus/meta /tmp/corpus/meta; ln -s /corpus/cache/unified /tmp/corpus/cache/unified
ln -s /corpus/cache/unified_sdf /tmp/corpus/cache/unified_sdf; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
for p in /work/data/poseidon/*; do b=$(basename $p); [ "$b" = "NS-Gauss" ] || ln -s $p /tmp/corpus/raw/poseidon/$b; done
ln -s /eu/data/poseidon_raw/NS-Gauss /tmp/corpus/raw/poseidon/NS-Gauss
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
env | grep -E "^(MOTION_|EQUIV_)" | tee $OUT/env.txt
python -u eval_equiv.py --ckpt ${CK:-/corpus/results/motion_m1f/ckpt_best.npz} ${XARGS:-} --out $OUT/equiv.npz --split test --n ${N:-128} --batch 8 --xla 1 \
  --fams ${FAMS} 2>&1 | grep --line-buffered -v graph_launch | tee $OUT/equiv.log
echo M1F_FULL_DONE | tee $OUT/done.txt
