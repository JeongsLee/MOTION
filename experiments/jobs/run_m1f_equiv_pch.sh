#!/bin/bash
# REVISION-R1 (10-22): MOTION-S (paper m1f) equivariance on the six Poseidon pretraining families scored on the Poseidon
# channels only (EQUIV_SCORE_SLOTS=3,0,1,4: rho, u, v, p; CE energy dropped), test split first 32 = the eval20k m1f run
# (A07) and the Poseidon-T test-split run. NS families are unaffected by the mask (reproduction check vs eval20k/m1f).
set -e -o pipefail
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 NO_DEAD_REPAIR=1 MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 EQUIV_SCORE_SLOTS=3,0,1,4
OUT=/work/runs/eval20k/m1f_poseidonch; mkdir -p $OUT
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
python -u eval_equiv.py --ckpt /corpus/results/motion_m1f/ckpt_best.npz --out $OUT/equiv.npz --split test --n 32 --batch 8 --xla 1 \
  --fams CE-RP,CE-CRP,CE-KH,CE-Gauss,NS-Sines,NS-Gauss 2>&1 | grep --line-buffered -v graph_launch | tee $OUT/equiv.log
echo M1F_PCH_DONE | tee $OUT/done.txt
