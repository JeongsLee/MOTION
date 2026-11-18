#!/bin/bash
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 NO_DEAD_REPAIR=1 MOTION_BANK_GENERIC=1
OUT=/work/runs/eval20k/V1a; mkdir -p $OUT
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
python -u eval_stencil_func.py --ckpt /corpus/results/rev_V1a_s0/ckpt_20000.npz --out $OUT/stencil_func.json --n 4 2>&1 \
  | grep --line-buffered -vE "graph_launch" | tee $OUT/stencil_func.log
