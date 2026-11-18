#!/bin/bash
# Redundancy vs collapse of the unmasked heads (V1a question): (1) head-field cosine similarity +
# participation ratio (eval_redund.py), (2) pairwise simultaneous knockouts among the unmasked heads
# (eval_stats ko:a+b) vs single knockouts. Env: TAG, CKPT, MOTION_* switches.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 NO_DEAD_REPAIR=1
OUT=/work/runs/eval20k/$TAG; mkdir -p $OUT
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
echo "=== redund"
python -u eval_redund.py --ckpt $CKPT --out $OUT/redund.json --n 8 2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/redund.log
U="geometry spectral gradvec compressible coefficient"; CFG=base
for h in $U; do CFG="$CFG,ko:$h"; done
set -- $U; L=("$@")
for ((i=0;i<${#L[@]};i++)); do for ((j=i+1;j<${#L[@]};j++)); do CFG="$CFG,ko:${L[$i]}+${L[$j]}"; done; done
FAMS=$(python -c "from data.registry import FAMILIES,MOTION_FAMILIES;print(','.join(f for f in MOTION_FAMILIES if FAMILIES[f].K==2))")
echo "=== pairKO ($CFG)"
python -u eval_stats.py --ckpt $CKPT --out $OUT/pairko.npz --split test --n 64 --batch 8 --xla 1 --fams $FAMS --configs $CFG \
  2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/pairko.log
echo REDUND_DONE | tee $OUT/done_redund.txt
