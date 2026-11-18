#!/bin/bash
# Combined 20k evaluation for ONE model (revision R1). Env: TAG, CKPT, MODE (stats|equiv|contrib|all),
# N (stats trajectories, default 128), NE (equiv, default 32), NC (contrib, default 16), plus the
# model's MOTION_* variant switches. Paper ckpts additionally need MOTION_LEGACY_VOCAB=1
# MOTION_VORTEX2D_LEGACY=1. Exact restore: NO_DEAD_REPAIR=1.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 NO_DEAD_REPAIR=1
MODE=${MODE:-all}; N=${N:-128}; NE=${NE:-32}; NC=${NC:-16}
OUT=/work/runs/eval20k/$TAG; mkdir -p $OUT
mkdir -p /tmp/corpus/cache /tmp/corpus/raw/poseidon
ln -s /corpus/meta /tmp/corpus/meta; ln -s /corpus/cache/unified /tmp/corpus/cache/unified
ln -s /corpus/cache/unified_sdf /tmp/corpus/cache/unified_sdf; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
# raw/poseidon: everything from the object volume EXCEPT NS-Gauss, which is read from the /eu copy
# (objvol random access stalled >25 min on NS-Gauss; staged by rev-stage-nsgauss 2026-10-15)
for p in /work/data/poseidon/*; do b=$(basename $p); [ "$b" = "NS-Gauss" ] || ln -s $p /tmp/corpus/raw/poseidon/$b; done
ln -s /eu/data/poseidon_raw/NS-Gauss /tmp/corpus/raw/poseidon/NS-Gauss
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
env | grep -E "^MOTION_" | tee $OUT/env.txt || true
HEADS=reaction,diffusion,buoyancy,shear,wave,advection,shock,elliptic,geometry,boundary_layer,dilatation,gradvec,bernoulli,kinematic,vortex,reaction_nl,spectral,coefficient,compressible
# hybrid models: also knock out each generic head and all generic heads together
NG=${MOTION_BANK_NGENERIC:-0}
if [ "$NG" -gt 0 ] && [ "${MOTION_BANK_GENERIC_MERGE:-0}" = 1 ]; then
  HEADS="$HEADS,generic"; NG=0                       # merged hybrid: one generic head
elif [ "$NG" -gt 0 ]; then
  for i in $(seq 0 $((NG-1))); do HEADS="$HEADS,generic$i"; done
  ALLG=$(seq -s+ -f "generic%g" 0 $((NG-1)))
fi
CFG="base,warp_off$(for h in ${HEADS//,/ }; do printf ',ko:%s' $h; done)"
[ "$NG" -gt 0 ] && CFG="$CFG,ko:$ALLG"
if [ "$MODE" = stats ] || [ "$MODE" = all ]; then
  echo "=== stats ($CFG)"
  python -u eval_stats.py --ckpt $CKPT --out $OUT/stats.npz --split test --n $N --batch 8 --batch3d 2 --xla 1 \
    --configs $CFG ${EVAL_ARGS:-} 2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/stats.log
fi
if [ "$MODE" = equiv ] || [ "$MODE" = all ]; then
  echo "=== equiv"
  python -u eval_equiv.py --ckpt $CKPT --out $OUT/equiv.npz --split test --n $NE --batch 8 --xla 1 2>&1 \
    | grep --line-buffered -vE "graph_launch" | tee $OUT/equiv.log
fi
if [ "$MODE" = contrib ] || [ "$MODE" = all ]; then
  echo "=== contrib"
  python -u eval_contrib.py --ckpt $CKPT --out $OUT/contrib.json --n $NC 2>&1 \
    | grep --line-buffered -vE "graph_launch" | tee $OUT/contrib.log
fi
echo EVAL20K_DONE | tee $OUT/done_$MODE.txt
