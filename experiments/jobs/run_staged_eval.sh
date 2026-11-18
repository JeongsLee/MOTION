#!/bin/bash
# Combined 20k evaluation for ONE model (revision R1). Env: TAG, CKPT, MODE (stats|equiv|contrib|all),
# N (stats trajectories, default 128), NE (equiv, default 32), NC (contrib, default 16), plus the
# model's MOTION_* variant switches. Paper ckpts additionally need MOTION_LEGACY_VOCAB=1
# MOTION_VORTEX2D_LEGACY=1. Exact restore: NO_DEAD_REPAIR=1.
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 NO_DEAD_REPAIR=1
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
cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
# R1-1 staged-loss control: copy training logs and evaluate both 20k checkpoints on the test split (base config)
export MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1
for A in s1 s2; do
  OUT=/work/runs/eval20k/staged_$A; mkdir -p $OUT
  cp -f /corpus/results/staged_m1f_$A/train.log $OUT/train.log 2>/dev/null; ls /corpus/results/staged_m1f_$A > $OUT/ls.txt
  CK=/corpus/results/staged_m1f_$A/ckpt_20000.npz; [ -f $CK ] || CK=/corpus/results/staged_m1f_$A/ckpt_best.npz
  echo "ckpt $CK" | tee $OUT/ckpt.txt
  python -u eval_stats.py --ckpt $CK --out $OUT/stats.npz --split test --n 128 --batch 8 --batch3d 2 --xla 1 --configs base 2>&1 \
    | grep --line-buffered -vE "graph_launch" | tee $OUT/stats.log
done
echo STAGED_EVAL_DONE
