#!/bin/bash
# REVISION-R1 (10-22; R2-3 / R3-3 cost table): inference time per trajectory and peak memory of the MOTION weights on 1 H100,
# eval_equiv.py EQUIV_TIME_ONLY=1 (identity rollout, warm second pass, model forward only), test split, 32 trajectories.
set -o pipefail
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
OUT=/work/runs/time_infer; mkdir -p $OUT
mkdir -p /tmp/corpus/cache /tmp/corpus/raw/poseidon
ln -s /corpus/meta /tmp/corpus/meta; ln -s /corpus/cache/unified /tmp/corpus/cache/unified
ln -s /corpus/cache/unified_sdf /tmp/corpus/cache/unified_sdf; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
for p in /work/data/poseidon/*; do b=$(basename $p); [ "$b" = "NS-Gauss" ] || ln -s $p /tmp/corpus/raw/poseidon/$b; done
ln -s /eu/data/poseidon_raw/NS-Gauss /tmp/corpus/raw/poseidon/NS-Gauss
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus EQUIV_TIME_ONLY=1
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
nvidia-smi --query-gpu=name --format=csv | tee $OUT/gpu.txt
run() { tag=$1; shift; echo "=== $tag" | tee -a $OUT/time.log
  ( env "$@" python -u eval_equiv.py --out $OUT/$tag.npz --split test --n 32 --batch 8 --xla 1 $XA 2>&1 | grep --line-buffered -E '^\[time|Traceback|Error|params' | tee -a $OUT/time.log ) || true; }
COMMON="MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1"
XA="--ckpt /corpus/results/motion_m1f/ckpt_best.npz --fams shallow_water,com_ns,NS-Sines,CE-RP" run MOTION-S $COMMON
XA="--ckpt /corpus/results/motion_m2s2/ckpt_best.npz --d 896 --depth 12 --d_w 48 --fams shallow_water,com_ns,NS-Sines,CE-RP" run MOTION $COMMON
XA="--ckpt /corpus/results/motion_nss_zsc/ckpt_160000.npz --fams shallow_water,com_ns,incom_ns,pdearena_ns" run MOTION-NS \
  MOTION_WARP_LIVE=1 MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_SPLIT_MTF=1 MOTION_BOX_UNIFORM=64 MOTION_INPUT_ZSCORE=1 MOTION_COND_STATS=1
XA="--ckpt /work/runs/ivps_gn_keep/ckpt_50000.npz --n_p 32 --fams NS-Sines,CE-RP" run MOTION-IVP \
  MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_FT_NOMETA=1 MOTION_FORCE_MODE=ss MOTION_SS_WARP=blend MOTION_WARP_LIVE=1 \
  MOTION_INPUT_ZSCORE=1 MOTION_GLOBAL_NORM=1 MOTION_BOX_UNIFORM=64 EQUIV_T_IN=1 EQUIV_KF=20
echo TIME_INFER_DONE | tee $OUT/done.txt
