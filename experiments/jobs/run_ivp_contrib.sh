#!/bin/bash
# REVISION-R1 10-27 (user): mechanism-head contribution shares of the pretrained MOTION-IVP (stage-3 option C,
# motion_ivps_gn3w/ckpt_best) on the six Poseidon pretraining families, for D / AUC (figs/align_metrics.py) in
# Supplementary Note 3.2. Same env as the pretraining (run_ivps_s3w.sh): no metadata, 8 generic branches merged, receding horizon on
# all six families -> eval_contrib.py probes the first hop-1 step (model.step) from a 10-frame window at t0 = 0.
# Env: CKPT, TAG, N (trajectories per family, default 32), SPLIT (default test).
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_IC_MAXFRAC=0.95 MOTION_FT_NOMETA=1 \
       MOTION_FORCE_MODE=ss MOTION_SS_WARP=blend MOTION_WARP_LIVE=1 MOTION_INPUT_ZSCORE=1 MOTION_GLOBAL_NORM=1 \
       MOTION_EVAL_L1=1 MOTION_RH_FAMS=CE-RP,CE-KH,CE-CRP,CE-Gauss,NS-Sines,NS-Gauss MOTION_RH_LEADS=1 \
       MOTION_BOX_UNIFORM=64 N_P=32 KF=20
CKPT=${CKPT:-/corpus/results/motion_ivps_gn3w/ckpt_best.npz}
OUT=/work/runs/ivp_mech/${TAG:-gn3w}; mkdir -p $OUT
# data links as in run_ivps_s3w.sh (NS-Gauss from poseidon_raw)
mkdir -p /tmp/corpus/cache /tmp/corpus/raw/poseidon
ln -s /corpus/meta /tmp/corpus/meta
ln -s /corpus/cache/unified /tmp/corpus/cache/unified
ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
for p in /work/data/poseidon/*; do b=$(basename $p); [ "$b" = "NS-Gauss" ] || ln -s $p /tmp/corpus/raw/poseidon/$b; done
ln -s /eu/data/poseidon_raw/NS-Gauss /tmp/corpus/raw/poseidon/NS-Gauss
ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
env | grep -E "^MOTION_|^N_P|^KF" | tee $OUT/env.txt
python -u eval_contrib.py --ckpt $CKPT --out $OUT/contrib.json --split ${SPLIT:-test} --n ${N:-32} --batch 4 \
  --fams CE-RP,CE-CRP,CE-KH,CE-Gauss,NS-Sines,NS-Gauss 2>&1 | grep --line-buffered -vE "graph_launch" | tee $OUT/contrib.log
echo CONTRIB_DONE | tee $OUT/done.txt
