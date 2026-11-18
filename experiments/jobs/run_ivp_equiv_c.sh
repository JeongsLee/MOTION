#!/bin/bash
# REVISION-R1 10-27 (user): equivariance (D4 x periodic shifts) of the final MOTION-IVP (stage-3 option C,
# motion_ivps_gn3w/ckpt_best) on the six Poseidon pretraining families, as main-text Fig. 5d for MOTION-S / Poseidon-T:
# test split, first 128 per family, rho,u,v,p scored (EQUIV_SCORE_SLOTS=3,0,1,4). Inference as scored for this model:
# one observed frame at t = 0 in the 10-frame window of this t_in-10 model (EQUIV_T_IN 10 + MOTION_TI_FIXED 1; EQUIV_T_IN 1
# built a t_in-1 model: encoder shape mismatch at load, identity CE-RP error 49.5%), receding-horizon hop-1 rollout over
# 20 leads (eval_equiv.py pred_rh); NO_DEAD_REPAIR=1 loads the checkpoint exactly.
# REVISION-R1 10-30 (user): SPLIT env (default test); SPLIT=heldout = rows held out by both MOTION-IVP and Poseidon.
set -e -o pipefail
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
export MOTION_BANK_NGENERIC=8 MOTION_BANK_GENERIC_MERGE=1 MOTION_IC_MAXFRAC=0.95 MOTION_FT_NOMETA=1 \
       MOTION_FORCE_MODE=ss MOTION_SS_WARP=blend MOTION_WARP_LIVE=1 MOTION_INPUT_ZSCORE=1 MOTION_GLOBAL_NORM=1 \
       MOTION_EVAL_L1=1 MOTION_RH_FAMS=CE-RP,CE-KH,CE-CRP,CE-Gauss,NS-Sines,NS-Gauss MOTION_RH_LEADS=1 \
       MOTION_BOX_UNIFORM=64 EQUIV_T_IN=10 MOTION_TI_FIXED=1 NO_DEAD_REPAIR=1 EQUIV_KF=20 EQUIV_SCORE_SLOTS=3,0,1,4
C=${CKPT:-/corpus/results/motion_ivps_gn3w/ckpt_best.npz}
# REVISION-R1 10-28: the npz is written to the cluster volume and then copied (the first run's npz never reached the
# object volume although EQUIV_OK was printed)
OUT=/work/runs/ivps_equiv/${TAG:-gn3w_test128}; mkdir -p $OUT
OUTC=/corpus/results/ivps_equiv/${TAG:-gn3w_test128}; mkdir -p $OUTC
mkdir -p /tmp/corpus/cache /tmp/corpus/raw/poseidon
ln -s /corpus/meta /tmp/corpus/meta; ln -s /eu/data/poseidon/_assembled /tmp/corpus/raw/poseidon_assembled
for p in /work/data/poseidon/*; do b=$(basename $p); [ "$b" = "NS-Gauss" ] || ln -s $p /tmp/corpus/raw/poseidon/$b; done
ln -s /eu/data/poseidon_raw/NS-Gauss /tmp/corpus/raw/poseidon/NS-Gauss
ln -s /corpus/raw/pdebench /tmp/corpus/raw/pdebench; ln -s /corpus/cache/unified /tmp/corpus/cache/unified; ln -s /eu/data/prose/prebuilt /tmp/corpus/cache/prose128
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
env | grep -E "^(MOTION_|EQUIV_)" | tee $OUT/env.txt
python -u eval_equiv.py --ckpt $C --out $OUTC/equiv.npz --split ${SPLIT:-test} --n ${N:-128} --batch 8 --xla 1 --n_p 32 \
  --fams ${FAMS:-CE-RP,CE-CRP,CE-KH,CE-Gauss,NS-Sines,NS-Gauss} --members d4xshift 2>&1 | grep --line-buffered -v graph_launch | tee $OUT/equiv.log
cp $OUTC/equiv.npz $OUT/equiv.npz && sync; sleep 120; ls -la $OUT $OUTC
echo IVP_EQUIV_DONE | tee $OUT/done.txt
