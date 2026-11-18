#!/bin/bash
# REVISION-R1 10-27 (user): mechanism-head contribution shares of the fifteen FINE-TUNED MOTION-IVP checkpoints (main
# rule: hop-1 AR on time-dependent tasks, K=1 relaxation on steady tasks), one task per FT checkpoint, for D / AUC
# across the downstream physics (figs/align_metrics.py). Each task re-exports its FT run's env.txt (subshell).
# Output: /work/runs/ivp_mech/ft15/<task>/contrib.json
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
LIST=${LIST:-"NS-PwC:ivps_c_t6 NS-SVS:ivps_c_t6 NS-BB:ivps_c_t6 NS-SL:ivps_c_t6 NS-Tracer-PwC:ivps_c_t6 CE-RPUI:ivps_c_t6 \
FNS-KF:ivps_c_sens_ar CE-RM:ivps_c_sens_ar GCE-RT:ivps_c_sens_ar4 Wave-Layer:ivps_c_sens_ar Poisson-Gauss:ivps_c_t6k1c \
Helmholtz:ivps_c_t6k1c SE-AF:ivps_c_t6k1 ACE:ivps_c_sens_ar Wave-Gauss:ivps_c_sens_ar"}
mkdir -p /tmp/corpus/raw /tmp/corpus/cache /tmp/ftdata/ft; ln -s /corpus/meta /tmp/corpus/meta
ln -s /tmp/ftdata /tmp/corpus/raw/poseidon_ft; ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
for e in $LIST; do
  T=${e%%:*}; TAG=${e#*:}
  case "$T" in Poisson-Gauss|Helmholtz|SE-AF) KF=1;; Wave-Gauss) KF=15;; ACE) KF=19;; GCE-RT) KF=10;; *) KF=20;; esac
  FB=$T; [ "$T" = NS-Tracer-PwC ] && FB=NS-PwC
  cp -n /eu/data/poseidon_ft/ft/$FB.* /tmp/ftdata/ft/ 2>/dev/null || true
  OUT=/work/runs/ivp_mech/ft15/$T; mkdir -p $OUT
  ( export $(grep -E '^(MOTION_|N_P=)' /work/runs/ft15/$TAG/$T/env.txt | xargs)
    export MOTION_BOX_UNIFORM=64 KF=$KF
    echo "=== $T ($TAG) KF $KF" | tee $OUT/contrib.log
    python -u eval_contrib.py --ckpt /corpus/results/ft15/$TAG/$T/ckpt_best.npz --out $OUT/contrib.json --split test \
      --n 32 --batch 4 --fams FT-$T 2>&1 | grep --line-buffered -vE "graph_launch" | tee -a $OUT/contrib.log ) || echo "FAILED $T"
done
echo FT15_CONTRIB_DONE
