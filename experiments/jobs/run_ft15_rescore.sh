#!/bin/bash
# REVISION-R1 (user 2026-10-30): per-trajectory test errors of the fine-tuned MOTION-IVP for paired tests against
# Poseidon-T/B (whose eval_scot_ft15.py JSONs already hold per_traj_fulltraj). Scoring only, no training: the fifteen
# checkpoints behind the main-text values (make_fig4b.py M_AR / M_STEADY), each scored with the environment it was
# trained and scored with (its env.txt on the code volume) and the scoring call of run_ft15_motion_v5.sh, plus
# PER_TRAJ_OUT. The printed medians must reproduce the reported values (checked locally).
# Env: TASKS (default all fifteen), OUTTAG (default ft15_pertraj).
set -e
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
OUT=/work/runs/${OUTTAG:-ft15_pertraj}; mkdir -p $OUT
mkdir -p /tmp/corpus/raw /tmp/corpus/cache /tmp/ftdata/ft
ln -s /corpus/meta /tmp/corpus/meta; ln -s /tmp/ftdata /tmp/corpus/raw/poseidon_ft; ln -s /corpus/results /tmp/corpus/results
export CORPUS_ROOT=/tmp/corpus
cp -r /work/MOTION/motion_tf /tmp/fv2; cd /tmp/fv2
pip install -q "numpy<2" h5py scipy netCDF4 2>&1 | tail -1
# task:tag:KF:K_RELAX:slots:n_test (tag = the run whose test score is reported)
SPEC="NS-PwC:ivps_c_t6:20:10:0,1:240 NS-SVS:ivps_c_t6:20:10:0,1:240 NS-BB:ivps_c_t6:20:10:0,1:240
NS-SL:ivps_c_t6:20:10:0,1:240 NS-Tracer-PwC:ivps_c_t6:20:10:0,1,6:240 FNS-KF:ivps_c_sens_ar:20:10:0,1:240
CE-RPUI:ivps_c_t6:20:10:3,0,1,4:240 CE-RM:ivps_c_sens_ar:20:10:3,0,1,4:130 GCE-RT:ivps_c_sens_ar4:10:10:3,0,1,4:130
Wave-Gauss:ivps_c_sens_ar:15:10:10:240 Wave-Layer:ivps_c_sens_ar:20:10:10:240 ACE:ivps_c_sens_ar:19:10:6:240
Poisson-Gauss:ivps_c_t6k1c:1:1:4:240 Helmholtz:ivps_c_t6k1c:1:1:4:512 SE-AF:ivps_c_t6k1:1:1:3:240"
for E in $SPEC; do
  IFS=: read -r T TAG KF KR SL NT <<< "$E"
  case " ${TASKS:-all} " in *" all "*|*" $T "*) ;; *) continue;; esac
  FB=$T; [ "$T" = NS-Tracer-PwC ] && FB=NS-PwC
  rm -f /tmp/ftdata/ft/*; cp /eu/data/poseidon_ft/ft/$FB.* /tmp/ftdata/ft/
  ENVF=/work/runs/ft15/$TAG/$T/env.txt
  HA=""; [ "$KF" -gt 1 ] && HA="HOP=1"
  GEO=""; [ "$T" = SE-AF ] && GEO="SCORE_EXCLUDE_GEOM=1"
  echo "===== $T ($TAG, KF $KF, K_RELAX $KR, slots $SL, n $NT, ${HA:-steady})" | tee -a $OUT/score.log
  ( set -a; . $ENVF; set +a
    env CKPT=/corpus/results/ft15/$TAG/$T/ckpt_best.npz FAMS=FT-$T KF=$KF K_RELAX=$KR T_IN=10 MOTION_TI_FIXED=1 \
      SCORE_BATCH=16 NEV=$NT SPLIT=test DECODE=ss SCORE_SLOTS=$SL BOX_UNIFORM=64 $HA $GEO \
      PER_TRAJ_OUT=$OUT/$T.json python -u score_ivp_l1.py 2>&1 \
      | grep -E 'EVAL-L1|@final|hop|global norm|per-traj|Error|Traceback' | tee -a $OUT/score.log )
done
echo RESCORE_DONE | tee $OUT/done.txt
