#!/bin/bash
# Local GPU chain 18 (m1f): NS-Sines diffusion knob on test trajectory 4 (= the shear/vortex trajectory of Fig. 2c), with the
# amplitude-calibrated factor f = 2.01 (nu x2; elasticity from calibration trajectories 0-3) and the uncalibrated x2.
L=${MOTION_LOCAL:-motion_local}; O=$L/out_knob
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp
PY=python
CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
# group steer recalibration: c=.138 -> nu x1.56, c=1 -> x8.0 (power ~1.29) -> c=0.217 for nu x2; traj 0-3 check + traj 4 test
H=diffusion,coefficient,compressible
LOCAL_UNITS=$L/units.pkl $CAP $PY -u latent_knob.py --ckpt $L/m1f_ckpt_best.npz --fam NS-Sines --n 4 --factors 1.217 --calib 1 \
  --lap_steer_heads $H --out $O/lsteerG_cal3.npz > $O/lsteerG_cal3.log 2>&1
LOCAL_UNITS=$L/units.pkl $CAP $PY -u latent_knob.py --ckpt $L/m1f_ckpt_best.npz --fam NS-Sines --n 5 --only_traj 4 --factors 1.217 --calib 1 \
  --lap_steer_heads $H --out $O/lsteerG_t4c.npz > $O/lsteerG_t4c.log 2>&1
grep -E "\[calib\] diff|Traceback|KNOB_OK" $O/lsteerG_cal3.log $O/lsteerG_t4c.log | cut -c1-200
echo CHAIN21_DONE
