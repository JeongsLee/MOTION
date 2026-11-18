#!/bin/bash
# Local GPU chain 18 (m1f): NS-Sines diffusion knob on test trajectory 4 (= the shear/vortex trajectory of Fig. 2c), with the
# amplitude-calibrated factor f = 2.01 (nu x2; elasticity from calibration trajectories 0-3) and the uncalibrated x2.
L=${MOTION_LOCAL:-motion_local}; O=$L/out_knob
exec 9>$L/chain.lock; flock 9
cd ${MOTION_REPO:-.}/motion_tf
export TF_FORCE_GPU_ALLOW_GROWTH=true MOTION_LEGACY_VOCAB=1 MOTION_VORTEX2D_LEGACY=1 NO_DEAD_REPAIR=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CORPUS_ROOT=/tmp
PY=python
CAP="systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0"
# unbiased calibration check: Nelder-Mead start at no change (adv 1, nu 1) instead of the commanded factor; traj 0-4
LOCAL_UNITS=$L/units.pkl $CAP $PY -u latent_knob.py --ckpt $L/m1f_ckpt_best.npz --fam NS-Sines --n 5 --factors 0.75,1.25,1.5,2 --calib 1 \
  --calib_x0 one --only_knob diff --lap_steer_heads diffusion --out $O/x0one_diffonly.npz > $O/x0one_diffonly.log 2>&1
LOCAL_UNITS=$L/units.pkl $CAP $PY -u latent_knob.py --ckpt $L/m1f_ckpt_best.npz --fam NS-Sines --n 5 --factors 1.217 --calib 1 \
  --calib_x0 one --only_knob diff --lap_steer_heads diffusion,coefficient,compressible --out $O/x0one_group.npz > $O/x0one_group.log 2>&1
grep -E "\[calib\]|Traceback|KNOB_OK" $O/x0one_diffonly.log $O/x0one_group.log | cut -c1-200
echo CHAIN23_DONE
